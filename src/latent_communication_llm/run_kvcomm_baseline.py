"""KVComm Baseline

Sender computes the full KV cache from the document forward pass and passes it
as prefix to the receiver.  No training required — purely inference-time.

Message size = 40 layers × 2 × 8 KV-heads × seq_len × 128 × 2 bytes
             ≈ 31.5 MB per sample for seq_len=192 (Qwen3-14B bfloat16).

Usage:
    python run_kvcomm_baseline.py \
        --run-dir results/full_run_20260316 \
        --num-examples 40
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def normalize_answer(text: str, candidates: list[str]) -> str | None:
    text_norm = re.sub(r"[^a-z ]+", " ", text.strip().lower())
    text_norm = " ".join(text_norm.split())
    for cand in candidates:
        cand_norm = re.sub(r"[^a-z ]+", " ", cand.lower())
        cand_norm = " ".join(cand_norm.split())
        if cand_norm and cand_norm in text_norm:
            return cand
    return None


def load_qwen(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def greedy_decode(
    model,
    input_ids: torch.Tensor,          # [1, q_len]
    attention_mask: torch.Tensor,     # [1, doc_len + q_len]
    past_key_values,
    max_new_tokens: int = 8,
    eos_token_id: int = 2,
) -> list[int]:
    """Single-sample greedy decode with a pre-computed KV prefix."""
    generated = []
    cur_ids = input_ids
    cur_mask = attention_mask
    cur_past = past_key_values

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=cur_ids,
                attention_mask=cur_mask,
                past_key_values=cur_past,
                use_cache=True,
                return_dict=True,
            )
        next_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_id)
        if next_id == eos_token_id:
            break
        cur_past = out.past_key_values
        cur_ids  = torch.tensor([[next_id]], device=cur_ids.device)
        cur_mask = torch.ones(1, cur_mask.shape[1] + 1, device=cur_mask.device,
                              dtype=cur_mask.dtype)
    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",        type=Path, default=Path("results/full_run_20260316"))
    parser.add_argument("--model-path",     type=str,  default="Qwen/Qwen3-14B")
    parser.add_argument("--num-examples",   type=int,  default=40)
    parser.add_argument("--max-doc-length", type=int,  default=192)
    args = parser.parse_args()

    # ---- load ANSWER_VOCAB from handoff module ----
    spec = importlib.util.spec_from_file_location(
        "qwen_handoff_kv",
        Path(__file__).parent / "run_qwen_handoff.py",
    )
    handoff_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = handoff_mod
    spec.loader.exec_module(handoff_mod)
    ANSWER_VOCAB: list[str] = handoff_mod.ANSWER_VOCAB

    # ---- load test samples ----
    test_path = args.run_dir / "test.json"
    test_samples = json.loads(test_path.read_text())[: args.num_examples]
    print(f"Evaluating KVComm on {len(test_samples)} samples …")

    # ---- load model ----
    print("Loading Qwen3-14B …")
    tokenizer, model = load_qwen(args.model_path)
    eos_id = tokenizer.eos_token_id or 2
    device = next(model.parameters()).device
    print(f"Model loaded, main device: {device}")

    # ---- eval loop ----
    correct = 0
    total = len(test_samples)
    kv_bytes_per_sample = None
    total_sender_ms = 0.0
    total_receiver_ms = 0.0

    for idx, item in enumerate(test_samples):
        # ── SENDER: doc → KV cache ──────────────────────────────────────────
        # Use a prompt that naturally leads into a Q&A continuation so the
        # receiver's question tokens are interpreted as a follow-up, not a
        # new instruction.
        doc_text = (
            "Answer with exactly the short phrase from the profile.\n"
            f"Profile:\n{item['profile_text']}\n"
        )
        doc_inputs = tokenizer(
            doc_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_doc_length,
        ).to(device)
        doc_len = doc_inputs["input_ids"].shape[1]
        doc_mask = doc_inputs["attention_mask"]  # [1, doc_len]

        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            doc_out = model(**doc_inputs, use_cache=True, return_dict=True)
        sync()
        total_sender_ms += 1000 * (time.perf_counter() - t0)
        past_kv = doc_out.past_key_values

        # Compute KV cache byte size (once)
        if kv_bytes_per_sample is None:
            kv_bytes_per_sample = sum(
                t.numel() * t.element_size()
                for layer in past_kv
                for t in layer
            )
            print(f"KV cache: {kv_bytes_per_sample / 1e6:.1f} MB / sample")

        # ── RECEIVER: question + KV prefix → answer ─────────────────────────
        # The KV prefix already contains the profile + instruction context.
        # The question tokens just need to continue naturally.
        q_text = f"Question: {item['question_text']}\nAnswer:"
        q_inputs = tokenizer(q_text, return_tensors="pt").to(device)
        q_ids  = q_inputs["input_ids"]   # [1, q_len]
        q_len  = q_ids.shape[1]

        # Attention mask must cover both doc (past) and question tokens
        full_mask = torch.cat(
            [doc_mask, torch.ones(1, q_len, device=device, dtype=doc_mask.dtype)],
            dim=1,
        )  # [1, doc_len + q_len]

        sync()
        t0 = time.perf_counter()
        gen_ids = greedy_decode(
            model,
            input_ids=q_ids,
            attention_mask=full_mask,
            past_key_values=past_kv,
            max_new_tokens=8,
            eos_token_id=eos_id,
        )
        sync()
        total_receiver_ms += 1000 * (time.perf_counter() - t0)

        answer_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = normalize_answer(answer_text, ANSWER_VOCAB)
        correct += int(pred == item["answer_text"])

        if (idx + 1) % 10 == 0 or idx == 0:
            print(
                f"  [{idx+1:3d}/{total}]  acc={correct/(idx+1):.3f}"
                f"  sender={total_sender_ms/(idx+1):.0f} ms"
                f"  receiver={total_receiver_ms/(idx+1):.0f} ms"
                f"  answer='{answer_text.strip()}' / gt='{item['answer_text']}'"
            )

    accuracy = correct / total
    avg_sender_ms   = total_sender_ms   / total
    avg_receiver_ms = total_receiver_ms / total

    print(f"\nKVComm Acc@{total}: {accuracy:.4f}")
    print(f"Message size : {kv_bytes_per_sample / 1e6:.1f} MB = {kv_bytes_per_sample:,} B")
    print(f"Avg sender   : {avg_sender_ms:.1f} ms")
    print(f"Avg receiver : {avg_receiver_ms:.1f} ms")
    print(f"Avg total    : {avg_sender_ms + avg_receiver_ms:.1f} ms")

    results = {
        "method":           "KVComm",
        "accuracy":         accuracy,
        "num_examples":     total,
        "message_bytes":    kv_bytes_per_sample,
        "message_mb":       round(kv_bytes_per_sample / 1e6, 2),
        "avg_sender_ms":    round(avg_sender_ms,   1),
        "avg_receiver_ms":  round(avg_receiver_ms, 1),
        "avg_total_ms":     round(avg_sender_ms + avg_receiver_ms, 1),
    }
    out_path = args.run_dir / "kvcomm_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
