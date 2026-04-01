from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_answer(text: str, candidates: list[str]) -> str | None:
    text_norm = text.strip().lower()
    text_norm = "".join(ch if ch.isalpha() else " " for ch in text_norm)
    text_norm = " ".join(text_norm.split())
    for candidate in candidates:
        cand_norm = "".join(ch if ch.isalpha() else " " for ch in candidate.lower())
        cand_norm = " ".join(cand_norm.split())
        if cand_norm and cand_norm in text_norm:
            return candidate
    return None


def token_count(tokenizer, text: str) -> int:
    return int(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"].__len__())


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def render_table(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0].keys())
    widths = {key: max(len(key), *(len(str(row[key])) for row in rows)) for key in headers}
    parts = []
    parts.append("| " + " | ".join(f"{key:{widths[key]}}" for key in headers) + " |")
    parts.append("| " + " | ".join("-" * widths[key] for key in headers) + " |")
    for row in rows:
        parts.append("| " + " | ".join(f"{str(row[key]):{widths[key]}}" for key in headers) + " |")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--num-examples", type=int, default=40)
    parser.add_argument("--latent-stage", type=str, default="stage2_purified")
    args = parser.parse_args()

    run_module = load_module(
        Path(__file__).parent / "run_qwen_handoff.py",
        "qwen_handoff_runtime",
    )
    run_module.set_seed(11)

    test_samples = json.loads((args.run_dir / "test.json").read_text())[: args.num_examples]
    metrics = json.loads((args.run_dir / f"{args.latent_stage}_metrics.json").read_text())
    metadata = json.loads((args.run_dir / "run_metadata.json").read_text())

    tokenizer, qwen_model = run_module.load_qwen_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cfg = run_module.DatasetConfig(**metadata["dataset"])
    stage_cfg_data = next(stage for stage in metadata["stages"] if stage["name"] == args.latent_stage)
    stage_cfg = run_module.StageConfig(**stage_cfg_data)
    hidden_dim = 5120
    latent_model = run_module.LatentHandoffModel(hidden_dim, len(run_module.ANSWER_VOCAB), dataset_cfg, stage_cfg).to(device)
    latent_model.load_state_dict(torch.load(args.run_dir / f"{args.latent_stage}.pt", map_location="cpu"))
    latent_model.eval()

    rows = []
    examples = []

    text_correct = 0
    text_sender_sec = 0.0
    text_receiver_sec = 0.0
    text_msg_tokens = 0
    text_msg_bytes = 0

    latent_correct = 0
    latent_sender_sec = 0.0
    latent_receiver_sec = 0.0
    latent_msg_floats = metrics["message_floats"]
    latent_msg_bytes_fp16 = latent_msg_floats * 2

    with torch.no_grad():
        for idx, item in enumerate(test_samples):
            sender_prompt = (
                "You are preparing a handoff note for another model.\n"
                "Read the profile and write one compact factual note that preserves every exact value.\n"
                "Use a single line of semicolon-separated field=value items with these keys only:\n"
                "city; pet; drink; hobby; commute; instrument; project; weekday.\n"
                "Output only the handoff note.\n"
                f"Profile:\n{item['profile_text']}\nHandoff note:"
            )
            sync()
            start = time.perf_counter()
            sender_inputs = tokenizer(sender_prompt, return_tensors="pt").to(qwen_model.device)
            sender_outputs = qwen_model.generate(
                **sender_inputs,
                max_new_tokens=48,
                do_sample=False,
                use_cache=True,
            )
            sync()
            text_sender_sec += time.perf_counter() - start
            handoff_raw = tokenizer.decode(
                sender_outputs[0][sender_inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            handoff = handoff_raw.splitlines()[0].strip()
            text_msg_tokens += token_count(tokenizer, handoff)
            text_msg_bytes += len(handoff.encode("utf-8"))

            receiver_prompt = (
                "You are answering from a handoff note.\n"
                "Return exactly the value from the note and nothing else.\n"
                f"Handoff note: {handoff}\n"
                f"Question: {item['question_text']}\nAnswer:"
            )
            sync()
            start = time.perf_counter()
            receiver_inputs = tokenizer(receiver_prompt, return_tensors="pt").to(qwen_model.device)
            receiver_outputs = qwen_model.generate(
                **receiver_inputs,
                max_new_tokens=6,
                do_sample=False,
                use_cache=True,
            )
            sync()
            text_receiver_sec += time.perf_counter() - start
            text_answer_raw = tokenizer.decode(
                receiver_outputs[0][receiver_inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            text_answer = normalize_answer(text_answer_raw, run_module.ANSWER_VOCAB)
            text_correct += int(text_answer == item["answer_text"])

            doc_text = f"Read the profile carefully.\nProfile:\n{item['profile_text']}"
            question_text = f"Question:\n{item['question_text']}\nAnswer with one short phrase."
            doc_inputs = tokenizer(
                doc_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=dataset_cfg.max_doc_length,
            ).to(qwen_model.device)
            sync()
            start = time.perf_counter()
            doc_outputs = qwen_model(**doc_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            doc_hidden = doc_outputs.hidden_states[-1].to(dtype=torch.float32)
            doc_mask = doc_inputs["attention_mask"]
            slots, _ = latent_model.encode(doc_hidden, doc_mask, training=False)
            sync()
            latent_sender_sec += time.perf_counter() - start

            question_inputs = tokenizer(
                question_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=dataset_cfg.max_question_length,
            ).to(qwen_model.device)
            sync()
            start = time.perf_counter()
            question_outputs = qwen_model(
                **question_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            question_hidden = run_module.last_token_pool(
                question_outputs.hidden_states[-1], question_inputs["attention_mask"]
            ).to(dtype=torch.float32)
            answer_logits = latent_model.receiver(question_hidden, slots)
            sync()
            latent_receiver_sec += time.perf_counter() - start
            latent_pred = run_module.ANSWER_VOCAB[int(answer_logits.argmax(dim=-1).item())]
            latent_correct += int(latent_pred == item["answer_text"])

            if idx < 3:
                examples.append(
                    {
                        "question": item["question_text"],
                        "gold": item["answer_text"],
                        "text_handoff": handoff,
                        "text_pred": text_answer_raw.strip(),
                        "latent_pred": latent_pred,
                    }
                )

    n = len(test_samples)
    comparison = {
        "num_examples": n,
        "text_handoff": {
            "accuracy": text_correct / n,
            "avg_message_tokens": text_msg_tokens / n,
            "avg_message_bytes": text_msg_bytes / n,
            "avg_sender_ms": 1000 * text_sender_sec / n,
            "avg_receiver_ms": 1000 * text_receiver_sec / n,
            "avg_total_ms": 1000 * (text_sender_sec + text_receiver_sec) / n,
        },
        "latent_handoff": {
            "stage": args.latent_stage,
            "accuracy_subset": latent_correct / n,
            "accuracy_full_test": metrics["clean_accuracy"],
            "message_floats": latent_msg_floats,
            "message_bytes_fp16": latent_msg_bytes_fp16,
            "avg_sender_ms": 1000 * latent_sender_sec / n,
            "avg_receiver_ms": 1000 * latent_receiver_sec / n,
            "avg_total_ms": 1000 * (latent_sender_sec + latent_receiver_sec) / n,
        },
        "examples": examples,
    }

    table_rows = [
        {
            "Method": "Text Handoff",
            "Eval N": str(n),
            "Accuracy": f"{comparison['text_handoff']['accuracy']:.3f}",
            "Msg Size": f"{comparison['text_handoff']['avg_message_tokens']:.1f} tok / {comparison['text_handoff']['avg_message_bytes']:.0f} B",
            "Sender ms": f"{comparison['text_handoff']['avg_sender_ms']:.1f}",
            "Receiver ms": f"{comparison['text_handoff']['avg_receiver_ms']:.1f}",
            "Total ms": f"{comparison['text_handoff']['avg_total_ms']:.1f}",
        },
        {
            "Method": f"Latent {args.latent_stage}",
            "Eval N": str(n),
            "Accuracy": f"{comparison['latent_handoff']['accuracy_subset']:.3f}",
            "Msg Size": f"{comparison['latent_handoff']['message_floats']} fp16 / {comparison['latent_handoff']['message_bytes_fp16']} B",
            "Sender ms": f"{comparison['latent_handoff']['avg_sender_ms']:.1f}",
            "Receiver ms": f"{comparison['latent_handoff']['avg_receiver_ms']:.1f}",
            "Total ms": f"{comparison['latent_handoff']['avg_total_ms']:.1f}",
        },
    ]
    comparison["table_markdown"] = render_table(table_rows)

    out_path = args.run_dir / f"text_vs_{args.latent_stage}_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    (args.run_dir / f"text_vs_{args.latent_stage}_comparison.md").write_text(comparison["table_markdown"] + "\n")
    print(comparison["table_markdown"])
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
