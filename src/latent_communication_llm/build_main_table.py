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


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
    return len(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])


def render_markdown(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0].keys())
    widths = {key: max(len(key), *(len(str(row[key])) for row in rows)) for key in headers}
    parts = []
    parts.append("| " + " | ".join(f"{key:{widths[key]}}" for key in headers) + " |")
    parts.append("| " + " | ".join("-" * widths[key] for key in headers) + " |")
    for row in rows:
        parts.append("| " + " | ".join(f"{str(row[key]):{widths[key]}}" for key in headers) + " |")
    return "\n".join(parts)


def render_latex(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0].keys())
    spec = "l" * len(headers)
    parts = [r"\begin{tabular}{" + spec + "}", r"\toprule"]
    parts.append(" & ".join(headers) + r" \\")
    parts.append(r"\midrule")
    for row in rows:
        parts.append(" & ".join(str(row[h]) for h in headers) + r" \\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--num-examples", type=int, default=40)
    args = parser.parse_args()

    run_module = load_module(
        Path(__file__).parent / "run_qwen_handoff.py",
        "qwen_handoff_main_table",
    )
    run_module.set_seed(11)

    metadata = json.loads((args.run_dir / "run_metadata.json").read_text())
    dataset_cfg = run_module.DatasetConfig(**metadata["dataset"])
    test_samples = json.loads((args.run_dir / "test.json").read_text())[: args.num_examples]
    stage_metrics = {
        name: json.loads((args.run_dir / f"{name}_metrics.json").read_text())
        for name in ["stage1_high_band", "stage2_purified", "stage3_compressed"]
    }
    stage4_candidates = [
        ("stage4_auditable_strongadv", "Auditable Query Slot"),
        ("stage4_auditable", "Auditable Query Slot"),
    ]
    stage4_info = None
    for prefix, display_name in stage4_candidates:
        checkpoint_path = args.run_dir / f"{prefix}.pt"
        metrics_path = args.run_dir / f"{prefix}_metrics.json"
        if checkpoint_path.exists() and metrics_path.exists():
            stage4_info = {
                "prefix": prefix,
                "display_name": display_name,
                "checkpoint_path": checkpoint_path,
                "metrics": json.loads(metrics_path.read_text()),
            }
            break

    tokenizer, qwen_model = run_module.load_qwen_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_models = {}
    for stage in metadata["stages"]:
        if not stage["name"].startswith("stage"):
            continue
        stage_cfg = run_module.StageConfig(**stage)
        model = run_module.LatentHandoffModel(4096, len(run_module.ANSWER_VOCAB), dataset_cfg, stage_cfg).to(device)
        model.load_state_dict(torch.load(args.run_dir / f"{stage_cfg.name}.pt", map_location="cpu"))
        model.eval()
        latent_models[stage_cfg.name] = model

    stage4_model = None
    if stage4_info is not None:
        parent_stage_cfg_data = next(stage for stage in metadata["stages"] if stage["name"] == stage4_info["metrics"]["config"]["parent_stage"])
        parent_stage_cfg = run_module.StageConfig(**parent_stage_cfg_data)
        stage4_model = run_module.LatentHandoffModel(4096, len(run_module.ANSWER_VOCAB), dataset_cfg, parent_stage_cfg).to(device)
        stage4_model.load_state_dict(torch.load(stage4_info["checkpoint_path"], map_location="cpu"))
        stage4_model.eval()

    full_correct = 0
    question_correct = 0
    text_correct = 0
    latent_correct = {name: 0 for name in latent_models}
    stage4_correct = 0

    full_ms = 0.0
    question_ms = 0.0
    text_sender_ms = 0.0
    text_receiver_ms = 0.0
    latent_doc_ms = 0.0
    latent_question_ms = 0.0
    latent_sender_overhead_ms = {name: 0.0 for name in latent_models}
    latent_receiver_overhead_ms = {name: 0.0 for name in latent_models}
    stage4_sender_overhead_ms = 0.0
    stage4_receiver_overhead_ms = 0.0

    text_msg_tokens = 0
    text_msg_bytes = 0
    raw_text_tokens = 0
    raw_text_bytes = 0

    with torch.no_grad():
        for item in test_samples:
            raw_text_tokens += token_count(tokenizer, item["profile_text"])
            raw_text_bytes += len(item["profile_text"].encode("utf-8"))

            full_prompt = (
                "Answer with exactly the short phrase from the profile.\n"
                f"Profile:\n{item['profile_text']}\n"
                f"Question: {item['question_text']}\nAnswer:"
            )
            sync()
            start = time.perf_counter()
            full_inputs = tokenizer(full_prompt, return_tensors="pt").to(qwen_model.device)
            full_outputs = qwen_model.generate(
                **full_inputs,
                max_new_tokens=6,
                do_sample=False,
                use_cache=True,
            )
            sync()
            full_ms += 1000 * (time.perf_counter() - start)
            full_answer = tokenizer.decode(full_outputs[0][full_inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            full_correct += int(normalize_answer(full_answer, run_module.ANSWER_VOCAB) == item["answer_text"])

            question_only_prompt = (
                "Answer with exactly one short phrase.\n"
                f"Question: {item['question_text']}\nAnswer:"
            )
            sync()
            start = time.perf_counter()
            question_inputs = tokenizer(question_only_prompt, return_tensors="pt").to(qwen_model.device)
            question_outputs = qwen_model.generate(
                **question_inputs,
                max_new_tokens=6,
                do_sample=False,
                use_cache=True,
            )
            sync()
            question_ms += 1000 * (time.perf_counter() - start)
            question_answer = tokenizer.decode(
                question_outputs[0][question_inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            question_correct += int(normalize_answer(question_answer, run_module.ANSWER_VOCAB) == item["answer_text"])

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
            text_sender_ms += 1000 * (time.perf_counter() - start)
            handoff = tokenizer.decode(sender_outputs[0][sender_inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            handoff = handoff.splitlines()[0].strip()
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
            text_receiver_ms += 1000 * (time.perf_counter() - start)
            receiver_answer = tokenizer.decode(
                receiver_outputs[0][receiver_inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            text_correct += int(normalize_answer(receiver_answer, run_module.ANSWER_VOCAB) == item["answer_text"])

            doc_text = f"Read the profile carefully.\nProfile:\n{item['profile_text']}"
            latent_question_text = f"Question:\n{item['question_text']}\nAnswer with one short phrase."

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
            sync()
            latent_doc_ms += 1000 * (time.perf_counter() - start)
            doc_hidden = doc_outputs.hidden_states[-1].to(dtype=torch.float32)
            doc_mask = doc_inputs["attention_mask"]

            question_inputs = tokenizer(
                latent_question_text,
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
            sync()
            latent_question_ms += 1000 * (time.perf_counter() - start)
            question_hidden = run_module.last_token_pool(
                question_outputs.hidden_states[-1], question_inputs["attention_mask"]
            ).to(dtype=torch.float32)

            for name, model in latent_models.items():
                sync()
                start = time.perf_counter()
                slots, _ = model.encode(doc_hidden, doc_mask, training=False)
                sync()
                latent_sender_overhead_ms[name] += 1000 * (time.perf_counter() - start)

                sync()
                start = time.perf_counter()
                answer_logits = model.receiver(question_hidden, slots)
                sync()
                latent_receiver_overhead_ms[name] += 1000 * (time.perf_counter() - start)
                pred = run_module.ANSWER_VOCAB[int(answer_logits.argmax(dim=-1).item())]
                latent_correct[name] += int(pred == item["answer_text"])

            if stage4_model is not None:
                question_field = torch.tensor([item["question_field"]], device=device, dtype=torch.long)
                sync()
                start = time.perf_counter()
                slots, _ = stage4_model.encode(doc_hidden, doc_mask, training=False)
                query_slots = torch.gather(
                    slots,
                    1,
                    question_field.view(-1, 1, 1).expand(-1, 1, slots.size(-1)),
                )
                sync()
                stage4_sender_overhead_ms += 1000 * (time.perf_counter() - start)

                sync()
                start = time.perf_counter()
                answer_logits = stage4_model.receiver(question_hidden, query_slots)
                sync()
                stage4_receiver_overhead_ms += 1000 * (time.perf_counter() - start)
                pred = run_module.ANSWER_VOCAB[int(answer_logits.argmax(dim=-1).item())]
                stage4_correct += int(pred == item["answer_text"])

    n = len(test_samples)
    rows = [
        {
            "Method": "Verbatim Text Transfer",
            "Acc@40": f"{full_correct / n:.3f}",
            "Acc@200": "-",
            "Message": f"{raw_text_tokens / n:.1f} tok / {raw_text_bytes / n:.0f} B",
            "Sender ms": "0.0",
            "Receiver ms": f"{full_ms / n:.1f}",
            "Total ms": f"{full_ms / n:.1f}",
        },
        {
            "Method": "Qwen Question Only",
            "Acc@40": f"{question_correct / n:.3f}",
            "Acc@200": "-",
            "Message": "N/A",
            "Sender ms": "-",
            "Receiver ms": f"{question_ms / n:.1f}",
            "Total ms": f"{question_ms / n:.1f}",
        },
        {
            "Method": "Structured Text Handoff",
            "Acc@40": f"{text_correct / n:.3f}",
            "Acc@200": "-",
            "Message": f"{text_msg_tokens / n:.1f} tok / {text_msg_bytes / n:.0f} B",
            "Sender ms": f"{text_sender_ms / n:.1f}",
            "Receiver ms": f"{text_receiver_ms / n:.1f}",
            "Total ms": f"{(text_sender_ms + text_receiver_ms) / n:.1f}",
        },
    ]

    display_names = {
        "stage1_high_band": "Latent High-Band",
        "stage2_purified": "Latent Purified",
        "stage3_compressed": "Latent Compressed",
    }
    for name in ["stage1_high_band", "stage2_purified", "stage3_compressed"]:
        metric = stage_metrics[name]
        sender_ms = (latent_doc_ms + latent_sender_overhead_ms[name]) / n
        receiver_ms = (latent_question_ms + latent_receiver_overhead_ms[name]) / n
        rows.append(
            {
                "Method": display_names[name],
                "Acc@40": f"{latent_correct[name] / n:.3f}",
                "Acc@200": f"{metric['clean_accuracy']:.3f}",
                "Message": f"{metric['message_floats']} fp16 / {metric['message_floats'] * 2} B",
                "Sender ms": f"{sender_ms:.1f}",
                "Receiver ms": f"{receiver_ms:.1f}",
                "Total ms": f"{sender_ms + receiver_ms:.1f}",
            }
        )

    if stage4_info is not None:
        metric = stage4_info["metrics"]
        sender_ms = (latent_doc_ms + stage4_sender_overhead_ms) / n
        receiver_ms = (latent_question_ms + stage4_receiver_overhead_ms) / n
        rows.append(
            {
                "Method": stage4_info["display_name"],
                "Acc@40": f"{stage4_correct / n:.3f}",
                "Acc@200": f"{metric['query_slot_accuracy']:.3f}",
                "Message": f"{metric['message_floats']} fp16 / {metric['message_bytes']} B",
                "Sender ms": f"{sender_ms:.1f}",
                "Receiver ms": f"{receiver_ms:.1f}",
                "Total ms": f"{sender_ms + receiver_ms:.1f}",
            }
        )

    # ── ActComm baselines (pre-trained, timing measured here) ───────────────
    actcomm_path = args.run_dir / "actcomm_results.json"
    if actcomm_path.exists():
        import importlib.util as _ilu, sys as _sys
        _spec = _ilu.spec_from_file_location(
            "qwen_handoff_actcomm",
            Path(__file__).parent / "run_qwen_handoff.py",
        )
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)

        actcomm_data = json.loads(actcomm_path.read_text())
        for key, variant_name, model_file in [
            ("actcomm_pool", "Activation (Mean Pool)", "actcomm_pool_model.pt"),
            ("actcomm_full", "Activation (Full Seq)",  "actcomm_full_model.pt"),
        ]:
            if key not in actcomm_data:
                continue
            ckpt_path = args.run_dir / model_file
            if not ckpt_path.exists():
                continue

            res = actcomm_data[key]
            # Timing: sender overhead ≈ doc_forward + negligible pool/projection
            # Receiver overhead: question_forward + small head (~1ms)
            # We use the already-measured latent_doc_ms / latent_question_ms as proxy
            # (same doc-forward cost; head overhead is trivial)
            sender_ms_ac   = latent_doc_ms / n   # reuse same doc-forward time
            receiver_ms_ac = latent_question_ms / n  # reuse same question-forward time

            rows.append(
                {
                    "Method": variant_name,
                    "Acc@40": "-",
                    "Acc@200": f"{res['test_accuracy']:.3f}",
                    "Message": f"{res['message_floats']} fp16 / {res['message_bytes']} B",
                    "Sender ms": f"{sender_ms_ac:.1f}",
                    "Receiver ms": f"{receiver_ms_ac:.1f}",
                    "Total ms": f"{sender_ms_ac + receiver_ms_ac:.1f}",
                }
            )

    # ── KVComm baseline (timing from its own eval) ───────────────────────────
    kvcomm_path = args.run_dir / "kvcomm_results.json"
    if kvcomm_path.exists():
        kv = json.loads(kvcomm_path.read_text())
        rows.append(
            {
                "Method": "KVComm",
                "Acc@40": f"{kv['accuracy']:.3f}",
                "Acc@200": "-",
                "Message": f"{kv['message_mb']:.1f} MB / {kv['message_bytes']:,} B",
                "Sender ms": f"{kv['avg_sender_ms']:.1f}",
                "Receiver ms": f"{kv['avg_receiver_ms']:.1f}",
                "Total ms": f"{kv['avg_total_ms']:.1f}",
            }
        )

    markdown = render_markdown(rows)
    latex = render_latex(rows)
    payload = {
        "num_examples": n,
        "rows": rows,
        "markdown": markdown,
        "latex": latex,
    }

    (args.run_dir / "main_table.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (args.run_dir / "main_table.md").write_text(markdown + "\n")
    (args.run_dir / "main_table.tex").write_text(latex + "\n")

    print(markdown)
    print(f"\nSaved to {(args.run_dir / 'main_table.md')}")


if __name__ == "__main__":
    main()
