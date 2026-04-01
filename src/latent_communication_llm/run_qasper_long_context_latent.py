from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qasper = load_module(
    "qasper_long_latent_pilot",
    Path(__file__).parent / "run_qasper_pilot.py",
)
long_ctx = load_module(
    "qasper_long_latent_eval",
    Path(__file__).parent / "run_qasper_long_context_eval.py",
)


@dataclass
class LongLatentConfig:
    output_dir: Path
    data_dir: Path = Path("data/qasper")
    model_path: str = "Qwen/Qwen3-14B"
    train_size: int = 400
    val_size: int = 100
    test_size: int = 100
    distractor_docs: int = 2
    distractor_chars: int = 8000
    max_doc_length: int = 256
    max_question_length: int = 96
    max_choice_length: int = 24
    extractor_batch_size: int = 2
    train_batch_size: int = 16
    chunk_words: int = 64
    chunk_stride: int = 48
    selector_top_k: int = 3
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    qasper.set_seed(seed)


def build_long_context_dataset(cfg: LongLatentConfig) -> tuple[dict[str, list[dict]], dict[str, dict[str, float | int | None]]]:
    base_cfg = long_ctx.LongContextConfig(
        output_dir=cfg.output_dir,
        data_dir=cfg.data_dir,
        model_path=cfg.model_path,
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        eval_examples=cfg.test_size,
        distractor_docs=cfg.distractor_docs,
        distractor_chars=cfg.distractor_chars,
        max_doc_length=cfg.max_doc_length,
        prefix_tokens=cfg.max_doc_length,
        chunk_words=cfg.chunk_words,
        chunk_stride=cfg.chunk_stride,
        selector_top_k=cfg.selector_top_k,
        seed=cfg.seed,
    )
    base_dataset = long_ctx.build_base_dataset(base_cfg)
    paper_pool = long_ctx.build_paper_pool(base_dataset)
    long_dataset = {
        split_name: long_ctx.build_long_context_samples(split_samples, paper_pool, base_cfg)
        for split_name, split_samples in base_dataset.items()
    }
    query_cfg = qasper.QasperConfig(
        sender_mode="query_select",
        max_doc_length=cfg.max_doc_length,
        max_question_length=cfg.max_question_length,
        max_choice_length=cfg.max_choice_length,
        extractor_batch_size=cfg.extractor_batch_size,
        train_batch_size=cfg.train_batch_size,
        chunk_words=cfg.chunk_words,
        chunk_stride=cfg.chunk_stride,
        selector_top_k=cfg.selector_top_k,
        seed=cfg.seed,
    )
    return qasper.attach_sender_views(long_dataset, query_cfg)


def write_summary(
    output_dir: Path,
    cfg: LongLatentConfig,
    sender_stats: dict[str, dict[str, float | int | None]],
    sender_message_stats: dict[str, float],
    question_only: dict[str, float],
    stage_summaries: list[dict[str, float]],
    text_results: dict | None,
) -> None:
    lines = ["# QASPER Long-Context Latent", ""]
    lines.append("## Setup")
    lines.append(f"- Train/val/test sizes: {cfg.train_size}/{cfg.val_size}/{cfg.test_size}")
    lines.append(f"- Added distractor docs per example: {cfg.distractor_docs}")
    lines.append(f"- Distractor chars per doc: {cfg.distractor_chars}")
    lines.append(f"- Sender mode: `query_select`")
    lines.append(f"- Sender chunk config: words={cfg.chunk_words}, stride={cfg.chunk_stride}, top_k={cfg.selector_top_k}")
    lines.append("")
    lines.append("## Sender Stats")
    lines.append(f"- Avg sender bytes on test: {sender_message_stats['avg_sender_bytes']:.1f}")
    lines.append(f"- Avg sender tokens on test: {sender_message_stats['avg_sender_tokens']:.1f}")
    lines.append(f"- Avg full long-context bytes on test: {sender_message_stats['avg_full_doc_bytes']:.1f}")
    lines.append(f"- Avg full long-context tokens on test: {sender_message_stats['avg_full_doc_tokens']:.1f}")
    lines.append(f"- Avg selected chunks on test: {sender_stats['test']['avg_selected_chunks']:.2f}")
    if sender_stats["test"]["avg_oracle_chunk_recall"] is not None:
        lines.append(f"- Avg oracle chunk recall on test: {sender_stats['test']['avg_oracle_chunk_recall']:.4f}")
    lines.append("")
    lines.append("## Accuracy")
    lines.append(f"- Question-only scorer: {question_only['test_accuracy']:.4f}")
    if text_results is not None:
        lines.append(f"- Long full-text generation: {text_results['full_context_accuracy']:.4f}")
        lines.append(f"- Long query-select generation: {text_results['query_select_accuracy']:.4f}")
        lines.append(f"- Long question-only generation: {text_results['question_only_accuracy']:.4f}")
    for stage in stage_summaries:
        lines.append(f"- {stage['name']}: {stage['test_accuracy']:.4f}")
    lines.append("")
    lines.append("## Main Table")
    lines.append("| Method | Test Acc. | Message | Relative Bytes vs Long Full Text |")
    lines.append("| --- | ---: | --- | ---: |")
    if text_results is not None:
        full_bytes = sender_message_stats["avg_full_doc_bytes"]
        query_bytes = sender_message_stats["avg_sender_bytes"]
        lines.append(f"| Qwen Long Question Only | {text_results['question_only_accuracy']:.3f} | 0 B | 0.0% |")
        lines.append(
            f"| Qwen Long Full-Text Handoff | {text_results['full_context_accuracy']:.3f} | "
            f"{sender_message_stats['avg_full_doc_tokens']:.1f} tok / {full_bytes:.1f} B | 100.0% |"
        )
        lines.append(
            f"| Qwen Long Query-Select Handoff | {text_results['query_select_accuracy']:.3f} | "
            f"{sender_message_stats['avg_sender_tokens']:.1f} tok / {query_bytes:.1f} B | "
            f"{100.0 * query_bytes / full_bytes:.1f}% |"
        )
    for stage in stage_summaries:
        full_bytes = sender_message_stats["avg_full_doc_bytes"]
        lines.append(
            f"| {stage['display_name']} | {stage['test_accuracy']:.3f} | "
            f"{stage['message_floats']} fp16 / {stage['message_bytes']} B | "
            f"{100.0 * stage['message_bytes'] / full_bytes:.1f}% |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/qasper"))
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--distractor-docs", type=int, default=2)
    parser.add_argument("--distractor-chars", type=int, default=8000)
    parser.add_argument("--max-doc-length", type=int, default=256)
    parser.add_argument("--extractor-batch-size", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--chunk-words", type=int, default=64)
    parser.add_argument("--chunk-stride", type=int, default=48)
    parser.add_argument("--selector-top-k", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--text-results-path", type=Path, default=Path("results/qasper_long_context_100/generation_metrics.json"))
    args = parser.parse_args()

    cfg = LongLatentConfig(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        model_path=args.model_path,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        distractor_docs=args.distractor_docs,
        distractor_chars=args.distractor_chars,
        max_doc_length=args.max_doc_length,
        extractor_batch_size=args.extractor_batch_size,
        train_batch_size=args.train_batch_size,
        chunk_words=args.chunk_words,
        chunk_stride=args.chunk_stride,
        selector_top_k=args.selector_top_k,
        seed=args.seed,
        device=args.device,
    )

    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, sender_stats = build_long_context_dataset(cfg)
    for split_name, samples in dataset.items():
        (cfg.output_dir / f"{split_name}.json").write_text(json.dumps(samples, indent=2))
    (cfg.output_dir / "sender_stats.json").write_text(json.dumps(sender_stats, indent=2))

    started = time.time()
    tokenizer, qwen_model = qasper.runtime.load_qwen_model(cfg.model_path)
    sender_message_stats = qasper.measure_sender_messages(dataset["test"], tokenizer)
    (cfg.output_dir / "sender_message_stats.json").write_text(json.dumps(sender_message_stats, indent=2))

    feature_cfg = qasper.QasperConfig(
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        sender_mode="query_select",
        max_doc_length=cfg.max_doc_length,
        extractor_batch_size=cfg.extractor_batch_size,
        train_batch_size=cfg.train_batch_size,
        chunk_words=cfg.chunk_words,
        chunk_stride=cfg.chunk_stride,
        selector_top_k=cfg.selector_top_k,
        seed=cfg.seed,
    )
    feature_splits = {}
    for split_name, samples in dataset.items():
        feature_splits[split_name] = qasper.extract_features(samples, feature_cfg, tokenizer, qwen_model, split_name)
        torch.save(feature_splits[split_name], cfg.output_dir / f"{split_name}_features.pt")

    hidden_dim = int(feature_splits["train"]["question_hidden"].size(-1))
    del qwen_model
    torch.cuda.empty_cache()

    device = torch.device(cfg.device)
    question_only = qasper.train_question_only(feature_splits["train"], feature_splits["validation"], feature_splits["test"], device)
    (cfg.output_dir / "question_only_metrics.json").write_text(json.dumps(question_only, indent=2))

    train_loader = DataLoader(qasper.build_tensor_dataset(feature_splits["train"]), batch_size=cfg.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(qasper.build_tensor_dataset(feature_splits["validation"]), batch_size=cfg.train_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(qasper.build_tensor_dataset(feature_splits["test"]), batch_size=cfg.train_batch_size, shuffle=False, num_workers=0)

    teacher = None
    stage_summaries = []
    for stage_cfg in qasper.build_stages():
        model, best_snapshot = qasper.train_latent_stage(stage_cfg, train_loader, val_loader, device, hidden_dim, teacher)
        test_metrics = qasper.evaluate_latent(model, test_loader, device)
        summary = {
            "name": stage_cfg.name,
            "display_name": stage_cfg.name.replace("qasper_", "Long ").replace("_", " ").title(),
            "message_floats": stage_cfg.num_slots * stage_cfg.slot_dim,
            "message_bytes": 2 * stage_cfg.num_slots * stage_cfg.slot_dim,
            "best_epoch": best_snapshot["epoch"],
            "best_val_accuracy": best_snapshot["val_accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_avg_kl_rate": test_metrics["avg_kl_rate"],
        }
        stage_summaries.append(summary)
        (cfg.output_dir / f"{stage_cfg.name}_metrics.json").write_text(json.dumps(summary, indent=2))
        torch.save(model.state_dict(), cfg.output_dir / f"{stage_cfg.name}.pt")
        teacher = model
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    text_results = None
    if args.text_results_path.exists():
        text_results = json.loads(args.text_results_path.read_text())

    (cfg.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "sender_stats": sender_stats,
                "sender_message_stats": sender_message_stats,
                "question_only_metrics": question_only,
                "stage_summaries": stage_summaries,
                "text_results_path": str(args.text_results_path) if args.text_results_path.exists() else None,
                "runtime_seconds": time.time() - started,
            },
            indent=2,
            default=str,
        )
    )
    write_summary(cfg.output_dir, cfg, sender_stats, sender_message_stats, question_only, stage_summaries, text_results)
    print(f"Results written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
