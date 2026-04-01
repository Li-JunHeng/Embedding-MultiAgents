from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def load_qasper_module():
    path = Path(__file__).parent / "run_qasper_pilot.py"
    spec = importlib.util.spec_from_file_location("qasper_long_context", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qasper = load_qasper_module()


@dataclass
class LongContextConfig:
    output_dir: Path
    data_dir: Path = Path("data/qasper")
    model_path: str = "Qwen/Qwen3-14B"
    train_size: int = 400
    val_size: int = 100
    test_size: int = 100
    eval_examples: int = 100
    distractor_docs: int = 2
    distractor_chars: int = 8000
    max_doc_length: int = 256
    prefix_tokens: int = 256
    chunk_words: int = 64
    chunk_stride: int = 48
    selector_top_k: int = 3
    seed: int = 13


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_base_dataset(cfg: LongContextConfig) -> dict[str, list[dict]]:
    dataset_cfg = qasper.QasperConfig(
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        sender_mode="query_select",
        max_doc_length=cfg.max_doc_length,
        chunk_words=cfg.chunk_words,
        chunk_stride=cfg.chunk_stride,
        selector_top_k=cfg.selector_top_k,
        generation_eval_examples=cfg.eval_examples,
        seed=cfg.seed,
    )
    dataset, _ = qasper.build_dataset(dataset_cfg, cfg.data_dir)
    return dataset


def build_paper_pool(dataset: dict[str, list[dict]]) -> dict[str, str]:
    pool = {}
    for split_samples in dataset.values():
        for sample in split_samples:
            pool.setdefault(sample["paper_id"], sample["doc_text"])
    return pool


def build_long_context_samples(samples: list[dict], paper_pool: dict[str, str], cfg: LongContextConfig) -> list[dict]:
    paper_ids = sorted(paper_pool.keys())
    out = []
    for sample_idx, sample in enumerate(samples):
        rng = random.Random(cfg.seed + sample_idx)
        candidate_ids = [paper_id for paper_id in paper_ids if paper_id != sample["paper_id"]]
        chosen_ids = rng.sample(candidate_ids, k=min(cfg.distractor_docs, len(candidate_ids)))

        blocks = [sample["doc_text"]]
        for distractor_id in chosen_ids:
            distractor_text = paper_pool[distractor_id][: cfg.distractor_chars]
            if distractor_text:
                blocks.append(distractor_text)
        rng.shuffle(blocks)

        parts = []
        for doc_idx, block in enumerate(blocks, start=1):
            parts.append(f"Document {doc_idx}:\n{block}")

        out.append(
            {
                **sample,
                "doc_text": "\n\n".join(parts),
                "long_context_stats": {
                    "num_documents": len(blocks),
                    "num_distractors": len(blocks) - 1,
                },
            }
        )
    return out


def build_prefix_sender_samples(samples: list[dict], tokenizer, prefix_tokens: int) -> list[dict]:
    out = []
    for sample in samples:
        token_ids = tokenizer.encode(sample["doc_text"], add_special_tokens=False)[:prefix_tokens]
        sender_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        sender_stats = {
            "mode": "prefix_tokens",
            "prefix_tokens": prefix_tokens,
            "bytes": len(sender_text.encode("utf-8")),
        }
        out.append({**sample, "sender_text": sender_text, "sender_stats": sender_stats})
    return out


@torch.no_grad()
def generate_letter(prompt: str, tokenizer, model, num_choices: int) -> int | None:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
    )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip().upper()
    for letter_idx in range(num_choices):
        if chr(65 + letter_idx) in text:
            return letter_idx
    return None


@torch.no_grad()
def run_long_context_generation(
    full_samples: list[dict],
    prefix_samples: list[dict],
    query_samples: list[dict],
    tokenizer,
    model,
    num_choices: int,
) -> dict[str, float]:
    total = len(full_samples)
    full_correct = 0
    prefix_correct = 0
    query_correct = 0
    question_correct = 0

    for idx, (full_sample, prefix_sample, query_sample) in enumerate(
        zip(full_samples, prefix_samples, query_samples, strict=True),
        start=1,
    ):
        option_block = full_sample["options_text"]
        full_prompt = (
            "Read the documents and answer the multiple-choice question.\n"
            "Output only the option letter.\n"
            f"Documents:\n{full_sample['doc_text']}\n"
            f"Question: {full_sample['question_text']}\n"
            f"Options:\n{option_block}\nAnswer:"
        )
        prefix_prompt = (
            "Read the document prefix and answer the multiple-choice question.\n"
            "Output only the option letter.\n"
            f"Document prefix:\n{prefix_sample['sender_text']}\n"
            f"Question: {prefix_sample['question_text']}\n"
            f"Options:\n{option_block}\nAnswer:"
        )
        query_prompt = (
            "Read the selected document excerpts and answer the multiple-choice question.\n"
            "Output only the option letter.\n"
            f"Selected excerpts:\n{query_sample['sender_text']}\n"
            f"Question: {query_sample['question_text']}\n"
            f"Options:\n{option_block}\nAnswer:"
        )
        question_prompt = (
            "Answer the multiple-choice question.\n"
            "Output only the option letter.\n"
            f"Question: {full_sample['question_text']}\n"
            f"Options:\n{option_block}\nAnswer:"
        )

        full_correct += int(generate_letter(full_prompt, tokenizer, model, num_choices) == full_sample["correct_index"])
        prefix_correct += int(generate_letter(prefix_prompt, tokenizer, model, num_choices) == full_sample["correct_index"])
        query_correct += int(generate_letter(query_prompt, tokenizer, model, num_choices) == full_sample["correct_index"])
        question_correct += int(generate_letter(question_prompt, tokenizer, model, num_choices) == full_sample["correct_index"])

        if idx % 10 == 0:
            print(f"[long-context generation] processed {idx}/{total}")

    return {
        "num_examples": total,
        "full_context_accuracy": full_correct / total,
        "prefix_context_accuracy": prefix_correct / total,
        "query_select_accuracy": query_correct / total,
        "question_only_accuracy": question_correct / total,
    }


def write_summary(
    output_dir: Path,
    cfg: LongContextConfig,
    generation: dict[str, float],
    base_full_stats: dict[str, float],
    long_full_stats: dict[str, float],
    prefix_stats: dict[str, float],
    query_stats: dict[str, float],
) -> None:
    full_bytes = long_full_stats["avg_sender_bytes"]
    lines = ["# QASPER Long-Context Eval", ""]
    lines.append("## Setup")
    lines.append(f"- Base split sizes: {cfg.train_size}/{cfg.val_size}/{cfg.test_size}")
    lines.append(f"- Test examples evaluated: {generation['num_examples']}")
    lines.append(f"- Added distractor documents per example: {cfg.distractor_docs}")
    lines.append(f"- Distractor chars per document: {cfg.distractor_chars}")
    lines.append(f"- Prefix sender budget: {cfg.prefix_tokens} tokens")
    lines.append(f"- Query-select sender: chunk_words={cfg.chunk_words}, chunk_stride={cfg.chunk_stride}, top_k={cfg.selector_top_k}")
    lines.append("")
    lines.append("## Message Stats")
    lines.append(f"- Base full-doc size: {base_full_stats['avg_sender_tokens']:.1f} tok / {base_full_stats['avg_sender_bytes']:.1f} B")
    lines.append(f"- Long full-doc size: {long_full_stats['avg_sender_tokens']:.1f} tok / {long_full_stats['avg_sender_bytes']:.1f} B")
    lines.append(f"- Long prefix size: {prefix_stats['avg_sender_tokens']:.1f} tok / {prefix_stats['avg_sender_bytes']:.1f} B")
    lines.append(f"- Long query-select size: {query_stats['avg_sender_tokens']:.1f} tok / {query_stats['avg_sender_bytes']:.1f} B")
    lines.append("")
    lines.append("## Accuracy")
    lines.append(f"- Full text: {generation['full_context_accuracy']:.4f}")
    lines.append(f"- Prefix-{cfg.prefix_tokens}: {generation['prefix_context_accuracy']:.4f}")
    lines.append(f"- Query-select: {generation['query_select_accuracy']:.4f}")
    lines.append(f"- Question only: {generation['question_only_accuracy']:.4f}")
    lines.append("")
    lines.append("## Main Table")
    lines.append("| Method | Test Acc. | Message | Relative Bytes vs Long Full Text |")
    lines.append("| --- | ---: | --- | ---: |")
    lines.append(f"| Qwen Question Only | {generation['question_only_accuracy']:.3f} | 0 B | 0.0% |")
    lines.append(
        f"| Qwen Long Full-Text Handoff | {generation['full_context_accuracy']:.3f} | "
        f"{long_full_stats['avg_sender_tokens']:.1f} tok / {long_full_stats['avg_sender_bytes']:.1f} B | 100.0% |"
    )
    lines.append(
        f"| Qwen Long Prefix-{cfg.prefix_tokens} Handoff | {generation['prefix_context_accuracy']:.3f} | "
        f"{prefix_stats['avg_sender_tokens']:.1f} tok / {prefix_stats['avg_sender_bytes']:.1f} B | "
        f"{100.0 * prefix_stats['avg_sender_bytes'] / full_bytes:.1f}% |"
    )
    lines.append(
        f"| Qwen Long Query-Select Handoff | {generation['query_select_accuracy']:.3f} | "
        f"{query_stats['avg_sender_tokens']:.1f} tok / {query_stats['avg_sender_bytes']:.1f} B | "
        f"{100.0 * query_stats['avg_sender_bytes'] / full_bytes:.1f}% |"
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
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--distractor-docs", type=int, default=2)
    parser.add_argument("--distractor-chars", type=int, default=8000)
    parser.add_argument("--max-doc-length", type=int, default=256)
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--chunk-words", type=int, default=64)
    parser.add_argument("--chunk-stride", type=int, default=48)
    parser.add_argument("--selector-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    cfg = LongContextConfig(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        model_path=args.model_path,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        eval_examples=args.eval_examples,
        distractor_docs=args.distractor_docs,
        distractor_chars=args.distractor_chars,
        max_doc_length=args.max_doc_length,
        prefix_tokens=args.prefix_tokens,
        chunk_words=args.chunk_words,
        chunk_stride=args.chunk_stride,
        selector_top_k=args.selector_top_k,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = build_base_dataset(cfg)
    paper_pool = build_paper_pool(base_dataset)
    base_test = base_dataset["test"][: cfg.eval_examples]
    long_test = build_long_context_samples(base_test, paper_pool, cfg)

    started = time.time()
    tokenizer, model = qasper.runtime.load_qwen_model(cfg.model_path)

    query_cfg = qasper.QasperConfig(
        sender_mode="query_select",
        chunk_words=cfg.chunk_words,
        chunk_stride=cfg.chunk_stride,
        selector_top_k=cfg.selector_top_k,
    )
    query_dataset, query_sender_stats = qasper.attach_sender_views({"test": long_test}, query_cfg)
    query_test = query_dataset["test"]
    prefix_test = build_prefix_sender_samples(long_test, tokenizer, cfg.prefix_tokens)

    generation = run_long_context_generation(
        long_test,
        prefix_test,
        query_test,
        tokenizer,
        model,
        num_choices=4,
    )
    generation["elapsed_seconds"] = time.time() - started

    base_full_samples = [{**sample, "sender_text": sample["doc_text"]} for sample in base_test]
    base_full_stats = qasper.measure_sender_messages(base_full_samples, tokenizer)
    long_full_samples = [{**sample, "sender_text": sample["doc_text"]} for sample in long_test]
    long_full_stats = qasper.measure_sender_messages(long_full_samples, tokenizer)
    prefix_stats = qasper.measure_sender_messages(prefix_test, tokenizer)
    query_stats = qasper.measure_sender_messages(query_test, tokenizer)

    (cfg.output_dir / "base_test.json").write_text(json.dumps(base_test, indent=2))
    (cfg.output_dir / "long_test.json").write_text(json.dumps(long_test, indent=2))
    (cfg.output_dir / "query_select_test.json").write_text(json.dumps(query_test, indent=2))
    (cfg.output_dir / "prefix_test.json").write_text(json.dumps(prefix_test, indent=2))
    (cfg.output_dir / "generation_metrics.json").write_text(json.dumps(generation, indent=2))
    (cfg.output_dir / "message_stats.json").write_text(
        json.dumps(
            {
                "base_full": base_full_stats,
                "long_full": long_full_stats,
                "prefix": prefix_stats,
                "query_select": query_stats,
                "query_select_sender_stats": query_sender_stats["test"],
            },
            indent=2,
        )
    )
    (cfg.output_dir / "run_metadata.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    write_summary(cfg.output_dir, cfg, generation, base_full_stats, long_full_stats, prefix_stats, query_stats)
    print(f"Results written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
