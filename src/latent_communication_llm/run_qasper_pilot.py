from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def load_runtime_module():
    path = Path(__file__).parent / "run_qwen_handoff.py"
    spec = importlib.util.spec_from_file_location("qwen_handoff_qasper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime = load_runtime_module()


TOKEN_RE = re.compile(r"[a-z0-9]+")
SELECTOR_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass
class QasperConfig:
    train_size: int = 600
    val_size: int = 150
    test_size: int = 150
    sender_mode: str = "prefix"
    max_doc_length: int = 384
    max_question_length: int = 96
    max_choice_length: int = 24
    extractor_batch_size: int = 2
    train_batch_size: int = 32
    num_choices: int = 4
    max_answer_words: int = 8
    max_doc_chars: int = 14000
    chunk_words: int = 64
    chunk_stride: int = 48
    selector_top_k: int = 3
    generation_eval_examples: int = 40
    seed: int = 13


@dataclass
class StageConfig:
    name: str
    num_slots: int
    slot_dim: int
    num_heads: int
    epochs: int
    lr: float
    weight_decay: float
    variational: bool = False
    rate_weight: float = 0.0
    slot_dropout: float = 0.0
    noise_std: float = 0.0
    orth_weight: float = 0.0


def build_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="qasper_high_band",
            num_slots=8,
            slot_dim=128,
            num_heads=4,
            epochs=12,
            lr=3e-4,
            weight_decay=1e-4,
        ),
        StageConfig(
            name="qasper_purified",
            num_slots=8,
            slot_dim=96,
            num_heads=4,
            epochs=14,
            lr=3e-4,
            weight_decay=1e-4,
            variational=True,
            rate_weight=2e-3,
            slot_dropout=0.08,
            noise_std=0.03,
            orth_weight=1e-2,
        ),
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_qasper_archives(data_dir: Path) -> dict[str, Path]:
    urls = {
        "train_dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
        "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "train_dev": data_dir / "qasper-train-dev-v0.3.tgz",
        "test": data_dir / "qasper-test-and-evaluator-v0.3.tgz",
    }
    for key, url in urls.items():
        path = out_paths[key]
        if path.exists() and path.stat().st_size > 0:
            continue
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
    return out_paths


def ensure_qasper_json(data_dir: Path) -> dict[str, Path]:
    archives = ensure_qasper_archives(data_dir)
    paths = {
        "train": data_dir / "qasper-train-v0.3.json",
        "validation": data_dir / "qasper-dev-v0.3.json",
        "test": data_dir / "qasper-test-v0.3.json",
    }
    if all(path.exists() for path in paths.values()):
        return paths

    archive_members = {
        archives["train_dev"]: {"qasper-train-v0.3.json", "qasper-dev-v0.3.json"},
        archives["test"]: {"qasper-test-v0.3.json"},
    }
    for archive_path, members in archive_members.items():
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if name not in members:
                    continue
                out_path = data_dir / name
                if out_path.exists():
                    continue
                extracted = tar.extractfile(member)
                assert extracted is not None
                out_path.write_bytes(extracted.read())
    return paths


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip().replace("\n", " ")
    return " ".join(text.split())


def normalize_selector_text(text: str | None) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower()))


def selector_terms(text: str | None) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall((text or "").lower())
        if len(token) > 1 and token not in SELECTOR_STOPWORDS
    ]


def canonical_answer_text(answer_payload: dict) -> tuple[str | None, str | None]:
    answer = answer_payload["answer"]
    if answer.get("unanswerable"):
        return None, None
    spans = [normalize_answer(span) for span in answer.get("extractive_spans", []) if normalize_answer(span)]
    if spans:
        return spans[0], "extractive"
    if answer.get("yes_no") is True:
        return "yes", "yes_no"
    if answer.get("yes_no") is False:
        return "no", "yes_no"
    free_form = normalize_answer(answer.get("free_form_answer", ""))
    if free_form:
        return free_form, "free_form"
    return None, None


def extract_text_evidence(answer_payload: dict) -> list[str]:
    evidence = []
    for text in answer_payload["answer"].get("evidence", []):
        normalized = normalize_answer(text)
        if not normalized or normalized.startswith("FLOAT SELECTED"):
            continue
        evidence.append(normalized)
    return evidence


def render_doc_text(paper: dict, *, max_chars: int) -> str:
    sections = [
        normalize_answer(paper.get("title")),
        normalize_answer(paper.get("abstract")),
    ]
    for section in paper.get("full_text", []):
        section_name = normalize_answer(section.get("section_name"))
        paragraphs = [normalize_answer(p) for p in section.get("paragraphs", []) if normalize_answer(p)]
        if not paragraphs:
            continue
        if section_name:
            sections.append(f"{section_name}:")
        sections.extend(paragraphs)
    text = "\n".join(part for part in sections if part)
    return text[:max_chars]


def build_doc_segments(doc_text: str, cfg: QasperConfig) -> list[dict]:
    segments = []
    lines = [line.strip() for line in doc_text.split("\n") if line.strip()]
    current_section = ""

    for line_idx, line in enumerate(lines):
        if line.endswith(":") and len(line.split()) <= 12:
            current_section = line[:-1]
            continue

        if line_idx == 0:
            line_prefix = "Title: "
        elif line_idx == 1:
            line_prefix = "Abstract: "
        elif current_section:
            line_prefix = f"{current_section}: "
        else:
            line_prefix = ""

        words = line.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            window = words[start : start + cfg.chunk_words]
            if not window:
                break
            segment_text = f"{line_prefix}{' '.join(window)}"
            segment_terms = selector_terms(segment_text)
            segments.append(
                {
                    "text": segment_text,
                    "source_text": line,
                    "source_norm": normalize_selector_text(line),
                    "term_set": set(segment_terms),
                    "term_freq": {term: segment_terms.count(term) for term in set(segment_terms)},
                }
            )
            if start + cfg.chunk_words >= len(words):
                break
            start += max(1, cfg.chunk_stride)
    return segments


def find_oracle_segment_indices(segments: list[dict], evidence_texts: list[str]) -> list[int]:
    evidence_norms = [normalize_selector_text(text) for text in evidence_texts if normalize_selector_text(text)]
    indices = []
    for idx, segment in enumerate(segments):
        source_norm = segment["source_norm"]
        if any(evidence_norm in source_norm or source_norm in evidence_norm for evidence_norm in evidence_norms):
            indices.append(idx)
    return indices


def score_segment(segment: dict, question_terms: set[str], option_term_sets: list[set[str]], option_phrases: list[str]) -> float:
    score = 0.0
    segment_freq = segment["term_freq"]
    segment_term_set = segment["term_set"]
    segment_norm = normalize_selector_text(segment["text"])

    score += sum(segment_freq.get(term, 0) for term in question_terms)
    for option_terms in option_term_sets:
        if option_terms:
            score += 0.35 * len(segment_term_set & option_terms)
    for option_phrase in option_phrases:
        if option_phrase and option_phrase in segment_norm:
            score += 3.0
    return score


def choose_sender_text(sample: dict, cfg: QasperConfig) -> tuple[str, dict[str, float | int | list[int] | str | None]]:
    if cfg.sender_mode == "prefix":
        sender_text = sample["doc_text"]
        return sender_text, {
            "mode": cfg.sender_mode,
            "selected_chunk_ids": [],
            "num_segments": 0,
            "num_selected_chunks": 0,
            "oracle_num_chunks": 0,
            "oracle_chunk_recall": None,
            "bytes": len(sender_text.encode("utf-8")),
        }

    segments = build_doc_segments(sample["doc_text"], cfg)
    if not segments:
        sender_text = sample["doc_text"]
        return sender_text, {
            "mode": cfg.sender_mode,
            "selected_chunk_ids": [],
            "num_segments": 0,
            "num_selected_chunks": 0,
            "oracle_num_chunks": 0,
            "oracle_chunk_recall": None,
            "bytes": len(sender_text.encode("utf-8")),
        }

    question_terms = set(selector_terms(sample["question_text"]))
    option_term_sets = [set(selector_terms(option)) for option in sample["options"]]
    option_phrases = [normalize_selector_text(option) for option in sample["options"]]
    ranked_indices = sorted(
        range(len(segments)),
        key=lambda idx: (score_segment(segments[idx], question_terms, option_term_sets, option_phrases), -idx),
        reverse=True,
    )

    oracle_indices = find_oracle_segment_indices(segments, sample.get("evidence_texts", []))
    oracle_index_set = set(oracle_indices)
    if cfg.sender_mode == "oracle_evidence" and oracle_indices:
        candidate_indices = [idx for idx in ranked_indices if idx in oracle_index_set]
    else:
        candidate_indices = ranked_indices

    selected = []
    seen_sources = set()
    for idx in candidate_indices:
        if len(selected) >= cfg.selector_top_k:
            break
        source_norm = segments[idx]["source_norm"]
        if source_norm in seen_sources:
            continue
        selected.append(idx)
        seen_sources.add(source_norm)

    if len(selected) < cfg.selector_top_k:
        for idx in candidate_indices:
            if idx in selected:
                continue
            selected.append(idx)
            if len(selected) >= cfg.selector_top_k:
                break

    if len(selected) < cfg.selector_top_k:
        for idx in range(len(segments)):
            if idx in selected:
                continue
            selected.append(idx)
            if len(selected) >= cfg.selector_top_k:
                break

    selected = sorted(selected)
    sender_text = "\n".join(segments[idx]["text"] for idx in selected)
    oracle_recall = None
    if oracle_indices:
        overlap = len(set(selected) & oracle_index_set)
        oracle_recall = overlap / len(oracle_index_set)
    return sender_text, {
        "mode": cfg.sender_mode,
        "selected_chunk_ids": selected,
        "num_segments": len(segments),
        "num_selected_chunks": len(selected),
        "oracle_num_chunks": len(oracle_indices),
        "oracle_chunk_recall": oracle_recall,
        "bytes": len(sender_text.encode("utf-8")),
    }


def attach_sender_views(dataset: dict[str, list[dict]], cfg: QasperConfig) -> tuple[dict[str, list[dict]], dict[str, dict[str, float | int | None]]]:
    out = {}
    stats = {}
    for split_name, samples in dataset.items():
        split_bytes = []
        split_recalls = []
        split_selected = []
        split_total_segments = []
        augmented = []
        for sample in samples:
            sender_text, sender_stats = choose_sender_text(sample, cfg)
            augmented.append({**sample, "sender_text": sender_text, "sender_stats": sender_stats})
            split_bytes.append(float(sender_stats["bytes"]))
            split_selected.append(int(sender_stats["num_selected_chunks"]))
            split_total_segments.append(int(sender_stats["num_segments"]))
            if sender_stats["oracle_chunk_recall"] is not None:
                split_recalls.append(float(sender_stats["oracle_chunk_recall"]))
        out[split_name] = augmented
        stats[split_name] = {
            "avg_sender_bytes": sum(split_bytes) / len(split_bytes) if split_bytes else 0.0,
            "avg_selected_chunks": sum(split_selected) / len(split_selected) if split_selected else 0.0,
            "avg_total_segments": sum(split_total_segments) / len(split_total_segments) if split_total_segments else 0.0,
            "avg_oracle_chunk_recall": sum(split_recalls) / len(split_recalls) if split_recalls else None,
        }
    return out, stats


def build_flat_samples(json_path: Path, split_name: str, cfg: QasperConfig) -> list[dict]:
    papers = json.loads(json_path.read_text())
    samples = []
    for paper_id, paper in papers.items():
        doc_text = render_doc_text(paper, max_chars=cfg.max_doc_chars)
        if not doc_text:
            continue
        for qa in paper.get("qas", []):
            question = normalize_answer(qa.get("question", ""))
            if not question:
                continue
            answer_text = None
            answer_type = None
            evidence_texts = []
            for answer_payload in qa.get("answers", []):
                candidate, candidate_type = canonical_answer_text(answer_payload)
                if candidate is None:
                    continue
                answer_text = candidate
                answer_type = candidate_type
                evidence_texts = extract_text_evidence(answer_payload)
                break
            if answer_text is None:
                continue
            if len(answer_text.split()) > cfg.max_answer_words:
                continue
            samples.append(
                {
                    "split": split_name,
                    "paper_id": paper_id,
                    "question_id": qa.get("question_id", ""),
                    "question_text": question,
                    "answer_text": answer_text,
                    "answer_type": answer_type,
                    "doc_text": doc_text,
                    "evidence_texts": evidence_texts,
                }
            )
    return samples


def pick_distractors(sample: dict, pool: list[dict], rng: random.Random, num_distractors: int) -> list[str]:
    answer = sample["answer_text"]
    answer_type = sample["answer_type"]
    word_count = len(answer.split())
    unique_candidates = []
    seen = {answer}

    for item in pool:
        candidate = item["answer_text"]
        if candidate in seen:
            continue
        if item["answer_type"] != answer_type:
            continue
        if abs(len(candidate.split()) - word_count) > 2:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    if len(unique_candidates) < num_distractors:
        for item in pool:
            candidate = item["answer_text"]
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)
            if len(unique_candidates) >= num_distractors:
                break

    if len(unique_candidates) < num_distractors:
        raise RuntimeError(f"Not enough distractors for sample {sample['question_id']}")
    return rng.sample(unique_candidates, num_distractors)


def build_mc_split(samples: list[dict], pool: list[dict], cfg: QasperConfig, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for sample in samples:
        distractors = pick_distractors(sample, pool, rng, cfg.num_choices - 1)
        options = [sample["answer_text"], *distractors]
        rng.shuffle(options)
        correct_index = options.index(sample["answer_text"])
        option_lines = [f"{chr(65 + idx)}. {text}" for idx, text in enumerate(options)]
        out.append(
            {
                **sample,
                "options": options,
                "correct_index": correct_index,
                "options_text": "\n".join(option_lines),
            }
        )
    return out


def build_dataset(cfg: QasperConfig, data_dir: Path) -> tuple[dict[str, list[dict]], dict[str, dict[str, float | int | None]]]:
    json_paths = ensure_qasper_json(data_dir)
    raw = {
        "train": build_flat_samples(json_paths["train"], "train", cfg),
        "validation": build_flat_samples(json_paths["validation"], "validation", cfg),
        "test": build_flat_samples(json_paths["test"], "test", cfg),
    }

    rng = random.Random(cfg.seed)
    for split in raw.values():
        rng.shuffle(split)

    selected = {
        "train": raw["train"][: cfg.train_size],
        "validation": raw["validation"][: cfg.val_size],
        "test": raw["test"][: cfg.test_size],
    }
    pool = raw["train"] + raw["validation"] + raw["test"]
    dataset = {
        split_name: build_mc_split(split_samples, pool, cfg, cfg.seed + idx)
        for idx, (split_name, split_samples) in enumerate(selected.items())
    }
    return attach_sender_views(dataset, cfg)


def format_question_prompt(sample: dict) -> str:
    return (
        f"Question:\n{sample['question_text']}\n"
        f"Options:\n{sample['options_text']}\n"
        "Choose the correct option."
    )


def format_doc_prompt(sample: dict) -> str:
    return f"Read the paper carefully.\nPaper:\n{sample['sender_text']}"


@torch.no_grad()
def extract_features(samples: list[dict], cfg: QasperConfig, tokenizer, model, split_name: str) -> dict[str, torch.Tensor]:
    doc_hidden_chunks = []
    doc_mask_chunks = []
    question_hidden_chunks = []
    choice_hidden_chunks = []
    labels = []

    total = len(samples)
    for start in range(0, total, cfg.extractor_batch_size):
        batch = samples[start : start + cfg.extractor_batch_size]
        doc_inputs = tokenizer(
            [format_doc_prompt(item) for item in batch],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.max_doc_length,
        ).to(model.device)
        doc_outputs = model(**doc_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        doc_hidden_chunks.append(doc_outputs.hidden_states[-1].to(dtype=torch.float16).cpu())
        doc_mask_chunks.append(doc_inputs["attention_mask"].cpu())

        question_inputs = tokenizer(
            [format_question_prompt(item) for item in batch],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.max_question_length,
        ).to(model.device)
        question_outputs = model(
            **question_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        question_hidden = runtime.last_token_pool(question_outputs.hidden_states[-1], question_inputs["attention_mask"])
        question_hidden_chunks.append(question_hidden.to(dtype=torch.float16).cpu())

        flat_choices = []
        for item in batch:
            for choice in item["options"]:
                flat_choices.append(f"Candidate answer:\n{choice}")
        choice_inputs = tokenizer(
            flat_choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.max_choice_length,
        ).to(model.device)
        choice_outputs = model(
            **choice_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        choice_hidden = runtime.last_token_pool(choice_outputs.hidden_states[-1], choice_inputs["attention_mask"])
        choice_hidden = choice_hidden.view(len(batch), cfg.num_choices, -1)
        choice_hidden_chunks.append(choice_hidden.to(dtype=torch.float16).cpu())

        labels.extend(item["correct_index"] for item in batch)
        if (start // cfg.extractor_batch_size) % 25 == 0:
            print(f"[{split_name}] extracted {min(start + len(batch), total)}/{total}")

    return {
        "doc_hidden": torch.cat(doc_hidden_chunks, dim=0),
        "doc_mask": torch.cat(doc_mask_chunks, dim=0),
        "question_hidden": torch.cat(question_hidden_chunks, dim=0),
        "choice_hidden": torch.cat(choice_hidden_chunks, dim=0),
        "label": torch.tensor(labels, dtype=torch.long),
    }


def build_tensor_dataset(features: dict[str, torch.Tensor]) -> TensorDataset:
    return TensorDataset(
        features["doc_hidden"],
        features["doc_mask"],
        features["question_hidden"],
        features["choice_hidden"],
        features["label"],
    )


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> list[torch.Tensor]:
    out = []
    for tensor in batch:
        if tensor.is_floating_point():
            out.append(tensor.to(device=device, dtype=torch.float32, non_blocking=True))
        else:
            out.append(tensor.to(device=device, non_blocking=True))
    return out


class MCQuestionOnlyModel(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int = 768) -> None:
        super().__init__()
        self.question_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
        )
        self.choice_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
        )

    def forward(self, question_hidden: torch.Tensor, choice_hidden: torch.Tensor) -> torch.Tensor:
        q = self.question_proj(question_hidden)
        c = self.choice_proj(choice_hidden)
        return torch.einsum("bd,bcd->bc", q, c)


class MCLatentHandoffModel(nn.Module):
    def __init__(self, hidden_dim: int, stage_cfg: StageConfig) -> None:
        super().__init__()
        self.stage_cfg = stage_cfg
        self.bottleneck = runtime.DocSlotBottleneck(
            hidden_dim=hidden_dim,
            num_slots=stage_cfg.num_slots,
            slot_dim=stage_cfg.slot_dim,
            num_heads=stage_cfg.num_heads,
            variational=stage_cfg.variational,
        )
        self.query_proj = nn.Linear(hidden_dim, stage_cfg.slot_dim)
        self.cross_attn = nn.MultiheadAttention(stage_cfg.slot_dim, stage_cfg.num_heads, batch_first=True)
        self.joint_proj = nn.Sequential(
            nn.LayerNorm(2 * stage_cfg.slot_dim),
            nn.Linear(2 * stage_cfg.slot_dim, stage_cfg.slot_dim),
            nn.GELU(),
        )
        self.choice_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, stage_cfg.slot_dim),
            nn.GELU(),
        )

    def encode(
        self,
        doc_hidden: torch.Tensor,
        doc_mask: torch.Tensor,
        *,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.bottleneck(
            doc_hidden,
            doc_mask,
            apply_channel=training,
            slot_dropout=self.stage_cfg.slot_dropout,
            noise_std=self.stage_cfg.noise_std,
        )

    def forward(self, doc_hidden: torch.Tensor, doc_mask: torch.Tensor, question_hidden: torch.Tensor, choice_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        slots, stats = self.encode(doc_hidden, doc_mask, training=self.training)
        q = self.query_proj(question_hidden).unsqueeze(1)
        ctx, _ = self.cross_attn(q, slots, slots, need_weights=False)
        joint = self.joint_proj(torch.cat([q.squeeze(1), ctx.squeeze(1)], dim=-1))
        choice_repr = self.choice_proj(choice_hidden)
        logits = torch.einsum("bd,bcd->bc", joint, choice_repr)
        return {
            "slots": slots,
            "mu": stats["mu"],
            "logvar": stats["logvar"],
            "logits": logits,
        }


def evaluate_question_only(model: MCQuestionOnlyModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            _, _, question_hidden, choice_hidden, label = move_batch(batch, device)
            logits = model(question_hidden, choice_hidden)
            pred = logits.argmax(dim=-1)
            correct += int((pred == label).sum().item())
            total += int(label.numel())
    return correct / total


def train_question_only(train_features: dict[str, torch.Tensor], val_features: dict[str, torch.Tensor], test_features: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    model = MCQuestionOnlyModel(int(train_features["question_hidden"].size(-1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_loader = DataLoader(build_tensor_dataset(train_features), batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(build_tensor_dataset(val_features), batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(build_tensor_dataset(test_features), batch_size=64, shuffle=False, num_workers=0)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = 0.0
    for _ in range(12):
        model.train()
        for batch in train_loader:
            _, _, question_hidden, choice_hidden, label = move_batch(batch, device)
            logits = model(question_hidden, choice_hidden)
            loss = F.cross_entropy(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_acc = evaluate_question_only(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {
        "val_accuracy": best_val,
        "test_accuracy": evaluate_question_only(model, test_loader, device),
    }


def evaluate_latent(model: MCLatentHandoffModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_kl = 0.0
    with torch.no_grad():
        for batch in loader:
            doc_hidden, doc_mask, question_hidden, choice_hidden, label = move_batch(batch, device)
            outputs = model(doc_hidden, doc_mask, question_hidden, choice_hidden)
            pred = outputs["logits"].argmax(dim=-1)
            correct += int((pred == label).sum().item())
            total += int(label.numel())
            total_kl += float(runtime.kl_rate_loss(outputs["mu"], outputs["logvar"], device).detach()) * label.size(0)
    return {"accuracy": correct / total, "avg_kl_rate": total_kl / total}


def train_latent_stage(stage_cfg: StageConfig, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, hidden_dim: int, teacher: MCLatentHandoffModel | None = None) -> tuple[MCLatentHandoffModel, dict[str, float]]:
    model = MCLatentHandoffModel(hidden_dim, stage_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage_cfg.lr, weight_decay=stage_cfg.weight_decay)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = 0.0
    best_snapshot = {}

    for epoch in range(1, stage_cfg.epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            doc_hidden, doc_mask, question_hidden, choice_hidden, label = move_batch(batch, device)
            outputs = model(doc_hidden, doc_mask, question_hidden, choice_hidden)
            loss = F.cross_entropy(outputs["logits"], label)
            if stage_cfg.rate_weight > 0.0:
                loss = loss + stage_cfg.rate_weight * runtime.kl_rate_loss(outputs["mu"], outputs["logvar"], device)
            if stage_cfg.orth_weight > 0.0:
                loss = loss + stage_cfg.orth_weight * runtime.orthogonality_loss(outputs["slots"])
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(doc_hidden, doc_mask, question_hidden, choice_hidden)["logits"]
                loss = loss + runtime.distill_loss(outputs["logits"], teacher_logits, 2.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_correct += int((outputs["logits"].argmax(dim=-1) == label).sum().item())
            train_total += int(label.numel())
        val_metrics = evaluate_latent(model, val_loader, device)
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_snapshot = {
                "epoch": epoch,
                "train_accuracy": train_correct / train_total,
                "val_accuracy": val_metrics["accuracy"],
                "val_avg_kl_rate": val_metrics["avg_kl_rate"],
            }
        print(f"[{stage_cfg.name}] epoch {epoch:02d} val_acc={val_metrics['accuracy']:.4f}")

    model.load_state_dict(best_state)
    return model, best_snapshot


@torch.no_grad()
def run_generation_baselines(samples: list[dict], cfg: QasperConfig, tokenizer, model) -> dict[str, float]:
    chosen = samples[: cfg.generation_eval_examples]
    full_correct = 0
    question_correct = 0
    sender_correct = 0
    for idx, sample in enumerate(chosen):
        option_lines = [f"{chr(65 + j)}. {choice}" for j, choice in enumerate(sample["options"])]
        option_block = "\n".join(option_lines)
        full_prompt = (
            "Read the paper and answer the multiple-choice question.\n"
            "Output only the option letter.\n"
            f"Paper:\n{sample['doc_text']}\n"
            f"Question: {sample['question_text']}\n"
            f"Options:\n{option_block}\nAnswer:"
        )
        question_prompt = (
            "Answer the multiple-choice question.\n"
            "Output only the option letter.\n"
            f"Question: {sample['question_text']}\n"
            f"Options:\n{option_block}\nAnswer:"
        )
        sender_prompt = (
            "Read the paper excerpt and answer the multiple-choice question.\n"
            "Output only the option letter.\n"
            f"Paper excerpt:\n{sample['sender_text']}\n"
            f"Question: {sample['question_text']}\n"
            f"Options:\n{option_block}\nAnswer:"
        )
        for prompt_name, prompt in [("full", full_prompt), ("question_only", question_prompt), ("sender", sender_prompt)]:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
            )
            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip().upper()
            pred = None
            for letter_idx in range(cfg.num_choices):
                if chr(65 + letter_idx) in text:
                    pred = letter_idx
                    break
            if prompt_name == "full":
                full_correct += int(pred == sample["correct_index"])
            elif prompt_name == "sender":
                sender_correct += int(pred == sample["correct_index"])
            else:
                question_correct += int(pred == sample["correct_index"])
        if idx % 10 == 0:
            print(f"[generation] processed {idx + 1}/{len(chosen)}")
    total = len(chosen)
    return {
        "num_examples": total,
        "sender_mode": cfg.sender_mode,
        "full_context_accuracy": full_correct / total,
        "sender_context_accuracy": sender_correct / total,
        "question_only_accuracy": question_correct / total,
    }


def measure_sender_messages(samples: list[dict], tokenizer) -> dict[str, float]:
    sender_bytes = []
    sender_tokens = []
    full_doc_bytes = []
    full_doc_tokens = []
    for sample in samples:
        sender_text = sample["sender_text"]
        doc_text = sample["doc_text"]
        sender_bytes.append(len(sender_text.encode("utf-8")))
        full_doc_bytes.append(len(doc_text.encode("utf-8")))
        sender_tokens.append(len(tokenizer.encode(sender_text, add_special_tokens=False)))
        full_doc_tokens.append(len(tokenizer.encode(doc_text, add_special_tokens=False)))
    return {
        "avg_sender_bytes": sum(sender_bytes) / len(sender_bytes),
        "avg_sender_tokens": sum(sender_tokens) / len(sender_tokens),
        "avg_full_doc_bytes": sum(full_doc_bytes) / len(full_doc_bytes),
        "avg_full_doc_tokens": sum(full_doc_tokens) / len(full_doc_tokens),
    }


def write_summary(
    output_dir: Path,
    cfg: QasperConfig,
    stage_cfgs: list[StageConfig],
    stage_summaries: list[dict[str, float]],
    generation: dict[str, float],
    question_only: dict[str, float],
    sender_stats: dict[str, dict[str, float | int | None]],
    sender_message_stats: dict[str, float],
) -> None:
    lines = ["# QASPER Pilot", ""]
    lines.append("## Task")
    lines.append(f"- Train/val/test sizes: {cfg.train_size}/{cfg.val_size}/{cfg.test_size}")
    lines.append(f"- Multiple-choice options: {cfg.num_choices}")
    lines.append(f"- Max answer words: {cfg.max_answer_words}")
    lines.append(f"- Sender mode: `{cfg.sender_mode}`")
    lines.append("- Receiver reads only question plus options.")
    lines.append("")
    lines.append("## Baselines")
    lines.append(f"- Question-only scorer accuracy: {question_only['test_accuracy']:.4f}")
    lines.append(f"- Qwen full-context generation accuracy on {generation['num_examples']} examples: {generation['full_context_accuracy']:.4f}")
    lines.append(
        f"- Qwen sender-context generation accuracy on {generation['num_examples']} examples: {generation['sender_context_accuracy']:.4f}"
    )
    lines.append(f"- Qwen question-only generation accuracy on {generation['num_examples']} examples: {generation['question_only_accuracy']:.4f}")
    lines.append("")
    lines.append("## Sender Stats")
    lines.append(f"- Avg sender bytes on test: {sender_message_stats['avg_sender_bytes']:.1f}")
    lines.append(f"- Avg sender tokens on test: {sender_message_stats['avg_sender_tokens']:.1f}")
    lines.append(f"- Avg full paper bytes on test: {sender_message_stats['avg_full_doc_bytes']:.1f}")
    lines.append(f"- Avg full paper tokens on test: {sender_message_stats['avg_full_doc_tokens']:.1f}")
    lines.append(f"- Avg selected chunks on test: {sender_stats['test']['avg_selected_chunks']:.2f}")
    if sender_stats["test"]["avg_oracle_chunk_recall"] is not None:
        lines.append(f"- Avg oracle chunk recall on test: {sender_stats['test']['avg_oracle_chunk_recall']:.4f}")
    lines.append("")
    lines.append("## Latent Handoff")
    for stage_cfg, summary in zip(stage_cfgs, stage_summaries, strict=True):
        lines.append(f"### {stage_cfg.name}")
        lines.append(f"- Message size: {summary['message_floats']} fp16 / {summary['message_bytes']} B")
        lines.append(f"- Val accuracy: {summary['best_val_accuracy']:.4f}")
        lines.append(f"- Test accuracy: {summary['test_accuracy']:.4f}")
        lines.append(f"- Avg KL rate: {summary['test_avg_kl_rate']:.4f}")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/qasper"))
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--train-size", type=int, default=600)
    parser.add_argument("--val-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--sender-mode", type=str, choices=["prefix", "query_select", "oracle_evidence"], default="prefix")
    parser.add_argument("--max-doc-length", type=int, default=384)
    parser.add_argument("--chunk-words", type=int, default=64)
    parser.add_argument("--chunk-stride", type=int, default=48)
    parser.add_argument("--selector-top-k", type=int, default=3)
    parser.add_argument("--extractor-batch-size", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--generation-eval-examples", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = QasperConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        sender_mode=args.sender_mode,
        max_doc_length=args.max_doc_length,
        chunk_words=args.chunk_words,
        chunk_stride=args.chunk_stride,
        selector_top_k=args.selector_top_k,
        extractor_batch_size=args.extractor_batch_size,
        train_batch_size=args.train_batch_size,
        generation_eval_examples=args.generation_eval_examples,
    )
    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, sender_stats = build_dataset(cfg, args.data_dir)
    for split_name, samples in dataset.items():
        (output_dir / f"{split_name}.json").write_text(json.dumps(samples, indent=2))
    (output_dir / "sender_stats.json").write_text(json.dumps(sender_stats, indent=2))

    started = time.time()
    tokenizer, qwen_model = runtime.load_qwen_model(args.model_path)
    generation = run_generation_baselines(dataset["test"], cfg, tokenizer, qwen_model)
    (output_dir / "generation_baseline.json").write_text(json.dumps(generation, indent=2))
    sender_message_stats = measure_sender_messages(dataset["test"], tokenizer)
    (output_dir / "sender_message_stats.json").write_text(json.dumps(sender_message_stats, indent=2))

    feature_splits = {}
    for split_name, samples in dataset.items():
        feature_splits[split_name] = extract_features(samples, cfg, tokenizer, qwen_model, split_name)
        torch.save(feature_splits[split_name], output_dir / f"{split_name}_features.pt")

    hidden_dim = int(feature_splits["train"]["question_hidden"].size(-1))
    del qwen_model
    torch.cuda.empty_cache()

    device = torch.device(args.device)
    question_only = train_question_only(feature_splits["train"], feature_splits["validation"], feature_splits["test"], device)
    (output_dir / "question_only_metrics.json").write_text(json.dumps(question_only, indent=2))

    train_loader = DataLoader(build_tensor_dataset(feature_splits["train"]), batch_size=cfg.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(build_tensor_dataset(feature_splits["validation"]), batch_size=cfg.train_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(build_tensor_dataset(feature_splits["test"]), batch_size=cfg.train_batch_size, shuffle=False, num_workers=0)

    teacher = None
    stage_summaries = []
    for stage_cfg in build_stages():
        model, best_snapshot = train_latent_stage(stage_cfg, train_loader, val_loader, device, hidden_dim, teacher)
        test_metrics = evaluate_latent(model, test_loader, device)
        summary = {
            "message_floats": stage_cfg.num_slots * stage_cfg.slot_dim,
            "message_bytes": 2 * stage_cfg.num_slots * stage_cfg.slot_dim,
            "best_epoch": best_snapshot["epoch"],
            "best_val_accuracy": best_snapshot["val_accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_avg_kl_rate": test_metrics["avg_kl_rate"],
        }
        stage_summaries.append(summary)
        (output_dir / f"{stage_cfg.name}_metrics.json").write_text(json.dumps(summary, indent=2))
        torch.save(model.state_dict(), output_dir / f"{stage_cfg.name}.pt")
        teacher = model
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    write_summary(output_dir, cfg, build_stages(), stage_summaries, generation, question_only, sender_stats, sender_message_stats)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset": asdict(cfg),
                "stages": [asdict(stage) for stage in build_stages()],
                "sender_stats": sender_stats,
                "sender_message_stats": sender_message_stats,
                "generation_baseline": generation,
                "question_only_metrics": question_only,
                "runtime_seconds": time.time() - started,
            },
            indent=2,
        )
    )
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
