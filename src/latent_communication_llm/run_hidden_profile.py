"""
Hidden Profile experiment for latent slot communication.

Implements the Hidden Profile paradigm: information about a person is split
across two agents. Neither agent alone can answer all questions — they must
communicate through a latent bottleneck.

Setup:
  - 8 fields split into 2 groups of 4 (sender_fields / receiver_fields)
  - Sender sees only sender_fields in the profile text
  - Receiver sees only receiver_fields + the question
  - Question may ask about ANY of the 8 fields
  - If question targets a sender_field → receiver MUST use sender's slot
  - If question targets a receiver_field → receiver can answer locally

This creates genuine information asymmetry: the slot channel is the ONLY
path for sender-private information to reach the receiver.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── reuse core definitions from run_qwen_handoff ──
from run_qwen_handoff import (
    FIELD_SPECS, FIELD_SENTENCES, QUESTION_TEMPLATES, PROFILE_STYLES,
    FIRST_NAMES, LAST_NAMES, ANSWER_VOCAB, ANSWER_TO_ID,
    set_seed, normalize_answer, load_qwen_model, last_token_pool,
    DocSlotBottleneck, QueryReceiver, StateHead, StyleAdversary,
    GradientReversal, grad_reverse,
)


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HiddenProfileConfig:
    # data
    train_size: int = 8000
    val_size: int = 800
    test_size: int = 800
    num_fields: int = 8
    num_sender_fields: int = 4          # sender sees these fields
    num_styles: int = 6
    max_doc_length: int = 128           # shorter: each agent sees half
    max_question_length: int = 48
    extractor_batch_size: int = 3
    train_batch_size: int = 64
    seed: int = 42
    # split strategy: "random" per sample, or "fixed" (same split for all)
    split_strategy: str = "fixed"
    # model
    num_slots: int = 4                  # sender compresses 4 fields → 4 slots
    slot_dim: int = 64
    num_heads: int = 4
    variational: bool = True
    rate_weight: float = 5e-3
    slot_dropout: float = 0.20
    noise_std: float = 0.08
    style_adv_weight: float = 0.5
    style_adv_lambda: float = 1.0
    slot_factor_weight: float = 0.25
    state_loss_weight: float = 0.15
    orth_weight: float = 1e-2
    # training
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-3
    # paths
    model_path: str = "Qwen/Qwen3-14B"
    output_dir: str = "results/hidden_profile"


# ═══════════════════════════════════════════════════════════════════
# Data generation — Hidden Profile splits
# ═══════════════════════════════════════════════════════════════════

def pick_name(sample_idx: int) -> str:
    first = FIRST_NAMES[sample_idx % len(FIRST_NAMES)]
    last = LAST_NAMES[(sample_idx // len(FIRST_NAMES)) % len(LAST_NAMES)]
    return f"{first} {last}"


def render_partial_profile(name: str, value_ids: list[int],
                           visible_fields: list[int], style_id: int) -> str:
    """Render a profile containing only the visible_fields."""
    style = PROFILE_STYLES[style_id]
    # filter style order to only include visible fields
    ordered_fields = [f for f in style["order"] if f in visible_fields]
    if not ordered_fields:
        return f"No information available for {name}."
    sentences = []
    for field_idx in ordered_fields:
        field_name, field_values = FIELD_SPECS[field_idx]
        value = field_values[value_ids[field_idx]]
        sentences.append(FIELD_SENTENCES[field_name].format(name=name, value=value))
    parts = [style["lead"].format(name=name), sentences[0]]
    for s in sentences[1:]:
        parts.append(f"{style['connector']}, {s[0].lower()}{s[1:]}")
    parts.append(style["closing"])
    return " ".join(parts)


def render_question(name: str, field_idx: int, phrasing_id: int) -> str:
    field_name, _ = FIELD_SPECS[field_idx]
    templates = QUESTION_TEMPLATES[field_name]
    return templates[phrasing_id % len(templates)].format(name=name)


def split_fields(num_fields: int, num_sender: int,
                 rng: random.Random) -> tuple[list[int], list[int]]:
    """Split field indices into sender and receiver groups."""
    all_fields = list(range(num_fields))
    rng.shuffle(all_fields)
    sender_fields = sorted(all_fields[:num_sender])
    receiver_fields = sorted(all_fields[num_sender:])
    return sender_fields, receiver_fields


def build_hidden_profile_split(size: int, seed: int,
                                cfg: HiddenProfileConfig) -> list[dict]:
    rng = random.Random(seed)
    # fixed split: same field assignment for all samples
    if cfg.split_strategy == "fixed":
        fixed_sender, fixed_receiver = split_fields(
            cfg.num_fields, cfg.num_sender_fields, random.Random(cfg.seed))

    samples = []
    for idx in range(size):
        sample_idx = seed * 10000 + idx
        name = pick_name(sample_idx)
        value_ids = [rng.randrange(len(vals)) for _, vals in FIELD_SPECS]
        style_id = rng.randrange(cfg.num_styles)
        question_field = rng.randrange(cfg.num_fields)
        phrasing_id = rng.randrange(3)
        answer = FIELD_SPECS[question_field][1][value_ids[question_field]]

        if cfg.split_strategy == "fixed":
            sender_fields, receiver_fields = fixed_sender, fixed_receiver
        else:
            sender_fields, receiver_fields = split_fields(
                cfg.num_fields, cfg.num_sender_fields, rng)

        # key label: does the question target sender-private info?
        needs_comm = int(question_field in sender_fields)

        samples.append({
            "sample_id": sample_idx,
            "name": name,
            "state_ids": value_ids,
            "style_id": style_id,
            "question_field": question_field,
            "answer_text": answer,
            "answer_id": ANSWER_TO_ID[answer],
            "sender_fields": sender_fields,
            "receiver_fields": receiver_fields,
            "needs_comm": needs_comm,
            "sender_profile": render_partial_profile(
                name, value_ids, sender_fields, style_id),
            "receiver_profile": render_partial_profile(
                name, value_ids, receiver_fields, style_id),
            "full_profile": render_partial_profile(
                name, value_ids, list(range(cfg.num_fields)), style_id),
            "question_text": render_question(name, question_field, phrasing_id),
        })
    return samples


def build_hidden_profile_dataset(cfg: HiddenProfileConfig):
    return {
        "train": build_hidden_profile_split(cfg.train_size, cfg.seed, cfg),
        "val": build_hidden_profile_split(cfg.val_size, cfg.seed + 1, cfg),
        "test": build_hidden_profile_split(cfg.test_size, cfg.seed + 2, cfg),
    }


# ═══════════════════════════════════════════════════════════════════
# Feature extraction — separate sender & receiver hidden states
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_hidden_profile_features(
    samples: list[dict], cfg: HiddenProfileConfig,
    tokenizer, model, split_name: str,
) -> dict[str, torch.Tensor]:
    sender_hidden_chunks, sender_mask_chunks = [], []
    receiver_hidden_chunks, receiver_mask_chunks = [], []
    question_hidden_chunks = []
    answer_ids, field_ids, style_ids, state_ids = [], [], [], []
    needs_comm_list = []
    sender_field_list, receiver_field_list = [], []

    total = len(samples)
    for start in range(0, total, cfg.extractor_batch_size):
        batch = samples[start:start + cfg.extractor_batch_size]

        # sender sees only sender_fields
        sender_texts = [
            f"Read the profile carefully.\nProfile:\n{item['sender_profile']}"
            for item in batch
        ]
        # receiver sees only receiver_fields
        receiver_texts = [
            f"Read the profile carefully.\nProfile:\n{item['receiver_profile']}"
            for item in batch
        ]
        question_texts = [
            f"Question:\n{item['question_text']}\nAnswer with one short phrase."
            for item in batch
        ]

        # extract sender hidden
        s_inputs = tokenizer(sender_texts, return_tensors="pt",
                             padding="max_length", truncation=True,
                             max_length=cfg.max_doc_length).to(model.device)
        s_out = model(**s_inputs, output_hidden_states=True,
                      return_dict=True, use_cache=False)
        sender_hidden_chunks.append(
            s_out.hidden_states[-1].to(dtype=torch.float16).cpu())
        sender_mask_chunks.append(s_inputs["attention_mask"].cpu())

        # extract receiver hidden
        r_inputs = tokenizer(receiver_texts, return_tensors="pt",
                             padding="max_length", truncation=True,
                             max_length=cfg.max_doc_length).to(model.device)
        r_out = model(**r_inputs, output_hidden_states=True,
                      return_dict=True, use_cache=False)
        receiver_hidden_chunks.append(
            r_out.hidden_states[-1].to(dtype=torch.float16).cpu())
        receiver_mask_chunks.append(r_inputs["attention_mask"].cpu())

        # extract question hidden
        q_inputs = tokenizer(question_texts, return_tensors="pt",
                             padding="max_length", truncation=True,
                             max_length=cfg.max_question_length).to(model.device)
        q_out = model(**q_inputs, output_hidden_states=True,
                      return_dict=True, use_cache=False)
        q_hidden = last_token_pool(
            q_out.hidden_states[-1], q_inputs["attention_mask"])
        question_hidden_chunks.append(q_hidden.to(dtype=torch.float16).cpu())

        for item in batch:
            answer_ids.append(item["answer_id"])
            field_ids.append(item["question_field"])
            style_ids.append(item["style_id"])
            state_ids.append(item["state_ids"])
            needs_comm_list.append(item["needs_comm"])
            sender_field_list.append(item["sender_fields"])
            receiver_field_list.append(item["receiver_fields"])

        done = min(start + len(batch), total)
        if (start // cfg.extractor_batch_size) % 25 == 0:
            print(f"[{split_name}] extracted {done}/{total}")

    return {
        "sender_hidden": torch.cat(sender_hidden_chunks, dim=0),
        "sender_mask": torch.cat(sender_mask_chunks, dim=0),
        "receiver_hidden": torch.cat(receiver_hidden_chunks, dim=0),
        "receiver_mask": torch.cat(receiver_mask_chunks, dim=0),
        "question_hidden": torch.cat(question_hidden_chunks, dim=0),
        "answer_id": torch.tensor(answer_ids, dtype=torch.long),
        "question_field": torch.tensor(field_ids, dtype=torch.long),
        "style_id": torch.tensor(style_ids, dtype=torch.long),
        "state_ids": torch.tensor(state_ids, dtype=torch.long),
        "needs_comm": torch.tensor(needs_comm_list, dtype=torch.long),
    }


# ═══════════════════════════════════════════════════════════════════
# Model — HiddenProfileModel
# ═══════════════════════════════════════════════════════════════════

class HiddenProfileModel(nn.Module):
    """
    Sender: compresses sender_hidden → slots via DocSlotBottleneck
    Receiver: cross-attends question over [sender_slots ; receiver_context]
              to produce answer logits
    """
    def __init__(self, hidden_dim: int, num_answers: int,
                 cfg: HiddenProfileConfig):
        super().__init__()
        self.cfg = cfg
        # sender bottleneck: compress sender hidden → slots
        self.sender_bottleneck = DocSlotBottleneck(
            hidden_dim=hidden_dim,
            num_slots=cfg.num_slots,
            slot_dim=cfg.slot_dim,
            num_heads=cfg.num_heads,
            variational=cfg.variational,
        )
        # receiver: project own hidden to slot_dim for local context
        self.receiver_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, cfg.slot_dim),
            nn.GELU(),
        )
        # question projection
        self.query_proj = nn.Linear(hidden_dim, cfg.slot_dim)
        # cross-attention: query attends over sender_slots
        self.cross_attn = nn.MultiheadAttention(
            cfg.slot_dim, cfg.num_heads, batch_first=True)
        # readout: [query, cross_attn_output, receiver_context] → answer
        self.readout = nn.Sequential(
            nn.LayerNorm(3 * cfg.slot_dim),
            nn.Linear(3 * cfg.slot_dim, 4 * cfg.slot_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * cfg.slot_dim, num_answers),
        )
        # style adversary on sender slots
        self.style_adversary = StyleAdversary(cfg.num_styles, cfg.slot_dim)

    def forward(self, sender_hidden, sender_mask, receiver_hidden,
                receiver_mask, question_hidden):
        training = self.training
        # sender encodes its partial profile → slots
        sender_slots, sender_stats = self.sender_bottleneck(
            sender_hidden, sender_mask,
            apply_channel=training,
            slot_dropout=self.cfg.slot_dropout if training else 0.0,
            noise_std=self.cfg.noise_std if training else 0.0,
        )
        # receiver: mean-pool own hidden → local context
        mask_f = receiver_mask.unsqueeze(-1).float()
        r_pooled = (receiver_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        r_context = self.receiver_proj(r_pooled)  # [B, slot_dim]
        # question → query
        query = self.query_proj(question_hidden).unsqueeze(1)  # [B, 1, slot_dim]
        # cross-attend query over sender_slots
        context, attn_weights = self.cross_attn(
            query, sender_slots, sender_slots, need_weights=True)
        # combine: [query, sender_context, receiver_context]
        joint = torch.cat([
            query.squeeze(1),
            context.squeeze(1),
            r_context,
        ], dim=-1)
        logits = self.readout(joint)
        # style adversary on sender slots (pooled)
        sender_pooled = sender_slots.mean(dim=1)
        style_logits = self.style_adversary(
            grad_reverse(sender_pooled, self.cfg.style_adv_lambda))
        return {
            "logits": logits,
            "sender_stats": sender_stats,
            "style_logits": style_logits,
            "attn_weights": attn_weights,
            "sender_slots": sender_slots,
            "receiver_context": r_context,
        }


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def compute_loss(out: dict, answer_id, style_id, cfg: HiddenProfileConfig):
    ce = F.cross_entropy(out["logits"], answer_id)
    loss = ce
    extras = {"ce": ce.item()}
    # KL for variational sender
    stats = out["sender_stats"]
    if cfg.variational and stats["mu"] is not None:
        mu, logvar = stats["mu"], stats["logvar"]
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        loss = loss + cfg.rate_weight * kl
        extras["kl"] = kl.item()
    # style adversary
    if cfg.style_adv_weight > 0:
        style_ce = F.cross_entropy(out["style_logits"], style_id)
        loss = loss + cfg.style_adv_weight * style_ce
        extras["style_ce"] = style_ce.item()
    return loss, extras


def train_epoch(model, loader, optimizer, device, cfg):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        (s_hid, s_mask, r_hid, r_mask, q_hid,
         ans, field, style, state, needs) = [b.to(device) for b in batch]
        # cast fp16 features to fp32 for model computation
        s_hid, r_hid, q_hid = s_hid.float(), r_hid.float(), q_hid.float()
        # masks stay as long tensors for key_padding_mask
        out = model(s_hid, s_mask, r_hid, r_mask, q_hid)
        loss, _ = compute_loss(out, ans, style, cfg)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * ans.size(0)
        total_correct += (out["logits"].argmax(-1) == ans).sum().item()
        total_n += ans.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    comm_correct, comm_n = 0, 0
    local_correct, local_n = 0, 0
    for batch in loader:
        (s_hid, s_mask, r_hid, r_mask, q_hid,
         ans, field, style, state, needs) = [b.to(device) for b in batch]
        s_hid, r_hid, q_hid = s_hid.float(), r_hid.float(), q_hid.float()
        # masks stay as long tensors for key_padding_mask
        out = model(s_hid, s_mask, r_hid, r_mask, q_hid)
        loss, _ = compute_loss(out, ans, style, cfg)
        preds = out["logits"].argmax(-1)
        correct = (preds == ans)
        total_loss += loss.item() * ans.size(0)
        total_correct += correct.sum().item()
        total_n += ans.size(0)
        # split by needs_comm
        comm_mask = (needs == 1)
        local_mask = (needs == 0)
        comm_correct += correct[comm_mask].sum().item()
        comm_n += comm_mask.sum().item()
        local_correct += correct[local_mask].sum().item()
        local_n += local_mask.sum().item()
    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
        "comm_acc": comm_correct / comm_n if comm_n > 0 else 0.0,
        "local_acc": local_correct / local_n if local_n > 0 else 0.0,
        "comm_n": comm_n,
        "local_n": local_n,
    }


# ═══════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════

class ReceiverOnlyModel(nn.Module):
    """Baseline: receiver sees only its own profile + question, no sender."""
    def __init__(self, hidden_dim, num_answers, cfg):
        super().__init__()
        d = hidden_dim // 8
        self.r_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, d), nn.GELU())
        self.q_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, d), nn.GELU())
        self.readout = nn.Sequential(
            nn.LayerNorm(d * 2), nn.Linear(d * 2, d), nn.GELU(),
            nn.Linear(d, num_answers))

    def forward(self, sender_hidden, sender_mask, receiver_hidden,
                receiver_mask, question_hidden):
        # ignore sender entirely
        # mean pool receiver hidden
        mask_f = receiver_mask.unsqueeze(-1).float()
        r_pooled = (receiver_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        r = self.r_proj(r_pooled)
        q = self.q_proj(question_hidden)
        logits = self.readout(torch.cat([r, q], dim=-1))
        return {"logits": logits, "sender_stats": {"mu": None, "logvar": None},
                "style_logits": torch.zeros(logits.size(0), 6, device=logits.device)}


class FullContextModel(nn.Module):
    """Upper bound: receiver sees full profile (all 8 fields)."""
    def __init__(self, hidden_dim, num_answers, cfg):
        super().__init__()
        d = hidden_dim // 8
        self.doc_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, d), nn.GELU())
        self.q_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, d), nn.GELU())
        self.readout = nn.Sequential(
            nn.LayerNorm(d * 2), nn.Linear(d * 2, d), nn.GELU(),
            nn.Linear(d, num_answers))

    def forward(self, full_hidden, full_mask, question_hidden):
        mask_f = full_mask.unsqueeze(-1).float()
        pooled = (full_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        d = self.doc_proj(pooled)
        q = self.q_proj(question_hidden)
        return self.readout(torch.cat([d, q], dim=-1))


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def make_loader(feats, cfg, shuffle=True):
    tensors = [
        feats["sender_hidden"], feats["sender_mask"],
        feats["receiver_hidden"], feats["receiver_mask"],
        feats["question_hidden"], feats["answer_id"],
        feats["question_field"], feats["style_id"],
        feats["state_ids"], feats["needs_comm"],
    ]
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=cfg.train_batch_size,
                      shuffle=shuffle, drop_last=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split-strategy", type=str, default="random",
                        choices=["random", "fixed"])
    args = parser.parse_args()

    cfg = HiddenProfileConfig()
    if args.model_path:
        cfg.model_path = args.model_path
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cfg.split_strategy = args.split_strategy

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 5120  # Qwen3-14B
    num_answers = len(ANSWER_VOCAB)

    # ── Step 1: Generate data ──
    print("=== Generating Hidden Profile dataset ===")
    dataset = build_hidden_profile_dataset(cfg)
    for split_name, samples in dataset.items():
        comm_count = sum(s["needs_comm"] for s in samples)
        print(f"  {split_name}: {len(samples)} samples, "
              f"{comm_count} need communication ({comm_count/len(samples):.1%})")

    # save dataset metadata
    with open(out_dir / "dataset_meta.json", "w") as f:
        meta = {
            "config": asdict(cfg),
            "splits": {k: len(v) for k, v in dataset.items()},
            "comm_ratio": {
                k: sum(s["needs_comm"] for s in v) / len(v)
                for k, v in dataset.items()
            },
        }
        json.dump(meta, f, indent=2)

    # ── Step 2: Extract features ──
    if not args.skip_extract:
        print(f"\n=== Loading Qwen model from {cfg.model_path} ===")
        tokenizer, qwen_model = load_qwen_model(cfg.model_path)
        for split_name, samples in dataset.items():
            print(f"\n--- Extracting {split_name} features ---")
            feats = extract_hidden_profile_features(
                samples, cfg, tokenizer, qwen_model, split_name)
            torch.save(feats, out_dir / f"{split_name}_features.pt")
            print(f"  Saved {split_name}_features.pt "
                  f"(sender: {feats['sender_hidden'].shape}, "
                  f"receiver: {feats['receiver_hidden'].shape})")
        del qwen_model, tokenizer
        torch.cuda.empty_cache()
    else:
        print("Skipping feature extraction (--skip-extract)")

    if args.extract_only:
        print("Done (extract only).")
        return

    # ── Step 3: Train ──
    print("\n=== Loading features ===")
    train_feats = torch.load(out_dir / "train_features.pt", weights_only=True)
    val_feats = torch.load(out_dir / "val_features.pt", weights_only=True)
    test_feats = torch.load(out_dir / "test_features.pt", weights_only=True)

    train_loader = make_loader(train_feats, cfg, shuffle=True)
    val_loader = make_loader(val_feats, cfg, shuffle=False)
    test_loader = make_loader(test_feats, cfg, shuffle=False)

    # --- Train HiddenProfileModel (ours) ---
    print("\n=== Training HiddenProfileModel (slot communication) ===")
    model = HiddenProfileModel(hidden_dim, num_answers, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val_acc, best_epoch = 0.0, 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, cfg)
        val_metrics = evaluate(model, val_loader, device, cfg)
        dt = time.time() - t0
        print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.3f} | val_acc={val_metrics['acc']:.3f} "
              f"(comm={val_metrics['comm_acc']:.3f} "
              f"local={val_metrics['local_acc']:.3f}) | {dt:.1f}s")
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    model.load_state_dict(torch.load(out_dir / "best_model.pt",
                                     weights_only=True))
    test_metrics = evaluate(model, test_loader, device, cfg)
    print(f"\n  Best epoch: {best_epoch} | Test results:")
    print(f"    Overall acc:  {test_metrics['acc']:.3f}")
    print(f"    Comm acc:     {test_metrics['comm_acc']:.3f} "
          f"(n={test_metrics['comm_n']})")
    print(f"    Local acc:    {test_metrics['local_acc']:.3f} "
          f"(n={test_metrics['local_n']})")

    # message size
    sender_msg_bytes = cfg.num_slots * cfg.slot_dim * 2  # fp16
    print(f"    Message size: {sender_msg_bytes} bytes "
          f"({cfg.num_slots} slots x {cfg.slot_dim} dim x fp16)")

    # --- Train ReceiverOnly baseline ---
    print("\n=== Training ReceiverOnly baseline (no communication) ===")
    baseline = ReceiverOnlyModel(hidden_dim, num_answers, cfg).to(device)
    opt_b = torch.optim.AdamW(baseline.parameters(),
                               lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_b_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        baseline.train()
        for batch in train_loader:
            (s_hid, s_mask, r_hid, r_mask, q_hid,
             ans, field, style, state, needs) = [b.to(device) for b in batch]
            s_hid, r_hid, q_hid = s_hid.float(), r_hid.float(), q_hid.float()
            # masks stay as long tensors for key_padding_mask
            out = baseline(s_hid, s_mask, r_hid, r_mask, q_hid)
            loss = F.cross_entropy(out["logits"], ans)
            opt_b.zero_grad(); loss.backward(); opt_b.step()
        val_m = evaluate_simple(baseline, val_loader, device)
        if val_m["acc"] > best_b_acc:
            best_b_acc = val_m["acc"]
            torch.save(baseline.state_dict(), out_dir / "baseline_model.pt")

    baseline.load_state_dict(torch.load(out_dir / "baseline_model.pt",
                                        weights_only=True))
    bl_test = evaluate(baseline, test_loader, device, cfg)
    print(f"  ReceiverOnly test: overall={bl_test['acc']:.3f} "
          f"comm={bl_test['comm_acc']:.3f} local={bl_test['local_acc']:.3f}")

    # ── Step 4: Save results ──
    results = {
        "slot_comm": {
            "test_acc": test_metrics["acc"],
            "test_comm_acc": test_metrics["comm_acc"],
            "test_local_acc": test_metrics["local_acc"],
            "message_bytes": sender_msg_bytes,
            "best_epoch": best_epoch,
        },
        "receiver_only": {
            "test_acc": bl_test["acc"],
            "test_comm_acc": bl_test["comm_acc"],
            "test_local_acc": bl_test["local_acc"],
            "message_bytes": 0,
        },
        "config": asdict(cfg),
    }
    with open(out_dir / "hidden_profile_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'hidden_profile_results.json'}")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("Hidden Profile Experiment Summary")
    print("=" * 70)
    print(f"{'Method':<25} {'Overall':>8} {'Comm':>8} {'Local':>8} {'Msg':>10}")
    print("-" * 70)
    print(f"{'Receiver Only':<25} {bl_test['acc']:>8.3f} "
          f"{bl_test['comm_acc']:>8.3f} {bl_test['local_acc']:>8.3f} "
          f"{'0 B':>10}")
    print(f"{'Slot Communication':<25} {test_metrics['acc']:>8.3f} "
          f"{test_metrics['comm_acc']:>8.3f} {test_metrics['local_acc']:>8.3f} "
          f"{sender_msg_bytes:>8} B")
    print("=" * 70)


def evaluate_simple(model, loader, device):
    """Simple evaluate for baseline models."""
    model.eval()
    correct, total = 0, 0
    comm_correct, comm_n = 0, 0
    local_correct, local_n = 0, 0
    with torch.no_grad():
        for batch in loader:
            (s_hid, s_mask, r_hid, r_mask, q_hid,
             ans, field, style, state, needs) = [b.to(device) for b in batch]
            s_hid, r_hid, q_hid = s_hid.float(), r_hid.float(), q_hid.float()
            # masks stay as long tensors for key_padding_mask
            out = model(s_hid, s_mask, r_hid, r_mask, q_hid)
            preds = out["logits"].argmax(-1)
            c = (preds == ans)
            correct += c.sum().item()
            total += ans.size(0)
            comm_mask = (needs == 1)
            local_mask = (needs == 0)
            comm_correct += c[comm_mask].sum().item()
            comm_n += comm_mask.sum().item()
            local_correct += c[local_mask].sum().item()
            local_n += local_mask.sum().item()
    return {
        "acc": correct / total,
        "comm_acc": comm_correct / comm_n if comm_n > 0 else 0.0,
        "local_acc": local_correct / local_n if local_n > 0 else 0.0,
    }


if __name__ == "__main__":
    main()
