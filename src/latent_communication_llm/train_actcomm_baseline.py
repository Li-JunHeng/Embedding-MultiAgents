"""Activation Communication Baseline (ActComm)

Two variants:
  ActComm_Pool : message = mean_pool(last_layer_doc_hidden)  [4096 fp16 = 8 KB]
  ActComm_Full : message = all_positions(last_layer_doc_hidden) [192x4096 fp16 ≈ 1.5 MB]
                 receiver uses query-conditioned cross-attention over the full sequence

Both use pre-extracted features from run_qwen_handoff.py (train/val/test_features.pt).
Training time: ~20-40 min on a single 3090.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJ_DIM = 512       # internal projection dim for ActComm_Full cross-attn


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def mean_pool(doc_hidden: torch.Tensor, doc_mask: torch.Tensor) -> torch.Tensor:
    """[B, L, H] + [B, L] -> [B, H]  (masked mean pool)."""
    mask = doc_mask.float().unsqueeze(-1)
    return (doc_hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-8)


# ---------------------------------------------------------------------------
# Model: ActComm_Pool
# ---------------------------------------------------------------------------

class ActCommPoolModel(nn.Module):
    """Message = mean-pooled doc hidden state (hidden_dim fp16 = 2*hidden_dim B)."""

    def __init__(self, hidden_dim: int, num_answers: int) -> None:
        super().__init__()
        d = hidden_dim // 8
        self.msg_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d),
            nn.GELU(),
        )
        self.q_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d),
            nn.GELU(),
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, num_answers),
        )

    def forward(
        self,
        doc_hidden: torch.Tensor,
        doc_mask: torch.Tensor,
        question_hidden: torch.Tensor,
    ) -> torch.Tensor:
        act = mean_pool(doc_hidden, doc_mask)   # [B, H]
        joint = torch.cat([self.msg_proj(act), self.q_proj(question_hidden)], dim=-1)
        return self.readout(joint)

    @staticmethod
    def message_floats(hidden_dim: int) -> int:
        return hidden_dim

    @staticmethod
    def message_bytes(hidden_dim: int) -> int:
        return hidden_dim * 2


# ---------------------------------------------------------------------------
# Model: ActComm_Full
# ---------------------------------------------------------------------------

class ActCommFullModel(nn.Module):
    """Message = full last-layer doc hidden sequence (seq_len × hidden_dim fp16).
    Receiver uses query-conditioned cross-attention (like our slot receiver but
    over the raw activation sequence instead of learned slots)."""

    def __init__(self, hidden_dim: int, num_answers: int, proj_dim: int = PROJ_DIM) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.doc_proj = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.q_proj   = nn.Linear(hidden_dim, proj_dim, bias=False)
        num_heads = 8
        self.cross_attn = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True)
        self.readout = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, num_answers),
        )

    def forward(
        self,
        doc_hidden: torch.Tensor,   # [B, L, H]
        doc_mask: torch.Tensor,     # [B, L]
        question_hidden: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        kv = self.doc_proj(doc_hidden)           # [B, L, P]
        q  = self.q_proj(question_hidden).unsqueeze(1)  # [B, 1, P]
        ctx, _ = self.cross_attn(
            q, kv, kv,
            key_padding_mask=(doc_mask == 0),
            need_weights=False,
        )
        joint = torch.cat([q.squeeze(1), ctx.squeeze(1)], dim=-1)
        return self.readout(joint)

    @staticmethod
    def message_floats(hidden_dim: int, seq_len: int = 192) -> int:
        return seq_len * hidden_dim

    @staticmethod
    def message_bytes(hidden_dim: int, seq_len: int = 192) -> int:
        return seq_len * hidden_dim * 2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_loader(features: dict[str, torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        features["doc_hidden"],
        features["doc_mask"],
        features["question_hidden"],
        features["answer_id"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    label: str,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for doc_h, doc_m, q_h, ans_id in train_loader:
            doc_h  = doc_h.to(device, dtype=torch.float32)
            doc_m  = doc_m.to(device)
            q_h    = q_h.to(device, dtype=torch.float32)
            ans_id = ans_id.to(device)

            logits = model(doc_h, doc_m, q_h)
            loss = F.cross_entropy(logits, ans_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ---- val ----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for doc_h, doc_m, q_h, ans_id in val_loader:
                doc_h  = doc_h.to(device, dtype=torch.float32)
                doc_m  = doc_m.to(device)
                q_h    = q_h.to(device, dtype=torch.float32)
                ans_id = ans_id.to(device)
                pred = model(doc_h, doc_m, q_h).argmax(dim=-1)
                correct += (pred == ans_id).sum().item()
                total   += len(ans_id)
        val_acc = correct / total
        print(f"[{label}] epoch {epoch:3d} | loss {total_loss/len(train_loader):.4f} | val {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ---- test ----
    model.load_state_dict(best_state)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for doc_h, doc_m, q_h, ans_id in test_loader:
            doc_h  = doc_h.to(device, dtype=torch.float32)
            doc_m  = doc_m.to(device)
            q_h    = q_h.to(device, dtype=torch.float32)
            ans_id = ans_id.to(device)
            pred = model(doc_h, doc_m, q_h).argmax(dim=-1)
            correct += (pred == ans_id).sum().item()
            total   += len(ans_id)
    test_acc = correct / total
    print(f"[{label}] best_val={best_val_acc:.4f}  test={test_acc:.4f}")
    return {"best_val_accuracy": best_val_acc, "test_accuracy": test_acc, "best_state": best_state}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",    type=Path, required=True)
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--gpu",        type=int,   default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- load pre-extracted features ----
    splits: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        path = args.run_dir / f"{split}_features.pt"
        splits[split] = torch.load(path, map_location="cpu")
        print(f"Loaded {split}: {splits[split]['doc_hidden'].shape}")

    num_answers = int(splits["train"]["answer_id"].max().item()) + 1
    print(f"num_answers = {num_answers}")

    hidden_dim = int(splits["train"]["doc_hidden"].size(-1))
    print(f"hidden_dim = {hidden_dim} (from features)")

    train_loader = make_loader(splits["train"], args.batch_size, shuffle=True)
    val_loader   = make_loader(splits["val"],   args.batch_size, shuffle=False)
    test_loader  = make_loader(splits["test"],  args.batch_size, shuffle=False)

    results = {}

    # ---- ActComm_Pool ----
    print(f"\n=== ActComm_Pool (mean pool, {hidden_dim} fp16) ===")
    pool_model = ActCommPoolModel(hidden_dim, num_answers).to(device)
    pool_res = train_one(
        pool_model, train_loader, val_loader, test_loader,
        epochs=args.epochs, lr=args.lr, device=device, label="Pool",
    )
    torch.save(pool_res["best_state"], args.run_dir / "actcomm_pool_model.pt")
    results["actcomm_pool"] = {
        "method": "ActComm (Mean Pool)",
        "message_floats": ActCommPoolModel.message_floats(hidden_dim),
        "message_bytes": ActCommPoolModel.message_bytes(hidden_dim),
        "best_val_accuracy": pool_res["best_val_accuracy"],
        "test_accuracy": pool_res["test_accuracy"],
    }

    # ---- ActComm_Full ----
    print(f"\n=== ActComm_Full (full sequence cross-attn, seq×{hidden_dim} fp16) ===")
    full_model = ActCommFullModel(hidden_dim, num_answers).to(device)
    full_res = train_one(
        full_model, train_loader, val_loader, test_loader,
        epochs=args.epochs, lr=args.lr, device=device, label="Full",
    )
    torch.save(full_res["best_state"], args.run_dir / "actcomm_full_model.pt")
    seq_len = splits["test"]["doc_mask"].sum(dim=1).float().mean().item()
    results["actcomm_full"] = {
        "method": "ActComm (Full Seq)",
        "message_floats": int(seq_len) * hidden_dim,
        "message_bytes": int(seq_len) * hidden_dim * 2,
        "avg_nonpad_tokens": seq_len,
        "best_val_accuracy": full_res["best_val_accuracy"],
        "test_accuracy": full_res["test_accuracy"],
    }

    # ---- save ----
    out_path = args.run_dir / "actcomm_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
    for k, v in results.items():
        print(f"  {v['method']:35s}  val={v['best_val_accuracy']:.4f}  test={v['test_accuracy']:.4f}  msg={v['message_bytes']} B")


if __name__ == "__main__":
    main()
