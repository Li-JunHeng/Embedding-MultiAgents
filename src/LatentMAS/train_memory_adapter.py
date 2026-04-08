"""
Train the LatentMemoryAdapter for MemoryMAS.

Mirrors train_compressor.py structure:
  1. Collect hidden sequences under MemoryMAS-style prefix chain
     (adapter + PerSampleMemoryBank, matching inference distribution).
  2. Loss: compress-retrieve-reconstruct + last-step fidelity
     + direction alignment + diversity regularization.
  3. Role embeddings trained jointly with projection weights.

Usage:
  python train_memory_adapter.py \
    --model_name Qwen/Qwen3-8B \
    --task gsm8k \
    --collect_samples 300 \
    --memory_dim 256 \
    --latent_steps 10 \
    --epochs 100 \
    --output_dir results/trained_adapter

Optional: --adapter_teacher_path to build prefixes with a trained teacher
(more stable collection). If omitted, uses a randomly initialized adapter.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import ModelWrapper
from memory_bank import LatentMemoryAdapter, PerSampleMemoryBank, ROLE_MAP
from methods import default_agents
from prompts import build_agent_message_sequential_latent_mas
from data import load_gsm8k, load_arc_challenge, load_gpqa_diamond, load_arc_easy


def collect_hidden_states_memory_aligned(
    model: ModelWrapper,
    task_data: list,
    args: argparse.Namespace,
    teacher: Optional[LatentMemoryAdapter] = None,
    max_samples: int = 500,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect (N * 3, 1+latent_steps, D) hidden sequences under MemoryMAS-style
    prefix chain (adapter + PerSampleMemoryBank for previous agents).

    Returns:
        hidden_data: (total, 1+latent_steps, D) float16
        role_ids:    (total,) long
    """
    print(
        f"[collect] Memory-aligned, max_samples={max_samples}, "
        f"latent_steps={args.latent_steps}"
    )

    device = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype
    agents = default_agents()

    if teacher is None:
        d_model = model.model.config.hidden_size
        teacher = LatentMemoryAdapter(
            d_model=d_model,
            memory_dim=args.memory_dim,
        ).to(device).to(dtype)
        teacher.eval()
        print("[collect] No teacher — using random adapter for prefix chain")
    else:
        teacher = teacher.to(device).to(dtype).eval()

    all_hidden_seqs: List[torch.Tensor] = []
    all_role_ids: List[int] = []

    for i, item in enumerate(task_data):
        if i >= max_samples:
            break
        if i % 50 == 0:
            print(f"  [{i}/{max_samples}]")

        bank = PerSampleMemoryBank()

        for agent in agents:
            if agent.role == "judger":
                continue

            messages = build_agent_message_sequential_latent_mas(
                role=agent.role,
                question=item["question"],
                context="",
                method="memory_mas",
                args=args,
            )
            prompt = model.render_chat(messages, add_generation_prompt=True)
            if getattr(args, "think", False):
                prompt = f"{prompt}<think>"
            enc = model.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                # Read memory prefix (matches memory_mas._read_memory_prefix)
                memory_prefix = None
                if not bank.is_empty():
                    _, _, query_hidden = model.rollout_latent_sequence(
                        input_ids,
                        attention_mask=attention_mask,
                        latent_steps=0,
                    )
                    memory_prefix = bank.read(query_hidden, teacher)

                # Forward with prefix (matches memory_mas.run_item for non-judger)
                _, hidden_seq, _ = model.rollout_latent_sequence(
                    input_ids,
                    attention_mask=attention_mask,
                    latent_steps=args.latent_steps,
                    prefix_embeds=memory_prefix,
                )

                all_hidden_seqs.append(hidden_seq.cpu().to(torch.float16))
                all_role_ids.append(ROLE_MAP[agent.role])

                # Add to bank for next agent's prefix
                hidden_for_bank = hidden_seq.to(device=device, dtype=dtype)
                bank.add(agent.role, hidden_for_bank, teacher)

        torch.cuda.empty_cache()

    all_h = torch.cat(all_hidden_seqs, dim=0)
    all_r = torch.tensor(all_role_ids, dtype=torch.long)
    print(f"[collect] {all_h.shape[0]} sequences, shape={tuple(all_h.shape)}")
    role_counts = {k: 0 for k in ROLE_MAP}
    for rid in all_role_ids:
        for name, idx in ROLE_MAP.items():
            if idx == rid:
                role_counts[name] += 1
    print(f"[collect] Roles: {role_counts}")
    return all_h, all_r


def _adapter_training_loss(
    adapter: LatentMemoryAdapter,
    batch_h: torch.Tensor,
    batch_roles: torch.Tensor,
) -> torch.Tensor:
    """
    Loss = 1.0 * recon_mse + 0.5 * last_mse + 0.35 * cos_loss + 0.1 * div_loss

    Simulates the compress→read→reconstruct cycle:
      - Keys:   LayerNorm → +role_embed → Linear  (matches bank.add)
      - Queries: LayerNorm → Linear               (matches adapter.read)
      - Values:  raw hidden states                 (matches bank storage)
    """
    B, L, D = batch_h.shape

    # Shared LayerNorm (computed once)
    h_normed = adapter.layer_norm(batch_h)  # (B, L, D)

    # --- Keys: with role embedding (vectorized) ---
    role_embs = adapter.role_embeddings(batch_roles)  # (B, D)
    h_with_role = h_normed + role_embs.unsqueeze(1)  # (B, L, D)
    compressed = adapter.memory_proj(h_with_role)  # (B, L, mem_dim)

    # --- Queries: without role (matches adapter.read internals) ---
    query_proj = adapter.memory_proj(h_normed)  # (B, L, mem_dim)

    # --- Attention + retrieval ---
    attn_logits = torch.bmm(query_proj, compressed.transpose(1, 2))  # (B, L, L)
    attn_logits = attn_logits / math.sqrt(adapter.memory_dim)
    attn_weights = F.softmax(attn_logits, dim=-1)  # (B, L, L)
    retrieved = torch.bmm(attn_weights, batch_h)  # (B, L, D)

    # --- 1) Compress-retrieve-reconstruct ---
    recon_mse = F.mse_loss(retrieved, batch_h)

    # --- 2) Last-step fidelity (most informative for handoff) ---
    last_mse = F.mse_loss(retrieved[:, -1], batch_h[:, -1])

    # --- 3) Pooled direction alignment ---
    inp_pool = batch_h.mean(dim=1)
    ret_pool = retrieved.mean(dim=1)
    cos_loss = 1.0 - F.cosine_similarity(ret_pool, inp_pool, dim=-1).mean()

    # --- 4) Diversity: entropy of last-query attention (prevent collapse) ---
    last_attn = attn_weights[:, -1, :]  # (B, L)
    entropy = -(last_attn * (last_attn + 1e-8).log()).sum(dim=-1).mean()
    max_ent = math.log(L) if L > 1 else 1.0
    div_loss = 1.0 - entropy / max_ent

    loss = 1.0 * recon_mse + 0.5 * last_mse + 0.35 * cos_loss + 0.1 * div_loss
    return loss


def train_adapter(
    hidden_data: torch.Tensor,
    role_data: torch.Tensor,
    d_model: int,
    memory_dim: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cuda:0",
) -> LatentMemoryAdapter:
    print(f"\n[train] memory_dim={memory_dim}, params="
          f"{sum(p.numel() for p in LatentMemoryAdapter(d_model, memory_dim).parameters()):,}")

    adapter = LatentMemoryAdapter(
        d_model=d_model,
        memory_dim=memory_dim,
    ).to(device).float()

    dataset = TensorDataset(hidden_data, role_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        adapter.train()
        total, total_n = 0.0, 0
        for batch_h, batch_roles in loader:
            batch_h = batch_h.float().to(device)
            batch_roles = batch_roles.to(device)
            loss = _adapter_training_loss(adapter, batch_h, batch_roles)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            total += loss.item() * batch_h.size(0)
            total_n += batch_h.size(0)

        scheduler.step()
        avg = total / max(total_n, 1)
        best_loss = min(best_loss, avg)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={avg:.6f} (best={best_loss:.6f})")

    return adapter


def _load_task(args: argparse.Namespace) -> list:
    if args.task == "gsm8k":
        return list(load_gsm8k())
    if args.task == "arc_challenge":
        return list(load_arc_challenge())
    if args.task == "arc_easy":
        return list(load_arc_easy())
    if args.task == "gpqa":
        return list(load_gpqa_diamond())
    raise ValueError(f"Unknown task: {args.task}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LatentMemoryAdapter for MemoryMAS"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--task", type=str, default="gsm8k",
                        help="gsm8k | arc_challenge | arc_easy | gpqa")
    parser.add_argument("--collect_samples", type=int, default=300)
    parser.add_argument("--memory_dim", type=int, default=256,
                        help="Teacher memory_dim for collection prefix chain")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/trained_adapter")
    parser.add_argument("--think", action="store_true", default=True)
    parser.add_argument("--prompt", type=str, default="sequential")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--adapter_teacher_path", type=str, default=None,
                        help="Optional .pt for prefix chain during collection "
                             "(same arch as memory_dim)")
    parser.add_argument("--train_dims", type=str, default="256,128,64",
                        help="Comma-separated memory_dims to train and save")
    parser.add_argument("--cache_tag", type=str, default="",
                        help="Suffix for cache file to avoid mixing caches")
    args = parser.parse_args()
    args.method = "memory_mas"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    print(f"Loading {args.model_name}...")
    model = ModelWrapper(args.model_name, device, args=args)
    d_model = model.model.config.hidden_size

    task_data = _load_task(args)
    tag = args.cache_tag or "mem_aligned_v1"
    cache_h = out_dir / f"{args.task}_hiddens_n{args.collect_samples}_{tag}.pt"
    cache_r = out_dir / f"{args.task}_roles_n{args.collect_samples}_{tag}.pt"

    teacher = None
    if args.adapter_teacher_path:
        teacher = LatentMemoryAdapter(
            d_model=d_model,
            memory_dim=args.memory_dim,
        ).to(device).to(next(model.model.parameters()).dtype)
        state = torch.load(
            args.adapter_teacher_path, map_location=device, weights_only=True
        )
        teacher.load_state_dict(state)
        print(f"[collect] Teacher loaded from {args.adapter_teacher_path}")

    if cache_h.exists() and cache_r.exists():
        print(f"Loading cached hidden states from {cache_h}")
        hidden_data = torch.load(cache_h, weights_only=True)
        role_data = torch.load(cache_r, weights_only=True)
    else:
        hidden_data, role_data = collect_hidden_states_memory_aligned(
            model, task_data, args,
            teacher=teacher,
            max_samples=args.collect_samples,
        )
        torch.save(hidden_data, cache_h)
        torch.save(role_data, cache_r)
        print(f"Saved to {cache_h}, {cache_r}")

    del model
    torch.cuda.empty_cache()

    dims = [int(x) for x in args.train_dims.split(",") if x.strip()]
    for md in dims:
        adapter = train_adapter(
            hidden_data,
            role_data,
            d_model,
            memory_dim=md,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=str(device),
        )
        save_path = out_dir / f"adapter_md{md}.pt"
        torch.save(adapter.state_dict(), save_path)
        print(f"Saved {save_path}")

    print("\nDone! Evaluate with:")
    print("  python run.py --method memory_mas --memory_dim D --adapter_path <path>")


if __name__ == "__main__":
    main()
