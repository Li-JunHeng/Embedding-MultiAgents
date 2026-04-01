"""
Train the SlotAttentionCompressor for SlotMAS.

Key fixes (vs. older version):
  1. Collect hidden sequences with the SAME forward as SlotMAS: prefix = decoded
     slots from previous agents (not full KV stack).
  2. Each latent step in hidden_seq matches inference: [h_after_prompt,
     latent_embed_1, ..., latent_embed_T] — NOT last_hidden after each step.
  3. Loss: slot-wise reconstruction via cross-attention from decoded slots to
     input sequence + per-step + pooled auxiliary (not only mean-pooled global).

Optional: --compressor_teacher_path to build prefixes with a trained teacher
(more stable collection). If omitted, uses a randomly initialized compressor
(on-policy with random chain, same as first-time inference).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import ModelWrapper
from methods.slot_mas import SlotAttentionCompressor
from methods import default_agents
from prompts import build_agent_message_sequential_latent_mas
from data import load_gsm8k, load_arc_challenge, load_gpqa_diamond, load_arc_easy


def _forward_one_agent_like_slot_mas(
    model: ModelWrapper,
    embedding_layer: nn.Module,
    ids: torch.Tensor,
    mask: torch.Tensor,
    prev_slots_decoded: Optional[torch.Tensor],
    latent_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    One agent forward matching methods/slot_mas.py::_agent_forward_and_compress
    (without running compressor on output). Returns hidden_seq (1, 1+latent_steps, D).
    """
    ids = ids.to(device)
    mask = mask.to(device)
    prompt_embeds = embedding_layer(ids)

    if prev_slots_decoded is not None:
        prev_slots_decoded = prev_slots_decoded.to(prompt_embeds.device)
        inputs_embeds = torch.cat([prev_slots_decoded, prompt_embeds], dim=1)
        slot_mask = torch.ones(
            prev_slots_decoded.shape[0],
            prev_slots_decoded.shape[1],
            dtype=mask.dtype,
            device=mask.device,
        )
        full_mask = torch.cat([slot_mask, mask], dim=1)
    else:
        inputs_embeds = prompt_embeds
        full_mask = mask

    outputs = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past_kv = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1:, :]

    all_hiddens: List[torch.Tensor] = [last_hidden]
    for _ in range(latent_steps):
        latent_vec = model._apply_latent_realignment(
            last_hidden.squeeze(1), model.model
        )
        latent_embed = latent_vec.unsqueeze(1)
        past_len = past_kv[0][0].shape[-2]
        latent_mask = torch.ones(
            latent_embed.shape[0],
            past_len + 1,
            dtype=torch.long,
            device=device,
        )
        outputs = model.model(
            inputs_embeds=latent_embed,
            attention_mask=latent_mask,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_kv = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1:, :]
        all_hiddens.append(latent_embed)

    return torch.cat(all_hiddens, dim=1)


def collect_hidden_states_slot_aligned(
    model: ModelWrapper,
    task_data: list,
    args: argparse.Namespace,
    teacher: Optional[SlotAttentionCompressor] = None,
    max_samples: int = 500,
) -> torch.Tensor:
    """
    Collect (N * num_non_judger_agents, 1+latent_steps, D) hidden sequences
    under SlotMAS-style prefix chain.
    """
    print(f"[collect] Slot-aligned, max_samples={max_samples}, latent_steps={args.latent_steps}")

    device = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype
    embedding_layer = model.model.get_input_embeddings()
    agents = default_agents()

    if teacher is None:
        d_model = model.model.config.hidden_size
        teacher = SlotAttentionCompressor(
            d_model=d_model,
            num_slots=args.num_slots,
            slot_dim=args.slot_dim if getattr(args, "slot_dim", 0) > 0 else 64,
        ).to(device).to(dtype)
        teacher.eval()
        print("[collect] No teacher checkpoint — using random compressor for prefix chain")
    else:
        teacher = teacher.to(device).to(dtype).eval()

    all_hidden_seqs: List[torch.Tensor] = []

    for i, item in enumerate(task_data):
        if i >= max_samples:
            break
        if i % 50 == 0:
            print(f"  [{i}/{max_samples}]")

        prev_decoded: Optional[torch.Tensor] = None

        for agent in agents:
            if agent.role == "judger":
                continue

            messages = [
                build_agent_message_sequential_latent_mas(
                    role=agent.role,
                    question=item["question"],
                    context="",
                    method="slot_mas",
                    args=args,
                )
            ]
            prompts, _, _, _ = model.prepare_chat_batch(
                messages, add_generation_prompt=True
            )
            if getattr(args, "think", False):
                # Match methods/slot_mas.py (Qwen3 think prompt)
                wrapped = [f"{p}<think>" for p in prompts]
            else:
                wrapped = prompts
            enc = model.tokenizer(
                wrapped,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                hidden_seq = _forward_one_agent_like_slot_mas(
                    model,
                    embedding_layer,
                    ids,
                    mask,
                    prev_decoded,
                    args.latent_steps,
                    device,
                )
                all_hidden_seqs.append(hidden_seq.cpu().to(torch.float16))

                _, prev_decoded = teacher(hidden_seq)
                prev_decoded = prev_decoded.detach()

        torch.cuda.empty_cache()

    all_h = torch.cat(all_hidden_seqs, dim=0)
    print(f"[collect] Stacked {all_h.shape[0]} sequences, shape={tuple(all_h.shape)}")
    return all_h


def _compressor_training_loss(
    compressor: SlotAttentionCompressor,
    batch_h: torch.Tensor,
    num_slots: int,
) -> torch.Tensor:
    """
    batch_h: (B, L, D)
    - cross_attn: each input step attends to decoded slots; MSE to input
    - slot_mse: orthogonal decoder targets each slot to weighted input mix
    - last_step + pooled cosine for stability
    """
    compressed, decoded = compressor(batch_h)
    # decoded: (B, K, D), batch_h: (B, L, D)

    B, L, D = batch_h.shape
    K = decoded.shape[1]

    # --- 1) Cross-attention reconstruction: soft assign each time step to slots ---
    dec = F.normalize(decoded, dim=-1)
    h = F.normalize(batch_h, dim=-1)
    logits = torch.bmm(h, dec.transpose(1, 2)) / (D ** 0.5)  # (B, L, K)
    attn = F.softmax(logits, dim=-1)
    recon_steps = torch.bmm(attn, decoded)  # (B, L, D)
    step_mse = F.mse_loss(recon_steps, batch_h)

    # --- 2) Slot-wise target: barycentric mix of input steps (same attn) ---
    slot_target = torch.bmm(attn.transpose(1, 2), batch_h)  # (B, K, D)
    slot_mse = F.mse_loss(decoded, slot_target)

    # --- 3) Last latent step (often most informative for handoff) ---
    last_mse = F.mse_loss(decoded[:, -1], batch_h[:, -1])

    # --- 4) Pooled direction ---
    inp_pool = batch_h.mean(dim=1)
    dec_pool = decoded.mean(dim=1)
    cos_loss = 1.0 - F.cosine_similarity(dec_pool, inp_pool, dim=-1).mean()

    # --- 5) Diversity on compressed codes ---
    if num_slots > 1 and compressed.shape[-1] > 1:
        zn = F.normalize(compressed, dim=-1)
        sim = torch.bmm(zn, zn.transpose(1, 2))
        eye = torch.eye(num_slots, device=compressed.device, dtype=compressed.dtype).unsqueeze(0)
        div = (sim * (1.0 - eye)).abs().mean()
    else:
        div = torch.tensor(0.0, device=batch_h.device, dtype=batch_h.dtype)

    loss = (
        1.0 * step_mse
        + 1.0 * slot_mse
        + 0.5 * last_mse
        + 0.35 * cos_loss
        + 0.08 * div
    )
    return loss


def train_compressor(
    hidden_data: torch.Tensor,
    d_model: int,
    num_slots: int = 4,
    slot_dim: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cuda:0",
) -> SlotAttentionCompressor:
    print(f"\n[train] slots={num_slots}, dim={slot_dim}, msg={num_slots * slot_dim * 2} B/agent")

    compressor = SlotAttentionCompressor(
        d_model=d_model,
        num_slots=num_slots,
        slot_dim=slot_dim,
    ).to(device).float()

    dataset = TensorDataset(hidden_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(compressor.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        compressor.train()
        total, total_n = 0.0, 0
        for (batch_h,) in loader:
            batch_h = batch_h.float().to(device)
            loss = _compressor_training_loss(compressor, batch_h, num_slots)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
            optimizer.step()

            total += loss.item() * batch_h.size(0)
            total_n += batch_h.size(0)

        scheduler.step()
        avg = total / max(total_n, 1)
        best_loss = min(best_loss, avg)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={avg:.6f} (best={best_loss:.6f})")

    return compressor


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name or local path (e.g. 'Qwen/Qwen3-14B')")
    parser.add_argument(
        "--task",
        type=str,
        default="gsm8k",
        help="gsm8k | arc_challenge | arc_easy | gpqa",
    )
    parser.add_argument("--collect_samples", type=int, default=300)
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--slot_dim", type=int, default=64, help="Teacher slot_dim when no --compressor_teacher_path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/trained_compressor")
    parser.add_argument("--think", action="store_true", default=True)
    parser.add_argument("--prompt", type=str, default="sequential")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument(
        "--compressor_teacher_path",
        type=str,
        default=None,
        help="Optional .pt for prefix chain during collection (same arch as num_slots+slot_dim)",
    )
    parser.add_argument(
        "--train_dims",
        type=str,
        default="64,32,16",
        help="Comma-separated slot_dims to train and save",
    )
    parser.add_argument(
        "--cache_tag",
        type=str,
        default="",
        help="Suffix for cache file to avoid mixing old KV-aligned caches",
    )
    args = parser.parse_args()
    args.method = "slot_mas"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    print(f"Loading {args.model_name}...")
    model = ModelWrapper(args.model_name, device, args=args)
    d_model = model.model.config.hidden_size

    task_data = _load_task(args)
    tag = args.cache_tag or "slot_aligned_v2"
    cache_file = out_dir / f"{args.task}_hiddens_n{args.collect_samples}_{tag}.pt"

    teacher = None
    if args.compressor_teacher_path:
        teacher = SlotAttentionCompressor(
            d_model=d_model,
            num_slots=args.num_slots,
            slot_dim=args.slot_dim,
        ).to(device).to(next(model.model.parameters()).dtype)
        state = torch.load(args.compressor_teacher_path, map_location=device, weights_only=True)
        teacher.load_state_dict(state)
        print(f"[collect] Teacher loaded from {args.compressor_teacher_path}")

    if cache_file.exists():
        print(f"Loading cached hidden states from {cache_file}")
        hidden_data = torch.load(cache_file, weights_only=True)
    else:
        hidden_data = collect_hidden_states_slot_aligned(
            model, task_data, args, teacher=teacher, max_samples=args.collect_samples
        )
        torch.save(hidden_data, cache_file)
        print(f"Saved to {cache_file}")

    del model
    torch.cuda.empty_cache()

    dims = [int(x) for x in args.train_dims.split(",") if x.strip()]
    for sd in dims:
        comp = train_compressor(
            hidden_data,
            d_model,
            num_slots=args.num_slots,
            slot_dim=sd,
            epochs=args.epochs,
            device=str(device),
        )
        save_path = out_dir / f"compressor_s{args.num_slots}_d{sd}.pt"
        torch.save(comp.state_dict(), save_path)
        print(f"Saved {save_path}  ({args.num_slots * sd * 2} B/agent)")

    print("\nDone! Evaluate: run.py --method slot_mas --slot_dim D --compressor_path ...")


if __name__ == "__main__":
    main()
