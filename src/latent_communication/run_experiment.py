from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TaskConfig:
    num_factors: int = 8
    num_values: int = 16
    num_styles: int = 8
    train_size: int = 30000
    val_size: int = 4000
    test_size: int = 4000
    batch_size: int = 512
    query_mix_retrieve: float = 0.5
    seed: int = 7


@dataclass
class StageConfig:
    name: str
    num_slots: int
    slot_dim: int
    d_model: int
    num_heads: int
    num_layers: int
    epochs: int
    lr: float
    weight_decay: float
    variational: bool = False
    rate_weight: float = 0.0
    slot_dropout: float = 0.0
    noise_std: float = 0.0
    state_loss_weight: float = 0.0
    slot_factor_weight: float = 0.0
    style_loss_weight: float = 0.0
    style_adv_lambda: float = 0.0
    orth_weight: float = 0.0
    distill_weight: float = 0.0
    distill_temp: float = 2.0


def build_default_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="stage1_high_band",
            num_slots=8,
            slot_dim=64,
            d_model=96,
            num_heads=4,
            num_layers=2,
            epochs=14,
            lr=3e-4,
            weight_decay=1e-4,
            state_loss_weight=0.15,
        ),
        StageConfig(
            name="stage2_purified",
            num_slots=8,
            slot_dim=48,
            d_model=96,
            num_heads=4,
            num_layers=2,
            epochs=16,
            lr=3e-4,
            weight_decay=1e-4,
            variational=True,
            rate_weight=2e-3,
            slot_dropout=0.10,
            noise_std=0.05,
            state_loss_weight=0.20,
            slot_factor_weight=0.35,
            style_loss_weight=0.50,
            style_adv_lambda=1.0,
            orth_weight=1e-2,
        ),
        StageConfig(
            name="stage3_compressed",
            num_slots=8,
            slot_dim=16,
            d_model=96,
            num_heads=4,
            num_layers=2,
            epochs=18,
            lr=3e-4,
            weight_decay=1e-4,
            variational=True,
            rate_weight=2e-3,
            slot_dropout=0.08,
            noise_std=0.04,
            state_loss_weight=0.25,
            style_loss_weight=0.50,
            style_adv_lambda=1.0,
            orth_weight=1e-2,
            distill_weight=1.0,
            distill_temp=2.0,
        ),
    ]


def make_split(cfg: TaskConfig, size: int, seed: int) -> TensorDataset:
    gen = torch.Generator().manual_seed(seed)
    state = torch.randint(cfg.num_values, (size, cfg.num_factors), generator=gen)
    style = torch.randint(cfg.num_styles, (size, cfg.num_factors), generator=gen)

    retrieve_mask = torch.rand(size, generator=gen) < cfg.query_mix_retrieve
    query_type = (~retrieve_mask).long()
    idx_a = torch.randint(cfg.num_factors, (size,), generator=gen)
    idx_b = torch.randint(cfg.num_factors - 1, (size,), generator=gen)
    idx_b = idx_b + (idx_b >= idx_a).long()

    row = torch.arange(size)
    retrieve_answer = state[row, idx_a]
    pair_answer = (state[row, idx_a] + state[row, idx_b]) % cfg.num_values
    answer = torch.where(retrieve_mask, retrieve_answer, pair_answer)
    return TensorDataset(state, style, query_type, idx_a, idx_b, answer)


def build_loaders(cfg: TaskConfig, device: torch.device) -> dict[str, DataLoader]:
    pin_memory = device.type == "cuda"
    return {
        "train": DataLoader(
            make_split(cfg, cfg.train_size, cfg.seed),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            make_split(cfg, cfg.val_size, cfg.seed + 1),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            make_split(cfg, cfg.test_size, cfg.seed + 2),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        ),
    }


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(input_tensor: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(input_tensor, lambd)


class SenderEncoder(nn.Module):
    def __init__(self, task_cfg: TaskConfig, d_model: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.value_emb = nn.Embedding(task_cfg.num_values, d_model)
        self.style_emb = nn.Embedding(task_cfg.num_styles, d_model)
        self.factor_emb = nn.Embedding(task_cfg.num_factors, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        factor_ids = torch.arange(state.size(1), device=state.device)
        x = self.value_emb(state) + self.style_emb(style) + self.factor_emb(factor_ids)[None, :, :]
        x = self.encoder(x)
        return self.norm(x)


class CommunicationBottleneck(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_slots: int,
        slot_dim: int,
        variational: bool,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.variational = variational
        self.slot_queries = nn.Parameter(torch.randn(num_slots, d_model) / math.sqrt(d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.to_slot = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, slot_dim))
        if variational:
            self.to_mu = nn.Linear(slot_dim, slot_dim)
            self.to_logvar = nn.Linear(slot_dim, slot_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        apply_channel: bool,
        slot_dropout: float,
        noise_std: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = hidden.size(0)
        if hidden.size(1) == self.num_slots:
            slots = self.to_slot(hidden)
        else:
            queries = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
            slots, _ = self.cross_attn(queries, hidden, hidden, need_weights=False)
            slots = self.to_slot(slots)

        mu = None
        logvar = None
        if self.variational:
            mu = self.to_mu(slots)
            logvar = self.to_logvar(slots).clamp(min=-8.0, max=6.0)
            if apply_channel:
                eps = torch.randn_like(mu)
                slots = mu + eps * torch.exp(0.5 * logvar)
            else:
                slots = mu

        if apply_channel and slot_dropout > 0.0:
            keep = torch.rand(batch_size, self.num_slots, 1, device=hidden.device) > slot_dropout
            slots = slots * keep

        if apply_channel and noise_std > 0.0:
            slots = slots + noise_std * torch.randn_like(slots)

        return slots, {"mu": mu, "logvar": logvar}


class Receiver(nn.Module):
    def __init__(self, task_cfg: TaskConfig, slot_dim: int, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(2, d_model)
        self.factor_emb = nn.Embedding(task_cfg.num_factors, d_model)
        self.slot_to_model = nn.Linear(slot_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        aligned_dim = 2 * slot_dim + d_model
        self.aligned_readout = nn.Sequential(
            nn.LayerNorm(aligned_dim),
            nn.Linear(aligned_dim, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, task_cfg.num_values),
        )
        self.attn_readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, task_cfg.num_values),
        )

    def forward(
        self,
        query_type: torch.Tensor,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        query = self.type_emb(query_type) + self.factor_emb(idx_a) + query_type[:, None] * self.factor_emb(idx_b)
        if slots.size(1) == self.factor_emb.num_embeddings:
            row = torch.arange(slots.size(0), device=slots.device)
            slot_a = slots[row, idx_a]
            slot_b = slots[row, idx_b] * query_type[:, None]
            return self.aligned_readout(torch.cat([slot_a, slot_b, query], dim=-1))
        kv = self.slot_to_model(slots)
        context, _ = self.cross_attn(query.unsqueeze(1), kv, kv, need_weights=False)
        return self.attn_readout(context.squeeze(1))


class StateHead(nn.Module):
    def __init__(self, task_cfg: TaskConfig, slot_dim: int) -> None:
        super().__init__()
        self.num_factors = task_cfg.num_factors
        self.num_values = task_cfg.num_values
        hidden_dim = 2 * slot_dim * max(2, task_cfg.num_factors // 4)
        self.net = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, task_cfg.num_factors * task_cfg.num_values),
        )

    def forward(self, pooled_slots: torch.Tensor) -> torch.Tensor:
        logits = self.net(pooled_slots)
        return logits.view(pooled_slots.size(0), self.num_factors, self.num_values)


class SlotFactorHead(nn.Module):
    def __init__(self, task_cfg: TaskConfig, slot_dim: int, num_slots: int) -> None:
        super().__init__()
        count = min(task_cfg.num_factors, num_slots)
        self.count = count
        self.heads = nn.ModuleList([nn.Linear(slot_dim, task_cfg.num_values) for _ in range(count)])

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        logits = [head(slots[:, i]) for i, head in enumerate(self.heads)]
        return torch.stack(logits, dim=1)


class StyleAdversary(nn.Module):
    def __init__(self, task_cfg: TaskConfig, slot_dim: int) -> None:
        super().__init__()
        self.num_factors = task_cfg.num_factors
        self.num_styles = task_cfg.num_styles
        hidden_dim = max(64, 4 * slot_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, task_cfg.num_factors * task_cfg.num_styles),
        )

    def forward(self, pooled_slots: torch.Tensor) -> torch.Tensor:
        logits = self.net(pooled_slots)
        return logits.view(pooled_slots.size(0), self.num_factors, self.num_styles)


class CommunicationGame(nn.Module):
    def __init__(self, task_cfg: TaskConfig, stage_cfg: StageConfig) -> None:
        super().__init__()
        self.task_cfg = task_cfg
        self.stage_cfg = stage_cfg
        self.sender = SenderEncoder(task_cfg, stage_cfg.d_model, stage_cfg.num_heads, stage_cfg.num_layers)
        self.bottleneck = CommunicationBottleneck(
            d_model=stage_cfg.d_model,
            num_heads=stage_cfg.num_heads,
            num_slots=stage_cfg.num_slots,
            slot_dim=stage_cfg.slot_dim,
            variational=stage_cfg.variational,
        )
        self.receiver = Receiver(task_cfg, stage_cfg.slot_dim, stage_cfg.d_model, stage_cfg.num_heads)
        self.state_head = StateHead(task_cfg, stage_cfg.slot_dim) if stage_cfg.state_loss_weight > 0.0 else None
        self.slot_factor_head = (
            SlotFactorHead(task_cfg, stage_cfg.slot_dim, stage_cfg.num_slots)
            if stage_cfg.slot_factor_weight > 0.0
            else None
        )
        self.style_adversary = (
            StyleAdversary(task_cfg, stage_cfg.slot_dim) if stage_cfg.style_loss_weight > 0.0 else None
        )

    def encode_message(
        self,
        state: torch.Tensor,
        style: torch.Tensor,
        *,
        training: bool,
        override_slot_dropout: float | None = None,
        override_noise_std: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.sender(state, style)
        slot_dropout = self.stage_cfg.slot_dropout if override_slot_dropout is None else override_slot_dropout
        noise_std = self.stage_cfg.noise_std if override_noise_std is None else override_noise_std
        apply_channel = training or override_slot_dropout is not None or override_noise_std is not None
        return self.bottleneck(
            hidden,
            apply_channel=apply_channel,
            slot_dropout=slot_dropout,
            noise_std=noise_std,
        )

    def forward(
        self,
        state: torch.Tensor,
        style: torch.Tensor,
        query_type: torch.Tensor,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        *,
        override_slot_dropout: float | None = None,
        override_noise_std: float | None = None,
    ) -> dict[str, torch.Tensor]:
        slots, stats = self.encode_message(
            state,
            style,
            training=self.training,
            override_slot_dropout=override_slot_dropout,
            override_noise_std=override_noise_std,
        )
        pooled = slots.mean(dim=1)
        outputs = {
            "slots": slots,
            "pooled": pooled,
            "mu": stats["mu"],
            "logvar": stats["logvar"],
            "answer_logits": self.receiver(query_type, idx_a, idx_b, slots),
        }
        if self.state_head is not None:
            outputs["state_logits"] = self.state_head(pooled)
        if self.slot_factor_head is not None:
            outputs["slot_factor_logits"] = self.slot_factor_head(slots)
        if self.style_adversary is not None:
            reversed_pooled = grad_reverse(pooled, self.stage_cfg.style_adv_lambda)
            outputs["style_logits"] = self.style_adversary(reversed_pooled)
        return outputs


def move_batch(batch: Iterable[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [tensor.to(device, non_blocking=True) for tensor in batch]


def kl_rate_loss(mu: torch.Tensor | None, logvar: torch.Tensor | None) -> torch.Tensor:
    if mu is None or logvar is None:
        return torch.zeros((), device=logvar.device if logvar is not None else mu.device if mu is not None else "cpu")
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def orthogonality_loss(slots: torch.Tensor) -> torch.Tensor:
    normed = F.normalize(slots, dim=-1)
    gram = normed @ normed.transpose(1, 2)
    eye = torch.eye(gram.size(-1), device=gram.device).unsqueeze(0)
    return ((gram - eye) ** 2).mean()


def state_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: list[torch.Tensor],
    stage_cfg: StageConfig,
    *,
    teacher_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    state, style, _, _, _, answer = batch
    answer_loss = F.cross_entropy(outputs["answer_logits"], answer)
    total = answer_loss
    metrics = {"answer_loss": float(answer_loss.detach())}

    rate = kl_rate_loss(outputs["mu"], outputs["logvar"])
    if stage_cfg.rate_weight > 0.0:
        total = total + stage_cfg.rate_weight * rate
    metrics["kl_rate"] = float(rate.detach())

    if "state_logits" in outputs:
        recon = state_ce(outputs["state_logits"], state)
        total = total + stage_cfg.state_loss_weight * recon
        metrics["state_loss"] = float(recon.detach())

    if "slot_factor_logits" in outputs:
        factor_targets = state[:, : outputs["slot_factor_logits"].size(1)]
        factor_loss = state_ce(outputs["slot_factor_logits"], factor_targets)
        total = total + stage_cfg.slot_factor_weight * factor_loss
        metrics["slot_factor_loss"] = float(factor_loss.detach())

    if "style_logits" in outputs:
        style_loss = state_ce(outputs["style_logits"], style)
        total = total + stage_cfg.style_loss_weight * style_loss
        metrics["style_loss"] = float(style_loss.detach())

    if stage_cfg.orth_weight > 0.0:
        orth = orthogonality_loss(outputs["slots"])
        total = total + stage_cfg.orth_weight * orth
        metrics["orth_loss"] = float(orth.detach())

    if teacher_logits is not None and stage_cfg.distill_weight > 0.0:
        kd = distill_loss(outputs["answer_logits"], teacher_logits, stage_cfg.distill_temp)
        total = total + stage_cfg.distill_weight * kd
        metrics["distill_loss"] = float(kd.detach())

    metrics["total_loss"] = float(total.detach())
    return total, metrics


@torch.no_grad()
def evaluate_model(
    model: CommunicationGame,
    loader: DataLoader,
    device: torch.device,
    *,
    override_slot_dropout: float | None = None,
    override_noise_std: float | None = None,
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_kl = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        state, style, query_type, idx_a, idx_b, answer = batch
        outputs = model(
            state,
            style,
            query_type,
            idx_a,
            idx_b,
            override_slot_dropout=override_slot_dropout,
            override_noise_std=override_noise_std,
        )
        pred = outputs["answer_logits"].argmax(dim=-1)
        correct += (pred == answer).sum().item()
        total += answer.numel()
        total_kl += float(kl_rate_loss(outputs["mu"], outputs["logvar"]).detach()) * answer.size(0)
    return {"accuracy": correct / total, "avg_kl_rate": total_kl / total}


def train_stage(
    task_cfg: TaskConfig,
    stage_cfg: StageConfig,
    loaders: dict[str, DataLoader],
    device: torch.device,
    *,
    teacher: CommunicationGame | None = None,
) -> tuple[CommunicationGame, dict[str, float]]:
    model = CommunicationGame(task_cfg, stage_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage_cfg.lr, weight_decay=stage_cfg.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = -1.0
    best_snapshot: dict[str, float] = {}

    for epoch in range(1, stage_cfg.epochs + 1):
        model.train()
        running_correct = 0
        running_total = 0
        running_loss = 0.0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            state, style, query_type, idx_a, idx_b, answer = batch
            teacher_logits = None
            if teacher is not None and stage_cfg.distill_weight > 0.0:
                teacher.eval()
                with torch.no_grad():
                    teacher_outputs = teacher(state, style, query_type, idx_a, idx_b)
                teacher_logits = teacher_outputs["answer_logits"]

            outputs = model(state, style, query_type, idx_a, idx_b)
            loss, _ = compute_loss(outputs, batch, stage_cfg, teacher_logits=teacher_logits)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += float(loss.detach()) * answer.size(0)
            running_correct += (outputs["answer_logits"].argmax(dim=-1) == answer).sum().item()
            running_total += answer.size(0)

        train_acc = running_correct / running_total
        train_loss = running_loss / running_total
        val_metrics = evaluate_model(model, loaders["val"], device)
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_snapshot = {
                "epoch": epoch,
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_avg_kl_rate": val_metrics["avg_kl_rate"],
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_snapshot


@torch.no_grad()
def collect_features(
    model: CommunicationGame,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    features = []
    states = []
    styles = []
    for batch in loader:
        batch = move_batch(batch, device)
        state, style, query_type, idx_a, idx_b, _ = batch
        slots, _ = model.encode_message(state, style, training=False)
        features.append(slots.reshape(slots.size(0), -1).cpu())
        states.append(state.cpu())
        styles.append(style.cpu())
    return torch.cat(features, dim=0), torch.cat(states, dim=0), torch.cat(styles, dim=0)


def train_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    *,
    epochs: int = 20,
    lr: float = 5e-3,
    batch_size: int = 1024,
    max_train_samples: int = 12000,
    max_test_samples: int = 4000,
) -> dict[str, float]:
    if train_x.size(0) > max_train_samples:
        train_x = train_x[:max_train_samples]
        train_y = train_y[:max_train_samples]
    if test_x.size(0) > max_test_samples:
        test_x = test_x[:max_test_samples]
        test_y = test_y[:max_test_samples]

    input_dim = train_x.size(-1)
    num_targets = train_y.size(-1)
    probe = nn.Linear(input_dim, num_targets * num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for _ in range(epochs):
        probe.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = probe(x_batch).view(x_batch.size(0), num_targets, num_classes)
            loss = state_ce(logits, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    probe.eval()
    x_test = test_x.to(device)
    y_test = test_y.to(device)
    logits = probe(x_test).view(x_test.size(0), num_targets, num_classes)
    pred = logits.argmax(dim=-1)
    per_factor_acc = (pred == y_test).float().mean(dim=0).cpu()
    exact_match = (pred == y_test).all(dim=-1).float().mean().item()
    return {
        "mean_factor_accuracy": float(per_factor_acc.mean().item()),
        "min_factor_accuracy": float(per_factor_acc.min().item()),
        "max_factor_accuracy": float(per_factor_acc.max().item()),
        "exact_match": exact_match,
    }


def summarise_stage(
    stage_cfg: StageConfig,
    best_snapshot: dict[str, float],
    clean_metrics: dict[str, float],
    robust_metrics: dict[str, float],
    state_probe: dict[str, float],
    style_probe: dict[str, float],
    *,
    stage1_message_floats: int,
) -> dict[str, float]:
    summary = {
        "message_floats": stage_cfg.num_slots * stage_cfg.slot_dim,
        "compression_vs_stage1": stage1_message_floats / (stage_cfg.num_slots * stage_cfg.slot_dim),
    }
    summary.update({f"best_{k}": v for k, v in best_snapshot.items()})
    summary.update({f"clean_{k}": v for k, v in clean_metrics.items()})
    summary.update({f"robust_{k}": v for k, v in robust_metrics.items()})
    summary.update({f"state_probe_{k}": v for k, v in state_probe.items()})
    summary.update({f"style_probe_{k}": v for k, v in style_probe.items()})
    return summary


def write_summary(
    output_dir: Path,
    task_cfg: TaskConfig,
    stage_cfgs: list[StageConfig],
    summaries: list[dict[str, float]],
) -> None:
    stage1_size = stage_cfgs[0].num_slots * stage_cfgs[0].slot_dim
    lines = ["# Latent Communication Experiment", ""]
    lines.append("## Task")
    lines.append(f"- Factors: {task_cfg.num_factors}")
    lines.append(f"- Semantic values per factor: {task_cfg.num_values}")
    lines.append(f"- Nuisance styles per factor: {task_cfg.num_styles}")
    lines.append("- Query mix: 50% retrieval, 50% pairwise modular sum")
    lines.append("")
    lines.append("## Results")
    for stage_cfg, summary in zip(stage_cfgs, summaries, strict=True):
        compression = stage1_size / summary["message_floats"]
        lines.append(f"### {stage_cfg.name}")
        lines.append(f"- Message size: {summary['message_floats']} floats ({compression:.2f}x smaller than stage 1)")
        lines.append(f"- Clean answer accuracy: {summary['clean_accuracy']:.4f}")
        lines.append(f"- Robust answer accuracy: {summary['robust_accuracy']:.4f}")
        lines.append(f"- Avg KL rate: {summary['clean_avg_kl_rate']:.4f}")
        lines.append(f"- State probe mean factor accuracy: {summary['state_probe_mean_factor_accuracy']:.4f}")
        lines.append(f"- State probe exact match: {summary['state_probe_exact_match']:.4f}")
        lines.append(f"- Style probe mean factor accuracy: {summary['style_probe_mean_factor_accuracy']:.4f}")
        lines.append(f"- Best val accuracy: {summary['best_val_accuracy']:.4f} at epoch {int(summary['best_epoch'])}")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(7)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    task_cfg = TaskConfig()
    stage_cfgs = build_default_stages()
    loaders = build_loaders(task_cfg, device)

    all_summaries = []
    teacher: CommunicationGame | None = None
    started = time.time()
    stage1_message_floats = stage_cfgs[0].num_slots * stage_cfgs[0].slot_dim

    for stage_cfg in stage_cfgs:
        print(f"==> Training {stage_cfg.name}")
        model, best_snapshot = train_stage(task_cfg, stage_cfg, loaders, device, teacher=teacher)
        clean_metrics = evaluate_model(model, loaders["test"], device)
        robust_metrics = evaluate_model(
            model,
            loaders["test"],
            device,
            override_slot_dropout=0.25,
            override_noise_std=0.10,
        )

        train_x, train_state, train_style = collect_features(model, loaders["train"], device)
        test_x, test_state, test_style = collect_features(model, loaders["test"], device)
        state_probe = train_linear_probe(train_x, train_state, test_x, test_state, task_cfg.num_values, device)
        style_probe = train_linear_probe(train_x, train_style, test_x, test_style, task_cfg.num_styles, device)

        summary = summarise_stage(
            stage_cfg,
            best_snapshot,
            clean_metrics,
            robust_metrics,
            state_probe,
            style_probe,
            stage1_message_floats=stage1_message_floats,
        )
        all_summaries.append(summary)

        torch.save(model.state_dict(), output_dir / f"{stage_cfg.name}.pt")
        with (output_dir / f"{stage_cfg.name}_metrics.json").open("w") as f:
            json.dump(summary, f, indent=2)

        teacher = model
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        print(
            f"{stage_cfg.name}: clean_acc={clean_metrics['accuracy']:.4f}, "
            f"robust_acc={robust_metrics['accuracy']:.4f}, "
            f"state_probe={state_probe['mean_factor_accuracy']:.4f}, "
            f"style_probe={style_probe['mean_factor_accuracy']:.4f}"
        )

    write_summary(output_dir, task_cfg, stage_cfgs, all_summaries)
    with (output_dir / "run_metadata.json").open("w") as f:
        json.dump(
            {
                "task": asdict(task_cfg),
                "stages": [asdict(cfg) for cfg in stage_cfgs],
                "results": all_summaries,
                "runtime_seconds": time.time() - started,
                "device": str(device),
            },
            f,
            indent=2,
        )

    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
