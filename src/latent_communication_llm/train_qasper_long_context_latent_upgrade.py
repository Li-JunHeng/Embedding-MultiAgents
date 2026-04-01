from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qasper = load_module(
    "qasper_long_upgrade_pilot",
    Path(__file__).parent / "run_qasper_pilot.py",
)


@dataclass
class UpgradeStageConfig:
    name: str
    display_name: str
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
    distill_weight: float = 1.0
    distill_temp: float = 2.0


def build_upgrade_stages() -> list[UpgradeStageConfig]:
    return [
        UpgradeStageConfig(
            name="improved_high_band",
            display_name="Long High Band + QC",
            num_slots=8,
            slot_dim=128,
            num_heads=4,
            epochs=16,
            lr=3e-4,
            weight_decay=1e-4,
            distill_weight=1.0,
            distill_temp=2.0,
        ),
        UpgradeStageConfig(
            name="improved_purified",
            display_name="Long Purified + QC",
            num_slots=8,
            slot_dim=96,
            num_heads=4,
            epochs=18,
            lr=3e-4,
            weight_decay=1e-4,
            variational=True,
            rate_weight=2e-3,
            slot_dropout=0.08,
            noise_std=0.03,
            orth_weight=1e-2,
            distill_weight=1.0,
            distill_temp=2.0,
        ),
    ]


class FullContextTeacher(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int = 1024, num_heads: int = 4) -> None:
        super().__init__()
        self.question_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
        )
        self.doc_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            batch_first=True,
            kdim=hidden_dim,
            vdim=hidden_dim,
        )
        self.gate_query = nn.Linear(hidden_dim, hidden_dim)
        self.gate_key = nn.Linear(hidden_dim, hidden_dim)
        self.doc_refine = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.joint_proj = nn.Sequential(
            nn.LayerNorm(2 * proj_dim),
            nn.Linear(2 * proj_dim, proj_dim),
            nn.GELU(),
        )
        self.choice_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
        )

    def forward(
        self,
        doc_hidden: torch.Tensor,
        doc_mask: torch.Tensor,
        question_hidden: torch.Tensor,
        choice_hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gate_scores = torch.einsum(
            "bd,bld->bl",
            self.gate_query(question_hidden),
            self.gate_key(doc_hidden),
        ) / math.sqrt(doc_hidden.size(-1))
        gate_scores = gate_scores.masked_fill(doc_mask == 0, -1e4)
        gate_weights = torch.softmax(gate_scores, dim=-1)
        refined_doc = doc_hidden + self.doc_refine(doc_hidden) * gate_weights.unsqueeze(-1)

        q = self.question_proj(question_hidden).unsqueeze(1)
        ctx, _ = self.doc_attn(q, refined_doc, refined_doc, key_padding_mask=(doc_mask == 0), need_weights=False)
        joint = self.joint_proj(torch.cat([q.squeeze(1), ctx.squeeze(1)], dim=-1))
        choice_repr = self.choice_proj(choice_hidden)
        logits = torch.einsum("bd,bcd->bc", joint, choice_repr)
        return {"logits": logits}


class QueryConditionedLatentModel(nn.Module):
    def __init__(self, hidden_dim: int, stage_cfg: UpgradeStageConfig) -> None:
        super().__init__()
        self.stage_cfg = stage_cfg
        self.bottleneck = qasper.runtime.DocSlotBottleneck(
            hidden_dim=hidden_dim,
            num_slots=stage_cfg.num_slots,
            slot_dim=stage_cfg.slot_dim,
            num_heads=stage_cfg.num_heads,
            variational=stage_cfg.variational,
        )
        self.slot_query_bias = nn.Linear(hidden_dim, stage_cfg.slot_dim)
        self.token_query = nn.Linear(hidden_dim, hidden_dim)
        self.token_key = nn.Linear(hidden_dim, hidden_dim)
        self.doc_refine = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.receiver_query_proj = nn.Linear(hidden_dim, stage_cfg.slot_dim)
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
        question_hidden: torch.Tensor,
        *,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gate_scores = torch.einsum(
            "bd,bld->bl",
            self.token_query(question_hidden),
            self.token_key(doc_hidden),
        ) / math.sqrt(doc_hidden.size(-1))
        gate_scores = gate_scores.masked_fill(doc_mask == 0, -1e4)
        gate_weights = torch.softmax(gate_scores, dim=-1)
        refined_doc = doc_hidden + self.doc_refine(doc_hidden) * gate_weights.unsqueeze(-1)

        base_queries = self.bottleneck.slot_queries.unsqueeze(0).expand(doc_hidden.size(0), -1, -1)
        queries = base_queries + self.slot_query_bias(question_hidden).unsqueeze(1)
        slots, _ = self.bottleneck.cross_attn(
            queries,
            refined_doc,
            refined_doc,
            key_padding_mask=(doc_mask == 0),
            need_weights=False,
        )
        slots = slots + self.bottleneck.ff(slots)

        mu = None
        logvar = None
        if self.stage_cfg.variational:
            mu = self.bottleneck.to_mu(slots)
            logvar = self.bottleneck.to_logvar(slots).clamp(min=-8.0, max=6.0)
            if training:
                eps = torch.randn_like(mu)
                slots = mu + eps * torch.exp(0.5 * logvar)
            else:
                slots = mu
        else:
            mu = slots
            logvar = torch.zeros_like(slots)

        if training and self.stage_cfg.slot_dropout > 0.0:
            keep = torch.rand(slots.size(0), slots.size(1), 1, device=slots.device) > self.stage_cfg.slot_dropout
            slots = slots * keep
        if training and self.stage_cfg.noise_std > 0.0:
            slots = slots + self.stage_cfg.noise_std * torch.randn_like(slots)

        return slots, {"mu": mu, "logvar": logvar}

    def forward(
        self,
        doc_hidden: torch.Tensor,
        doc_mask: torch.Tensor,
        question_hidden: torch.Tensor,
        choice_hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        slots, stats = self.encode(doc_hidden, doc_mask, question_hidden, training=self.training)
        q = self.receiver_query_proj(question_hidden).unsqueeze(1)
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


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> list[torch.Tensor]:
    return qasper.move_batch(batch, device)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            doc_hidden, doc_mask, question_hidden, choice_hidden, label = move_batch(batch, device)
            outputs = model(doc_hidden, doc_mask, question_hidden, choice_hidden)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            pred = logits.argmax(dim=-1)
            correct += int((pred == label).sum().item())
            total += int(label.numel())
    return correct / total


def evaluate_latent(model: QueryConditionedLatentModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
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
            total_kl += float(qasper.runtime.kl_rate_loss(outputs["mu"], outputs["logvar"], device).detach()) * label.size(0)
    return {"accuracy": correct / total, "avg_kl_rate": total_kl / total}


def train_teacher(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    hidden_dim: int,
    device: torch.device,
) -> tuple[FullContextTeacher, dict[str, float]]:
    model = FullContextTeacher(hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = 0.0
    best_epoch = 0
    for epoch in range(1, 13):
        model.train()
        for batch in train_loader:
            doc_hidden, doc_mask, question_hidden, choice_hidden, label = move_batch(batch, device)
            logits = model(doc_hidden, doc_mask, question_hidden, choice_hidden)["logits"]
            loss = F.cross_entropy(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_acc = evaluate_model(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"[teacher] epoch {epoch:02d} val_acc={val_acc:.4f}")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "test_accuracy": evaluate_model(model, test_loader, device),
    }


def train_latent_stage(
    stage_cfg: UpgradeStageConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    hidden_dim: int,
    teacher: FullContextTeacher,
    device: torch.device,
) -> tuple[QueryConditionedLatentModel, dict[str, float]]:
    model = QueryConditionedLatentModel(hidden_dim, stage_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage_cfg.lr, weight_decay=stage_cfg.weight_decay)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = 0.0
    best_epoch = 0

    for epoch in range(1, stage_cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            doc_hidden, doc_mask, question_hidden, choice_hidden, label = move_batch(batch, device)
            outputs = model(doc_hidden, doc_mask, question_hidden, choice_hidden)
            loss = F.cross_entropy(outputs["logits"], label)
            if stage_cfg.rate_weight > 0.0:
                loss = loss + stage_cfg.rate_weight * qasper.runtime.kl_rate_loss(outputs["mu"], outputs["logvar"], device)
            if stage_cfg.orth_weight > 0.0:
                loss = loss + stage_cfg.orth_weight * qasper.runtime.orthogonality_loss(outputs["slots"])
            with torch.no_grad():
                teacher_logits = teacher(doc_hidden, doc_mask, question_hidden, choice_hidden)["logits"]
            loss = loss + stage_cfg.distill_weight * qasper.runtime.distill_loss(
                outputs["logits"],
                teacher_logits,
                stage_cfg.distill_temp,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_metrics = evaluate_latent(model, val_loader, device)
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"[{stage_cfg.name}] epoch {epoch:02d} val_acc={val_metrics['accuracy']:.4f}")

    model.load_state_dict(best_state)
    test_metrics = evaluate_latent(model, test_loader, device)
    return model, {
        "name": stage_cfg.name,
        "display_name": stage_cfg.display_name,
        "message_floats": stage_cfg.num_slots * stage_cfg.slot_dim,
        "message_bytes": 2 * stage_cfg.num_slots * stage_cfg.slot_dim,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "test_accuracy": test_metrics["accuracy"],
        "test_avg_kl_rate": test_metrics["avg_kl_rate"],
    }


def write_summary(
    output_dir: Path,
    sender_message_stats: dict[str, float],
    text_results: dict | None,
    teacher_metrics: dict[str, float],
    question_only: dict[str, float],
    stage_summaries: list[dict[str, float]],
) -> None:
    lines = ["# QASPER Long-Context Latent Upgrade", ""]
    lines.append("## Accuracy")
    lines.append(f"- Question-only scorer: {question_only['test_accuracy']:.4f}")
    lines.append(f"- Full-context teacher: {teacher_metrics['test_accuracy']:.4f}")
    if text_results is not None:
        lines.append(f"- Long full-text generation: {text_results['full_context_accuracy']:.4f}")
        lines.append(f"- Long query-select generation: {text_results['query_select_accuracy']:.4f}")
    for stage in stage_summaries:
        lines.append(f"- {stage['display_name']}: {stage['test_accuracy']:.4f}")
    lines.append("")
    lines.append("## Main Table")
    lines.append("| Method | Test Acc. | Message | Relative Bytes vs Long Full Text |")
    lines.append("| --- | ---: | --- | ---: |")
    full_bytes = sender_message_stats["avg_full_doc_bytes"]
    if text_results is not None:
        lines.append(f"| Qwen Long Question Only | {question_only['test_accuracy']:.3f} | 0 B | 0.0% |")
        lines.append(
            f"| Qwen Long Full-Text Handoff | {text_results['full_context_accuracy']:.3f} | "
            f"{sender_message_stats['avg_full_doc_tokens']:.1f} tok / {full_bytes:.1f} B | 100.0% |"
        )
        lines.append(
            f"| Qwen Long Query-Select Handoff | {text_results['query_select_accuracy']:.3f} | "
            f"{sender_message_stats['avg_sender_tokens']:.1f} tok / {sender_message_stats['avg_sender_bytes']:.1f} B | "
            f"{100.0 * sender_message_stats['avg_sender_bytes'] / full_bytes:.1f}% |"
        )
    lines.append(f"| Full-Context Teacher | {teacher_metrics['test_accuracy']:.3f} | N/A | N/A |")
    for stage in stage_summaries:
        lines.append(
            f"| {stage['display_name']} | {stage['test_accuracy']:.3f} | "
            f"{stage['message_floats']} fp16 / {stage['message_bytes']} B | "
            f"{100.0 * stage['message_bytes'] / full_bytes:.1f}% |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", type=Path, default=Path(__file__).parent / "results/qasper_long_context_latent_400_100_100")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    qasper.set_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_features = torch.load(args.source_run_dir / "train_features.pt", map_location="cpu", weights_only=False)
    val_features = torch.load(args.source_run_dir / "validation_features.pt", map_location="cpu", weights_only=False)
    test_features = torch.load(args.source_run_dir / "test_features.pt", map_location="cpu", weights_only=False)

    sender_message_stats = json.loads((args.source_run_dir / "sender_message_stats.json").read_text())
    question_only = json.loads((args.source_run_dir / "question_only_metrics.json").read_text())
    text_results = None
    source_meta = json.loads((args.source_run_dir / "run_metadata.json").read_text())
    text_results_path = source_meta.get("text_results_path")
    if text_results_path and Path(text_results_path).exists():
        text_results = json.loads(Path(text_results_path).read_text())

    device = torch.device(args.device)
    train_loader = DataLoader(qasper.build_tensor_dataset(train_features), batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(qasper.build_tensor_dataset(val_features), batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(qasper.build_tensor_dataset(test_features), batch_size=32, shuffle=False, num_workers=0)
    hidden_dim = int(train_features["question_hidden"].size(-1))

    started = time.time()
    teacher, teacher_metrics = train_teacher(train_loader, val_loader, test_loader, hidden_dim, device)
    torch.save(teacher.state_dict(), args.output_dir / "full_context_teacher.pt")
    (args.output_dir / "full_context_teacher_metrics.json").write_text(json.dumps(teacher_metrics, indent=2))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    stage_summaries = []
    for stage_cfg in build_upgrade_stages():
        model, summary = train_latent_stage(stage_cfg, train_loader, val_loader, test_loader, hidden_dim, teacher, device)
        stage_summaries.append(summary)
        torch.save(model.state_dict(), args.output_dir / f"{stage_cfg.name}.pt")
        (args.output_dir / f"{stage_cfg.name}_metrics.json").write_text(json.dumps(summary, indent=2))

    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "source_run_dir": str(args.source_run_dir),
                "seed": args.seed,
                "device": args.device,
                "teacher_metrics": teacher_metrics,
                "question_only_metrics": question_only,
                "stage_summaries": stage_summaries,
                "runtime_seconds": time.time() - started,
            },
            indent=2,
        )
    )
    write_summary(args.output_dir, sender_message_stats, text_results, teacher_metrics, question_only, stage_summaries)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
