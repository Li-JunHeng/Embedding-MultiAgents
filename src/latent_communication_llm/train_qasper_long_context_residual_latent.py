from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import importlib.util
import sys


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qasper = load_module(
    "qasper_long_residual_pilot",
    Path(__file__).parent / "run_qasper_pilot.py",
)


@dataclass
class ResidualStageConfig:
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
    gate_l1_weight: float = 0.01


def build_stages() -> list[ResidualStageConfig]:
    return [
        ResidualStageConfig(
            name="residual_high_band",
            display_name="Long Residual High Band",
            num_slots=8,
            slot_dim=128,
            num_heads=4,
            epochs=16,
            lr=3e-4,
            weight_decay=1e-4,
            gate_l1_weight=0.01,
        ),
        ResidualStageConfig(
            name="residual_purified",
            display_name="Long Residual Purified",
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
            gate_l1_weight=0.01,
        ),
    ]


class ResidualLatentModel(nn.Module):
    def __init__(self, hidden_dim: int, stage_cfg: ResidualStageConfig, base_model: qasper.MCQuestionOnlyModel) -> None:
        super().__init__()
        self.stage_cfg = stage_cfg
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

        self.bottleneck = qasper.runtime.DocSlotBottleneck(
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
        self.gate_proj = nn.Sequential(
            nn.LayerNorm(2 * stage_cfg.slot_dim),
            nn.Linear(2 * stage_cfg.slot_dim, stage_cfg.slot_dim),
            nn.GELU(),
            nn.Linear(stage_cfg.slot_dim, 1),
        )
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))

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

    def forward(
        self,
        doc_hidden: torch.Tensor,
        doc_mask: torch.Tensor,
        question_hidden: torch.Tensor,
        choice_hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            base_logits = self.base_model(question_hidden, choice_hidden)

        slots, stats = self.encode(doc_hidden, doc_mask, training=self.training)
        q = self.query_proj(question_hidden).unsqueeze(1)
        ctx, _ = self.cross_attn(q, slots, slots, need_weights=False)
        joint_input = torch.cat([q.squeeze(1), ctx.squeeze(1)], dim=-1)
        joint = self.joint_proj(joint_input)
        choice_repr = self.choice_proj(choice_hidden)
        latent_logits = torch.einsum("bd,bcd->bc", joint, choice_repr)
        gate = torch.sigmoid(self.gate_proj(joint_input) + self.gate_bias)
        logits = base_logits + gate * latent_logits
        return {
            "slots": slots,
            "mu": stats["mu"],
            "logvar": stats["logvar"],
            "base_logits": base_logits,
            "latent_logits": latent_logits,
            "gate": gate,
            "logits": logits,
        }


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> list[torch.Tensor]:
    return qasper.move_batch(batch, device)


def train_question_only_model(
    train_features: dict[str, torch.Tensor],
    val_features: dict[str, torch.Tensor],
    test_features: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[qasper.MCQuestionOnlyModel, dict[str, float]]:
    model = qasper.MCQuestionOnlyModel(int(train_features["question_hidden"].size(-1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_loader = DataLoader(qasper.build_tensor_dataset(train_features), batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(qasper.build_tensor_dataset(val_features), batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(qasper.build_tensor_dataset(test_features), batch_size=64, shuffle=False, num_workers=0)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = 0.0
    best_epoch = 0
    for epoch in range(1, 13):
        model.train()
        for batch in train_loader:
            _, _, question_hidden, choice_hidden, label = move_batch(batch, device)
            logits = model(question_hidden, choice_hidden)
            loss = F.cross_entropy(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_acc = qasper.evaluate_question_only(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"[question_only_base] epoch {epoch:02d} val_acc={val_acc:.4f}")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "test_accuracy": qasper.evaluate_question_only(model, test_loader, device),
    }


def evaluate_residual(model: ResidualLatentModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_kl = 0.0
    total_gate = 0.0
    with torch.no_grad():
        for batch in loader:
            doc_hidden, doc_mask, question_hidden, choice_hidden, label = move_batch(batch, device)
            outputs = model(doc_hidden, doc_mask, question_hidden, choice_hidden)
            pred = outputs["logits"].argmax(dim=-1)
            correct += int((pred == label).sum().item())
            total += int(label.numel())
            total_gate += float(outputs["gate"].mean().item()) * label.size(0)
            total_kl += float(qasper.runtime.kl_rate_loss(outputs["mu"], outputs["logvar"], device).detach()) * label.size(0)
    return {
        "accuracy": correct / total,
        "avg_kl_rate": total_kl / total,
        "avg_gate": total_gate / total,
    }


def train_residual_stage(
    stage_cfg: ResidualStageConfig,
    base_model: qasper.MCQuestionOnlyModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    hidden_dim: int,
    device: torch.device,
) -> tuple[ResidualLatentModel, dict[str, float]]:
    model = ResidualLatentModel(hidden_dim, stage_cfg, base_model).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=stage_cfg.lr, weight_decay=stage_cfg.weight_decay)
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
            loss = loss + stage_cfg.gate_l1_weight * outputs["gate"].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_metrics = evaluate_residual(model, val_loader, device)
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"[{stage_cfg.name}] epoch {epoch:02d} val_acc={val_metrics['accuracy']:.4f} gate={val_metrics['avg_gate']:.4f}")

    model.load_state_dict(best_state)
    test_metrics = evaluate_residual(model, test_loader, device)
    return model, {
        "name": stage_cfg.name,
        "display_name": stage_cfg.display_name,
        "message_floats": stage_cfg.num_slots * stage_cfg.slot_dim,
        "message_bytes": 2 * stage_cfg.num_slots * stage_cfg.slot_dim,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "test_accuracy": test_metrics["accuracy"],
        "test_avg_kl_rate": test_metrics["avg_kl_rate"],
        "test_avg_gate": test_metrics["avg_gate"],
    }


def write_summary(
    output_dir: Path,
    sender_message_stats: dict[str, float],
    text_results: dict | None,
    base_metrics: dict[str, float],
    stage_summaries: list[dict[str, float]],
) -> None:
    full_bytes = sender_message_stats["avg_full_doc_bytes"]
    lines = ["# QASPER Long-Context Residual Latent", ""]
    lines.append("## Accuracy")
    lines.append(f"- Question-only base: {base_metrics['test_accuracy']:.4f}")
    if text_results is not None:
        lines.append(f"- Long full-text generation: {text_results['full_context_accuracy']:.4f}")
        lines.append(f"- Long query-select generation: {text_results['query_select_accuracy']:.4f}")
    for stage in stage_summaries:
        lines.append(f"- {stage['display_name']}: {stage['test_accuracy']:.4f}")
    lines.append("")
    lines.append("## Main Table")
    lines.append("| Method | Test Acc. | Message | Relative Bytes vs Long Full Text |")
    lines.append("| --- | ---: | --- | ---: |")
    lines.append(f"| Question-Only Base | {base_metrics['test_accuracy']:.3f} | 0 B | 0.0% |")
    if text_results is not None:
        lines.append(
            f"| Qwen Long Full-Text Handoff | {text_results['full_context_accuracy']:.3f} | "
            f"{sender_message_stats['avg_full_doc_tokens']:.1f} tok / {full_bytes:.1f} B | 100.0% |"
        )
        lines.append(
            f"| Qwen Long Query-Select Handoff | {text_results['query_select_accuracy']:.3f} | "
            f"{sender_message_stats['avg_sender_tokens']:.1f} tok / {sender_message_stats['avg_sender_bytes']:.1f} B | "
            f"{100.0 * sender_message_stats['avg_sender_bytes'] / full_bytes:.1f}% |"
        )
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
    source_meta = json.loads((args.source_run_dir / "run_metadata.json").read_text())
    text_results = None
    text_results_path = source_meta.get("text_results_path")
    if text_results_path and Path(text_results_path).exists():
        text_results = json.loads(Path(text_results_path).read_text())

    device = torch.device(args.device)
    train_loader = DataLoader(qasper.build_tensor_dataset(train_features), batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(qasper.build_tensor_dataset(val_features), batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(qasper.build_tensor_dataset(test_features), batch_size=32, shuffle=False, num_workers=0)

    started = time.time()

    base_model, base_metrics = train_question_only_model(train_features, val_features, test_features, device)
    torch.save(base_model.state_dict(), args.output_dir / "question_only_base.pt")
    (args.output_dir / "question_only_base_metrics.json").write_text(json.dumps(base_metrics, indent=2))

    hidden_dim = int(train_features["question_hidden"].size(-1))
    stage_summaries = []
    for stage_cfg in build_stages():
        model, summary = train_residual_stage(stage_cfg, base_model, train_loader, val_loader, test_loader, hidden_dim, device)
        stage_summaries.append(summary)
        torch.save(model.state_dict(), args.output_dir / f"{stage_cfg.name}.pt")
        (args.output_dir / f"{stage_cfg.name}_metrics.json").write_text(json.dumps(summary, indent=2))

    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "source_run_dir": str(args.source_run_dir),
                "seed": args.seed,
                "device": args.device,
                "question_only_base_metrics": base_metrics,
                "stage_summaries": stage_summaries,
                "runtime_seconds": time.time() - started,
            },
            indent=2,
        )
    )
    write_summary(args.output_dir, sender_message_stats, text_results, base_metrics, stage_summaries)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
