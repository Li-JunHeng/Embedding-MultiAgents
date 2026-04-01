from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Stage4Config:
    parent_stage: str = "stage2_purified"
    epochs: int = 18
    lr: float = 1e-4
    weight_decay: float = 1e-4
    receiver_warmstart_epochs: int = 10
    receiver_warmstart_lr: float = 1e-4
    weak_field_boost: float = 2.0
    max_field_weight: float = 3.0
    selected_answer_weight: float = 1.0
    selected_value_weight: float = 0.70
    slot_factor_weight: float = 0.25
    offtarget_adv_weight: float = 0.15
    style_adv_weight: float = 0.15
    distill_weight: float = 0.50
    distill_temp: float = 2.0
    rate_weight: float = 1e-3
    orth_weight: float = 1e-2
    grad_clip: float = 1.0


def build_loader(run_module, features: dict[str, torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        run_module.build_tensor_dataset(features),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def gather_query_slots(slots: torch.Tensor, question_field: torch.Tensor) -> torch.Tensor:
    gather = question_field.view(-1, 1, 1).expand(-1, 1, slots.size(-1))
    return torch.gather(slots, 1, gather)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def evaluate_query_slot_accuracy(run_module, model, loader: DataLoader, device: torch.device) -> tuple[float, list[dict[str, float]]]:
    model.eval()
    total = 0
    correct = 0
    field_correct = [0 for _ in range(model.dataset_cfg.num_fields)]
    field_total = [0 for _ in range(model.dataset_cfg.num_fields)]

    with torch.no_grad():
        for batch in loader:
            batch = run_module.move_batch(batch, device)
            doc_hidden, doc_mask, question_hidden, answer_id, question_field, _, _ = batch
            slots, _ = model.encode(doc_hidden, doc_mask, training=False)
            query_slots = gather_query_slots(slots, question_field)
            logits = model.receiver(question_hidden, query_slots)
            pred = logits.argmax(dim=-1)
            correct_mask = pred == answer_id
            correct += int(correct_mask.sum().item())
            total += int(answer_id.numel())
            for field_idx in range(model.dataset_cfg.num_fields):
                mask = question_field == field_idx
                field_total[field_idx] += int(mask.sum().item())
                field_correct[field_idx] += int(correct_mask[mask].sum().item())

    per_field = []
    for field_idx, (corr, denom) in enumerate(zip(field_correct, field_total, strict=True)):
        per_field.append(
            {
                "field_idx": field_idx,
                "field_name": run_module.FIELD_SPECS[field_idx][0],
                "num_examples": denom,
                "accuracy": (corr / denom) if denom else 0.0,
            }
        )
    return correct / total, per_field


def warmstart_receiver(run_module, model, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, cfg: Stage4Config) -> dict[str, float]:
    original_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("receiver."))

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.receiver_warmstart_lr,
        weight_decay=cfg.weight_decay,
    )

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val_acc, _ = evaluate_query_slot_accuracy(run_module, model, val_loader, device)
    history = []

    for epoch in range(1, cfg.receiver_warmstart_epochs + 1):
        model.train()
        running_loss = 0.0
        running_examples = 0
        for batch in train_loader:
            batch = run_module.move_batch(batch, device)
            doc_hidden, doc_mask, question_hidden, answer_id, question_field, _, _ = batch
            with torch.no_grad():
                slots, _ = model.encode(doc_hidden, doc_mask, training=False)
            query_slots = gather_query_slots(slots, question_field)
            logits = model.receiver(question_hidden, query_slots)
            loss = F.cross_entropy(logits, answer_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * answer_id.size(0)
            running_examples += int(answer_id.numel())

        val_acc, _ = evaluate_query_slot_accuracy(run_module, model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / running_examples,
                "val_query_slot_accuracy": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_requires_grad[name])
    return {"best_val_query_slot_accuracy": best_val_acc, "history": history}


def build_field_weights(per_field_accuracy: list[dict[str, float]], cfg: Stage4Config, device: torch.device) -> torch.Tensor:
    weights = []
    for item in per_field_accuracy:
        weight = 1.0 + cfg.weak_field_boost * (1.0 - item["accuracy"])
        weights.append(min(cfg.max_field_weight, weight))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def weighted_answer_loss(answer_logits: torch.Tensor, answer_id: torch.Tensor, question_field: torch.Tensor, field_weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(answer_logits, answer_id, reduction="none")
    sample_weights = field_weights[question_field]
    return weighted_mean(losses, sample_weights)


def weighted_selected_value_loss(slot_field_logits: torch.Tensor, state_ids: torch.Tensor, question_field: torch.Tensor, field_weights: torch.Tensor) -> torch.Tensor:
    gather_logits = question_field.view(-1, 1, 1).expand(-1, 1, slot_field_logits.size(-1))
    selected_logits = torch.gather(slot_field_logits, 1, gather_logits).squeeze(1)
    selected_targets = torch.gather(state_ids, 1, question_field.view(-1, 1)).squeeze(1)
    losses = F.cross_entropy(selected_logits, selected_targets, reduction="none")
    sample_weights = field_weights[question_field]
    return weighted_mean(losses, sample_weights)


def weighted_slot_factor_loss(slot_field_logits: torch.Tensor, state_ids: torch.Tensor, field_weights: torch.Tensor) -> torch.Tensor:
    batch_size, num_fields, num_values = slot_field_logits.shape
    losses = F.cross_entropy(
        slot_field_logits.reshape(batch_size * num_fields, num_values),
        state_ids.reshape(batch_size * num_fields),
        reduction="none",
    ).view(batch_size, num_fields)
    return weighted_mean(losses, field_weights.view(1, -1).expand_as(losses))


def masked_offtarget_loss(state_logits: torch.Tensor, state_ids: torch.Tensor, question_field: torch.Tensor) -> torch.Tensor:
    batch_size, num_fields, num_values = state_logits.shape
    losses = F.cross_entropy(
        state_logits.reshape(batch_size * num_fields, num_values),
        state_ids.reshape(batch_size * num_fields),
        reduction="none",
    ).view(batch_size, num_fields)
    mask = torch.ones_like(losses, dtype=torch.bool)
    mask.scatter_(1, question_field.view(-1, 1), False)
    return losses[mask].mean()


def weighted_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    question_field: torch.Tensor,
    field_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    per_example = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * (temperature ** 2)
    sample_weights = 1.0 / field_weights[question_field]
    return weighted_mean(per_example, sample_weights)


def evaluate_probe_metrics(audit_module, run_module, model, train_features: dict[str, torch.Tensor], val_features: dict[str, torch.Tensor], test_features: dict[str, torch.Tensor], device: torch.device, *, probe_epochs: int) -> dict[str, object]:
    train_slots = audit_module.collect_slots(run_module, model, train_features, device)
    val_slots = audit_module.collect_slots(run_module, model, val_features, device)
    test_slots = audit_module.collect_slots(run_module, model, test_features, device)
    query_slot_accuracy, per_field_accuracy = audit_module.evaluate_query_slot_mode(
        run_module,
        model,
        test_slots["slots"],
        test_slots["question_hidden"],
        test_slots["answer_id"],
        test_slots["question_field"],
        device,
    )
    val_query_slot_accuracy, val_per_field_accuracy = audit_module.evaluate_query_slot_mode(
        run_module,
        model,
        val_slots["slots"],
        val_slots["question_hidden"],
        val_slots["answer_id"],
        val_slots["question_field"],
        device,
    )
    field_leakage = audit_module.measure_field_leakage(
        train_slots["slots"],
        train_slots["state_ids"],
        test_slots["slots"],
        test_slots["state_ids"],
        device,
        epochs=probe_epochs,
    )
    style_leakage = audit_module.measure_style_leakage(
        train_slots["slots"],
        train_slots["style_id"],
        test_slots["slots"],
        test_slots["style_id"],
        device,
        epochs=probe_epochs,
    )
    audit_module.attach_field_names(run_module, field_leakage, style_leakage)
    return {
        "query_slot_accuracy": query_slot_accuracy,
        "per_field_accuracy": per_field_accuracy,
        "val_query_slot_accuracy": val_query_slot_accuracy,
        "val_per_field_accuracy": val_per_field_accuracy,
        "field_leakage": field_leakage,
        "style_leakage": style_leakage,
    }


def train_stage4(
    run_module,
    model,
    teacher_model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    field_weights: torch.Tensor,
    cfg: Stage4Config,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], dict[str, object]]:
    num_fields = model.dataset_cfg.num_fields
    num_values = len(run_module.FIELD_SPECS[0][1])
    selected_style_adversary = run_module.StyleAdversary(model.dataset_cfg.num_styles, model.stage_cfg.slot_dim).to(device)
    offtarget_head = run_module.StateHead(num_fields, num_values, model.stage_cfg.slot_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(selected_style_adversary.parameters()) + list(offtarget_head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_bundle = {
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "style_adv": {key: value.detach().cpu().clone() for key, value in selected_style_adversary.state_dict().items()},
        "offtarget_head": {key: value.detach().cpu().clone() for key, value in offtarget_head.state_dict().items()},
    }
    best_val_acc, best_val_fields = evaluate_query_slot_accuracy(run_module, model, val_loader, device)
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        selected_style_adversary.train()
        offtarget_head.train()

        running = {
            "selected_answer_loss": 0.0,
            "selected_value_loss": 0.0,
            "slot_factor_loss": 0.0,
            "offtarget_loss": 0.0,
            "style_loss": 0.0,
            "distill_loss": 0.0,
            "rate_loss": 0.0,
            "orth_loss": 0.0,
            "total_loss": 0.0,
            "num_examples": 0,
        }

        for batch in train_loader:
            batch = run_module.move_batch(batch, device)
            doc_hidden, doc_mask, question_hidden, answer_id, question_field, style_id, state_ids = batch

            outputs = model(doc_hidden, doc_mask, question_hidden)
            slots = outputs["slots"]
            query_slots = gather_query_slots(slots, question_field)
            selected_slot = query_slots.squeeze(1)
            answer_logits = model.receiver(question_hidden, query_slots)
            slot_field_logits = model.slot_field_head(slots)

            losses = {}
            losses["selected_answer_loss"] = weighted_answer_loss(answer_logits, answer_id, question_field, field_weights)
            losses["selected_value_loss"] = weighted_selected_value_loss(
                slot_field_logits,
                state_ids,
                question_field,
                field_weights,
            )
            losses["slot_factor_loss"] = weighted_slot_factor_loss(slot_field_logits, state_ids, field_weights)
            reversed_slot = run_module.grad_reverse(selected_slot, 1.0)
            offtarget_logits = offtarget_head(reversed_slot)
            losses["offtarget_loss"] = masked_offtarget_loss(offtarget_logits, state_ids, question_field)
            style_logits = selected_style_adversary(run_module.grad_reverse(selected_slot, 1.0))
            losses["style_loss"] = F.cross_entropy(style_logits, style_id)
            losses["rate_loss"] = run_module.kl_rate_loss(outputs["mu"], outputs["logvar"], device)
            losses["orth_loss"] = run_module.orthogonality_loss(slots)

            with torch.no_grad():
                teacher_outputs = teacher_model(doc_hidden, doc_mask, question_hidden)
                teacher_query_slots = gather_query_slots(teacher_outputs["slots"], question_field)
                teacher_logits = teacher_model.receiver(question_hidden, teacher_query_slots)
            losses["distill_loss"] = weighted_distill_loss(
                answer_logits,
                teacher_logits,
                question_field,
                field_weights,
                cfg.distill_temp,
            )

            total_loss = (
                cfg.selected_answer_weight * losses["selected_answer_loss"]
                + cfg.selected_value_weight * losses["selected_value_loss"]
                + cfg.slot_factor_weight * losses["slot_factor_loss"]
                + cfg.offtarget_adv_weight * losses["offtarget_loss"]
                + cfg.style_adv_weight * losses["style_loss"]
                + cfg.distill_weight * losses["distill_loss"]
                + cfg.rate_weight * losses["rate_loss"]
                + cfg.orth_weight * losses["orth_loss"]
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(selected_style_adversary.parameters()) + list(offtarget_head.parameters()),
                cfg.grad_clip,
            )
            optimizer.step()

            batch_size = int(answer_id.numel())
            for key, value in losses.items():
                running[key] += float(value.detach()) * batch_size
            running["total_loss"] += float(total_loss.detach()) * batch_size
            running["num_examples"] += batch_size

        val_acc, val_fields = evaluate_query_slot_accuracy(run_module, model, val_loader, device)
        epoch_summary = {
            "epoch": epoch,
            "val_query_slot_accuracy": val_acc,
            "train_selected_answer_loss": running["selected_answer_loss"] / running["num_examples"],
            "train_selected_value_loss": running["selected_value_loss"] / running["num_examples"],
            "train_slot_factor_loss": running["slot_factor_loss"] / running["num_examples"],
            "train_offtarget_loss": running["offtarget_loss"] / running["num_examples"],
            "train_style_loss": running["style_loss"] / running["num_examples"],
            "train_distill_loss": running["distill_loss"] / running["num_examples"],
            "train_rate_loss": running["rate_loss"] / running["num_examples"],
            "train_orth_loss": running["orth_loss"] / running["num_examples"],
            "train_total_loss": running["total_loss"] / running["num_examples"],
        }
        history.append(epoch_summary)
        print(
            f"[stage4] epoch {epoch:02d} "
            f"val_acc={val_acc:.4f} "
            f"ans={epoch_summary['train_selected_answer_loss']:.4f} "
            f"off={epoch_summary['train_offtarget_loss']:.4f} "
            f"style={epoch_summary['train_style_loss']:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_fields = val_fields
            best_bundle = {
                "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "style_adv": {key: value.detach().cpu().clone() for key, value in selected_style_adversary.state_dict().items()},
                "offtarget_head": {key: value.detach().cpu().clone() for key, value in offtarget_head.state_dict().items()},
            }

    model.load_state_dict(best_bundle["model"])
    selected_style_adversary.load_state_dict(best_bundle["style_adv"])
    offtarget_head.load_state_dict(best_bundle["offtarget_head"])
    return best_bundle, history, {"best_val_query_slot_accuracy": best_val_acc, "best_val_per_field_accuracy": best_val_fields}


def write_summary(path: Path, payload: dict[str, object]) -> None:
    lines = ["# Stage4 Auditable", ""]
    lines.append("## Main Results")
    lines.append(f"- Parent stage: {payload['config']['parent_stage']}")
    lines.append(f"- Warmstarted receiver val accuracy: {payload['warmstart']['best_val_query_slot_accuracy']:.4f}")
    lines.append(f"- Final val query-slot accuracy: {payload['best_snapshot']['best_val_query_slot_accuracy']:.4f}")
    lines.append(f"- Test query-slot accuracy: {payload['query_slot_accuracy']:.4f}")
    lines.append(f"- Message size: {payload['message_floats']} fp16 / {payload['message_bytes']} B")
    lines.append(f"- Field self-probe mean accuracy: {payload['field_leakage']['diag_mean_accuracy']:.4f}")
    lines.append(f"- Off-target field leak mean accuracy: {payload['field_leakage']['offdiag_mean_accuracy']:.4f}")
    lines.append(f"- Style leak mean accuracy: {payload['style_leakage']['mean_accuracy']:.4f}")
    lines.append("")
    lines.append("## Field Weights")
    for item in payload["field_weights"]:
        lines.append(f"- {item['field_name']}: {item['weight']:.3f} (val acc before stage4 {item['val_accuracy_before_stage4']:.3f})")
    lines.append("")
    lines.append("## Test Per-Field Accuracy")
    for item in payload["per_field_accuracy"]:
        lines.append(f"- {item['field_name']}: {item['accuracy']:.3f} on {item['num_examples']} examples")
    lines.append("")
    lines.append("## Leakage Summary")
    lines.append(f"- Diagonal field probe mean: {payload['field_leakage']['diag_mean_accuracy']:.3f}")
    lines.append(f"- Off-diagonal field probe mean: {payload['field_leakage']['offdiag_mean_accuracy']:.3f}")
    lines.append(f"- Worst off-diagonal field probe: {payload['field_leakage']['offdiag_max_accuracy']:.3f}")
    lines.append(f"- Style probe mean: {payload['style_leakage']['mean_accuracy']:.3f}")
    lines.append(f"- Style probe chance: {payload['style_leakage']['chance']:.3f}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--epoch-scale", type=float, default=1.0)
    parser.add_argument("--output-prefix", type=str, default="stage4_auditable")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--receiver-warmstart-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--receiver-warmstart-lr", type=float, default=None)
    parser.add_argument("--selected-value-weight", type=float, default=None)
    parser.add_argument("--slot-factor-weight", type=float, default=None)
    parser.add_argument("--offtarget-adv-weight", type=float, default=None)
    parser.add_argument("--style-adv-weight", type=float, default=None)
    parser.add_argument("--distill-weight", type=float, default=None)
    parser.add_argument("--weak-field-boost", type=float, default=None)
    parser.add_argument("--max-field-weight", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_module = load_module(
        Path(__file__).parent / "run_qwen_handoff.py",
        "qwen_handoff_stage4",
    )
    audit_module = load_module(
        Path(__file__).parent / "evaluate_auditable_handoff.py",
        "qwen_auditable_eval_stage4",
    )
    run_module.set_seed(11)

    cfg = Stage4Config()
    if args.epoch_scale != 1.0:
        updated = asdict(cfg)
        updated["epochs"] = max(1, int(round(updated["epochs"] * args.epoch_scale)))
        updated["receiver_warmstart_epochs"] = max(1, int(round(updated["receiver_warmstart_epochs"] * args.epoch_scale)))
        cfg = Stage4Config(**updated)
    override_fields = {
        "epochs": args.epochs,
        "receiver_warmstart_epochs": args.receiver_warmstart_epochs,
        "lr": args.lr,
        "receiver_warmstart_lr": args.receiver_warmstart_lr,
        "selected_value_weight": args.selected_value_weight,
        "slot_factor_weight": args.slot_factor_weight,
        "offtarget_adv_weight": args.offtarget_adv_weight,
        "style_adv_weight": args.style_adv_weight,
        "distill_weight": args.distill_weight,
        "weak_field_boost": args.weak_field_boost,
        "max_field_weight": args.max_field_weight,
    }
    if any(value is not None for value in override_fields.values()):
        updated = asdict(cfg)
        for key, value in override_fields.items():
            if value is not None:
                updated[key] = value
        cfg = Stage4Config(**updated)

    metadata = json.loads((args.run_dir / "run_metadata.json").read_text())
    dataset_cfg = run_module.DatasetConfig(**metadata["dataset"])
    parent_stage_cfg_data = next(stage for stage in metadata["stages"] if stage["name"] == cfg.parent_stage)
    parent_stage_cfg = run_module.StageConfig(**parent_stage_cfg_data)

    train_features = torch.load(args.run_dir / "train_features.pt", map_location="cpu")
    val_features = torch.load(args.run_dir / "val_features.pt", map_location="cpu")
    test_features = torch.load(args.run_dir / "test_features.pt", map_location="cpu")
    hidden_dim = int(train_features["question_hidden"].size(-1))
    device = torch.device(args.device)

    model = run_module.LatentHandoffModel(hidden_dim, len(run_module.ANSWER_VOCAB), dataset_cfg, parent_stage_cfg).to(device)
    model.load_state_dict(torch.load(args.run_dir / f"{cfg.parent_stage}.pt", map_location="cpu"))

    train_loader = build_loader(run_module, train_features, args.train_batch_size, shuffle=True)
    val_loader = build_loader(run_module, val_features, args.train_batch_size, shuffle=False)
    test_loader = build_loader(run_module, test_features, args.train_batch_size, shuffle=False)

    print("[stage4] receiver warmstart")
    warmstart = warmstart_receiver(run_module, model, train_loader, val_loader, device, cfg)
    warm_val_acc, warm_val_fields = evaluate_query_slot_accuracy(run_module, model, val_loader, device)
    field_weights_tensor = build_field_weights(warm_val_fields, cfg, device)
    field_weights = [
        {
            "field_idx": field_idx,
            "field_name": item["field_name"],
            "val_accuracy_before_stage4": item["accuracy"],
            "weight": float(field_weights_tensor[field_idx].item()),
        }
        for field_idx, item in enumerate(warm_val_fields)
    ]

    teacher_model = copy.deepcopy(model).to(device)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    print("[stage4] full fine-tune")
    best_bundle, history, best_snapshot = train_stage4(
        run_module,
        model,
        teacher_model,
        train_loader,
        val_loader,
        device,
        field_weights_tensor,
        cfg,
    )

    metrics = evaluate_probe_metrics(
        audit_module,
        run_module,
        model,
        train_features,
        val_features,
        test_features,
        device,
        probe_epochs=args.probe_epochs,
    )
    test_query_slot_acc, _ = evaluate_query_slot_accuracy(run_module, model, test_loader, device)

    payload = {
        "config": asdict(cfg),
        "warmstart": warmstart,
        "best_snapshot": best_snapshot,
        "message_floats": parent_stage_cfg.slot_dim,
        "message_bytes": 2 * parent_stage_cfg.slot_dim + 1,
        "field_weights": field_weights,
        "query_slot_accuracy": test_query_slot_acc,
        "per_field_accuracy": metrics["per_field_accuracy"],
        "val_query_slot_accuracy": metrics["val_query_slot_accuracy"],
        "val_per_field_accuracy": metrics["val_per_field_accuracy"],
        "field_leakage": metrics["field_leakage"],
        "style_leakage": metrics["style_leakage"],
        "training_history": history,
    }

    checkpoint_path = args.run_dir / f"{args.output_prefix}.pt"
    aux_path = args.run_dir / f"{args.output_prefix}_aux.pt"
    config_path = args.run_dir / f"{args.output_prefix}_config.json"
    metrics_path = args.run_dir / f"{args.output_prefix}_metrics.json"
    summary_path = args.run_dir / f"{args.output_prefix}_summary.md"
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(best_bundle, aux_path)
    config_path.write_text(json.dumps(asdict(cfg), indent=2))
    metrics_path.write_text(json.dumps(payload, indent=2))
    write_summary(summary_path, payload)
    print(f"[stage4] saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
