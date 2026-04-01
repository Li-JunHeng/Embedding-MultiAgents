from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render_table(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0].keys())
    widths = {key: max(len(key), *(len(str(row[key])) for row in rows)) for key in headers}
    parts = []
    parts.append("| " + " | ".join(f"{key:{widths[key]}}" for key in headers) + " |")
    parts.append("| " + " | ".join("-" * widths[key] for key in headers) + " |")
    for row in rows:
        parts.append("| " + " | ".join(f"{str(row[key]):{widths[key]}}" for key in headers) + " |")
    return "\n".join(parts)


@torch.no_grad()
def collect_slots(run_module, model, features: dict[str, torch.Tensor], device: torch.device):
    loader = DataLoader(run_module.build_tensor_dataset(features), batch_size=128, shuffle=False, num_workers=0)
    slots_all = []
    question_fields = []
    answer_ids = []
    state_ids = []
    style_ids = []
    for batch in loader:
        batch = run_module.move_batch(batch, device)
        doc_hidden, doc_mask, _, answer_id, question_field, style_id, sample_state_ids = batch
        slots, _ = model.encode(doc_hidden, doc_mask, training=False)
        slots_all.append(slots.cpu())
        question_fields.append(question_field.cpu())
        answer_ids.append(answer_id.cpu())
        state_ids.append(sample_state_ids.cpu())
        style_ids.append(style_id.cpu())
    return {
        "slots": torch.cat(slots_all, dim=0),
        "question_field": torch.cat(question_fields, dim=0),
        "answer_id": torch.cat(answer_ids, dim=0),
        "state_ids": torch.cat(state_ids, dim=0),
        "style_id": torch.cat(style_ids, dim=0),
        "question_hidden": features["question_hidden"].cpu(),
    }


@torch.no_grad()
def evaluate_answer_accuracy(run_module, model, slots: torch.Tensor, question_hidden: torch.Tensor, answer_id: torch.Tensor, device: torch.device) -> float:
    total = 0
    correct = 0
    loader = DataLoader(TensorDataset(slots, question_hidden, answer_id), batch_size=128, shuffle=False, num_workers=0)
    for slot_batch, question_batch, answer_batch in loader:
        slot_batch = slot_batch.to(device=device, dtype=torch.float32)
        question_batch = question_batch.to(device=device, dtype=torch.float32)
        answer_batch = answer_batch.to(device=device)
        logits = model.receiver(question_batch, slot_batch)
        pred = logits.argmax(dim=-1)
        correct += int((pred == answer_batch).sum().item())
        total += int(answer_batch.numel())
    return correct / total


@torch.no_grad()
def evaluate_query_slot_mode(run_module, model, slot_tensor: torch.Tensor, question_hidden: torch.Tensor, answer_id: torch.Tensor, question_field: torch.Tensor, device: torch.device) -> tuple[float, list[dict[str, float]]]:
    num_fields = slot_tensor.size(1)
    gather = question_field.view(-1, 1, 1).expand(-1, 1, slot_tensor.size(-1))
    query_slots = torch.gather(slot_tensor, 1, gather)
    overall_accuracy = evaluate_answer_accuracy(run_module, model, query_slots, question_hidden, answer_id, device)

    field_metrics = []
    for field_idx in range(num_fields):
        mask = question_field == field_idx
        if not mask.any():
            continue
        field_acc = evaluate_answer_accuracy(
            run_module,
            model,
            query_slots[mask],
            question_hidden[mask],
            answer_id[mask],
            device,
        )
        field_metrics.append(
            {
                "field_idx": field_idx,
                "field_name": run_module.FIELD_SPECS[field_idx][0],
                "num_examples": int(mask.sum().item()),
                "accuracy": field_acc,
            }
        )
    return overall_accuracy, field_metrics


def finetune_receiver_on_query_slots(
    model,
    train_slots: torch.Tensor,
    train_question_hidden: torch.Tensor,
    train_answer_id: torch.Tensor,
    train_question_field: torch.Tensor,
    val_slots: torch.Tensor,
    val_question_hidden: torch.Tensor,
    val_answer_id: torch.Tensor,
    val_question_field: torch.Tensor,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
) -> tuple[dict[str, torch.Tensor], float]:
    original_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_state = {key: value.clone() for key, value in original_state.items()}

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("receiver."))

    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr, weight_decay=1e-4)

    def build_query_dataset(slots: torch.Tensor, question_hidden: torch.Tensor, answer_id: torch.Tensor, question_field: torch.Tensor) -> TensorDataset:
        gather = question_field.view(-1, 1, 1).expand(-1, 1, slots.size(-1))
        query_slots = torch.gather(slots, 1, gather)
        return TensorDataset(query_slots, question_hidden, answer_id)

    train_loader = DataLoader(
        build_query_dataset(train_slots, train_question_hidden, train_answer_id, train_question_field),
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )
    val_dataset = build_query_dataset(val_slots, val_question_hidden, val_answer_id, val_question_field)

    @torch.no_grad()
    def evaluate_dataset(dataset: TensorDataset) -> float:
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        total = 0
        correct = 0
        model.eval()
        for slot_batch, question_batch, answer_batch in loader:
            slot_batch = slot_batch.to(device=device, dtype=torch.float32)
            question_batch = question_batch.to(device=device, dtype=torch.float32)
            answer_batch = answer_batch.to(device=device)
            pred = model.receiver(question_batch, slot_batch).argmax(dim=-1)
            correct += int((pred == answer_batch).sum().item())
            total += int(answer_batch.numel())
        return correct / total

    best_val = evaluate_dataset(val_dataset)
    for _ in range(epochs):
        model.train()
        for slot_batch, question_batch, answer_batch in train_loader:
            slot_batch = slot_batch.to(device=device, dtype=torch.float32)
            question_batch = question_batch.to(device=device, dtype=torch.float32)
            answer_batch = answer_batch.to(device=device)
            logits = model.receiver(question_batch, slot_batch)
            loss = F.cross_entropy(logits, answer_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_accuracy = evaluate_dataset(val_dataset)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    for _, parameter in model.named_parameters():
        parameter.requires_grad_(True)
    return original_state, best_val


def train_probe(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor, num_classes: int, device: torch.device, *, epochs: int) -> float:
    probe = nn.Linear(train_x.size(-1), num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=5e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True, num_workers=0)

    for _ in range(epochs):
        probe.train()
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device)
            loss = F.cross_entropy(probe(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(test_x.to(device=device, dtype=torch.float32)).argmax(dim=-1).cpu()
    return float((pred == test_y).float().mean().item())


def measure_field_leakage(train_slots: torch.Tensor, train_state: torch.Tensor, test_slots: torch.Tensor, test_state: torch.Tensor, device: torch.device, *, epochs: int) -> dict[str, object]:
    num_slots = train_slots.size(1)
    num_fields = train_state.size(1)
    num_values = len(torch.unique(train_state[:, 0]))
    matrix = []

    for slot_idx in range(num_slots):
        row = []
        slot_train = train_slots[:, slot_idx]
        slot_test = test_slots[:, slot_idx]
        for field_idx in range(num_fields):
            acc = train_probe(
                slot_train,
                train_state[:, field_idx],
                slot_test,
                test_state[:, field_idx],
                num_values,
                device,
                epochs=epochs,
            )
            row.append(acc)
        matrix.append(row)

    diagonal = [matrix[idx][idx] for idx in range(num_slots)]
    off_diagonal = [
        matrix[slot_idx][field_idx]
        for slot_idx in range(num_slots)
        for field_idx in range(num_fields)
        if slot_idx != field_idx
    ]
    return {
        "matrix": matrix,
        "diag_mean_accuracy": sum(diagonal) / len(diagonal),
        "diag_min_accuracy": min(diagonal),
        "diag_max_accuracy": max(diagonal),
        "offdiag_mean_accuracy": sum(off_diagonal) / len(off_diagonal),
        "offdiag_max_accuracy": max(off_diagonal),
        "per_slot": [
            {
                "slot_idx": slot_idx,
                "field_name": None,
                "self_field_accuracy": matrix[slot_idx][slot_idx],
                "off_field_mean_accuracy": sum(
                    matrix[slot_idx][field_idx] for field_idx in range(num_fields) if field_idx != slot_idx
                )
                / (num_fields - 1),
            }
            for slot_idx in range(num_slots)
        ],
    }


def measure_style_leakage(train_slots: torch.Tensor, train_style: torch.Tensor, test_slots: torch.Tensor, test_style: torch.Tensor, device: torch.device, *, epochs: int) -> dict[str, object]:
    num_slots = train_slots.size(1)
    num_styles = len(torch.unique(train_style))
    per_slot = []
    for slot_idx in range(num_slots):
        acc = train_probe(
            train_slots[:, slot_idx],
            train_style,
            test_slots[:, slot_idx],
            test_style,
            num_styles,
            device,
            epochs=epochs,
        )
        per_slot.append({"slot_idx": slot_idx, "accuracy": acc})
    accuracies = [item["accuracy"] for item in per_slot]
    return {
        "chance": 1.0 / num_styles,
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "max_accuracy": max(accuracies),
        "min_accuracy": min(accuracies),
        "per_slot": per_slot,
    }


def attach_field_names(run_module, leakage_metrics: dict[str, object], style_metrics: dict[str, object]) -> None:
    for slot_idx, item in enumerate(leakage_metrics["per_slot"]):
        item["field_name"] = run_module.FIELD_SPECS[slot_idx][0]
    for slot_idx, item in enumerate(style_metrics["per_slot"]):
        item["field_name"] = run_module.FIELD_SPECS[slot_idx][0]


def write_summary(path: Path, summary: dict[str, object]) -> None:
    rows = [
        {
            "Method": "Latent Purified (All Slots)",
            "Acc@200": f"{summary['all_slot_accuracy']:.3f}",
            "Message": f"{summary['all_slot_message_floats']} fp16 / {summary['all_slot_message_bytes']} B",
            "Field Manifest": "No",
            "Field Self-Probe": "-",
            "Off-Field Leak": "-",
        },
        {
            "Method": "Auditable Query Slot",
            "Acc@200": f"{summary['query_slot_accuracy']:.3f}",
            "Message": f"{summary['query_slot_message_floats']} fp16 / {summary['query_slot_total_bytes']} B",
            "Field Manifest": "Yes",
            "Field Self-Probe": f"{summary['field_leakage']['diag_mean_accuracy']:.3f}",
            "Off-Field Leak": f"{summary['field_leakage']['offdiag_mean_accuracy']:.3f}",
        },
        {
            "Method": "Auditable Query Slot + Receiver Tune",
            "Acc@200": f"{summary['query_slot_tuned_accuracy']:.3f}",
            "Message": f"{summary['query_slot_message_floats']} fp16 / {summary['query_slot_total_bytes']} B",
            "Field Manifest": "Yes",
            "Field Self-Probe": f"{summary['field_leakage']['diag_mean_accuracy']:.3f}",
            "Off-Field Leak": f"{summary['field_leakage']['offdiag_mean_accuracy']:.3f}",
        },
    ]
    lines = ["# Auditable Handoff Evaluation", ""]
    lines.append("## Main Table")
    lines.append(render_table(rows))
    lines.append("")
    lines.append("## Per-Field Query-Slot Accuracy")
    for item in summary["per_field_accuracy"]:
        lines.append(
            f"- {item['field_name']}: {item['accuracy']:.3f} on {item['num_examples']} examples"
        )
    lines.append("")
    lines.append("## Per-Field Query-Slot Accuracy After Receiver Tune")
    for item in summary["query_slot_tuned_per_field_accuracy"]:
        lines.append(
            f"- {item['field_name']}: {item['accuracy']:.3f} on {item['num_examples']} examples"
        )
    lines.append("")
    lines.append("## Leakage Summary")
    lines.append(f"- Field self-probe mean accuracy: {summary['field_leakage']['diag_mean_accuracy']:.3f}")
    lines.append(f"- Field off-target probe mean accuracy: {summary['field_leakage']['offdiag_mean_accuracy']:.3f}")
    lines.append(f"- Worst off-target field probe accuracy: {summary['field_leakage']['offdiag_max_accuracy']:.3f}")
    lines.append(f"- Style leakage mean accuracy: {summary['style_leakage']['mean_accuracy']:.3f}")
    lines.append(f"- Style leakage chance accuracy: {summary['style_leakage']['chance']:.3f}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--latent-stage", type=str, default="stage2_purified")
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--receiver-finetune-epochs", type=int, default=10)
    parser.add_argument("--receiver-finetune-lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_module = load_module(
        Path(__file__).parent / "run_qwen_handoff.py",
        "qwen_handoff_audit_runtime",
    )
    run_module.set_seed(11)

    metadata = json.loads((args.run_dir / "run_metadata.json").read_text())
    dataset_cfg = run_module.DatasetConfig(**metadata["dataset"])
    stage_cfg_data = next(stage for stage in metadata["stages"] if stage["name"] == args.latent_stage)
    stage_cfg = run_module.StageConfig(**stage_cfg_data)

    train_features = torch.load(args.run_dir / "train_features.pt", map_location="cpu")
    val_features = torch.load(args.run_dir / "val_features.pt", map_location="cpu")
    test_features = torch.load(args.run_dir / "test_features.pt", map_location="cpu")
    hidden_dim = int(train_features["question_hidden"].size(-1))

    device = torch.device(args.device)
    model = run_module.LatentHandoffModel(hidden_dim, len(run_module.ANSWER_VOCAB), dataset_cfg, stage_cfg).to(device)
    model.load_state_dict(torch.load(args.run_dir / f"{args.latent_stage}.pt", map_location="cpu"))
    model.eval()

    train_slots = collect_slots(run_module, model, train_features, device)
    val_slots = collect_slots(run_module, model, val_features, device)
    test_slots = collect_slots(run_module, model, test_features, device)

    all_slot_accuracy = float(json.loads((args.run_dir / f"{args.latent_stage}_metrics.json").read_text())["clean_accuracy"])
    query_slot_accuracy, per_field_accuracy = evaluate_query_slot_mode(
        run_module,
        model,
        test_slots["slots"],
        test_slots["question_hidden"],
        test_slots["answer_id"],
        test_slots["question_field"],
        device,
    )
    original_state, receiver_tuned_val_accuracy = finetune_receiver_on_query_slots(
        model,
        train_slots["slots"],
        train_slots["question_hidden"],
        train_slots["answer_id"],
        train_slots["question_field"],
        val_slots["slots"],
        val_slots["question_hidden"],
        val_slots["answer_id"],
        val_slots["question_field"],
        device,
        epochs=args.receiver_finetune_epochs,
        lr=args.receiver_finetune_lr,
    )
    query_slot_tuned_accuracy, tuned_per_field_accuracy = evaluate_query_slot_mode(
        run_module,
        model,
        test_slots["slots"],
        test_slots["question_hidden"],
        test_slots["answer_id"],
        test_slots["question_field"],
        device,
    )
    model.load_state_dict(original_state)

    field_leakage = measure_field_leakage(
        train_slots["slots"],
        train_slots["state_ids"],
        test_slots["slots"],
        test_slots["state_ids"],
        device,
        epochs=args.probe_epochs,
    )
    style_leakage = measure_style_leakage(
        train_slots["slots"],
        train_slots["style_id"],
        test_slots["slots"],
        test_slots["style_id"],
        device,
        epochs=args.probe_epochs,
    )
    attach_field_names(run_module, field_leakage, style_leakage)

    query_slot_message_floats = stage_cfg.slot_dim
    manifest_bytes = 1
    summary = {
        "latent_stage": args.latent_stage,
        "all_slot_accuracy": all_slot_accuracy,
        "all_slot_message_floats": stage_cfg.num_slots * stage_cfg.slot_dim,
        "all_slot_message_bytes": 2 * stage_cfg.num_slots * stage_cfg.slot_dim,
        "query_slot_accuracy": query_slot_accuracy,
        "query_slot_message_floats": query_slot_message_floats,
        "query_slot_payload_bytes": 2 * query_slot_message_floats,
        "query_slot_manifest_bytes": manifest_bytes,
        "query_slot_total_bytes": 2 * query_slot_message_floats + manifest_bytes,
        "field_manifest_consistency": 1.0,
        "receiver_tuned_val_accuracy": receiver_tuned_val_accuracy,
        "per_field_accuracy": per_field_accuracy,
        "query_slot_tuned_accuracy": query_slot_tuned_accuracy,
        "query_slot_tuned_per_field_accuracy": tuned_per_field_accuracy,
        "field_leakage": field_leakage,
        "style_leakage": style_leakage,
    }

    out_json = args.run_dir / f"auditable_{args.latent_stage}_evaluation.json"
    out_md = args.run_dir / f"auditable_{args.latent_stage}_evaluation.md"
    out_json.write_text(json.dumps(summary, indent=2))
    write_summary(out_md, summary)
    print(render_table([
        {
            "Method": "Latent Purified (All Slots)",
            "Acc@200": f"{summary['all_slot_accuracy']:.3f}",
            "Message": f"{summary['all_slot_message_floats']} fp16 / {summary['all_slot_message_bytes']} B",
            "Field Manifest": "No",
            "Self-Probe": "-",
            "Off-Field Leak": "-",
        },
        {
            "Method": "Auditable Query Slot",
            "Acc@200": f"{summary['query_slot_accuracy']:.3f}",
            "Message": f"{summary['query_slot_message_floats']} fp16 / {summary['query_slot_total_bytes']} B",
            "Field Manifest": "Yes",
            "Self-Probe": f"{summary['field_leakage']['diag_mean_accuracy']:.3f}",
            "Off-Field Leak": f"{summary['field_leakage']['offdiag_mean_accuracy']:.3f}",
        },
        {
            "Method": "Auditable Query Slot + Receiver Tune",
            "Acc@200": f"{summary['query_slot_tuned_accuracy']:.3f}",
            "Message": f"{summary['query_slot_message_floats']} fp16 / {summary['query_slot_total_bytes']} B",
            "Field Manifest": "Yes",
            "Self-Probe": f"{summary['field_leakage']['diag_mean_accuracy']:.3f}",
            "Off-Field Leak": f"{summary['field_leakage']['offdiag_mean_accuracy']:.3f}",
        },
    ]))
    print(f"\nSaved to {out_json}")


if __name__ == "__main__":
    main()
