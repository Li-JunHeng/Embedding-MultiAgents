from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
class DatasetConfig:
    train_size: int = 1200
    val_size: int = 200
    test_size: int = 200
    num_fields: int = 8
    num_styles: int = 6
    max_doc_length: int = 192
    max_question_length: int = 48
    extractor_batch_size: int = 3
    train_batch_size: int = 128
    seed: int = 11
    generation_eval_examples: int = 40


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
            slot_dim=128,
            num_heads=4,
            epochs=24,
            lr=3e-4,
            weight_decay=1e-4,
            state_loss_weight=0.10,
        ),
        StageConfig(
            name="stage2_purified",
            num_slots=8,
            slot_dim=96,
            num_heads=4,
            epochs=26,
            lr=3e-4,
            weight_decay=1e-4,
            variational=True,
            rate_weight=2e-3,
            slot_dropout=0.08,
            noise_std=0.03,
            state_loss_weight=0.15,
            slot_factor_weight=0.25,
            style_loss_weight=0.50,
            style_adv_lambda=1.0,
            orth_weight=1e-2,
        ),
        StageConfig(
            name="stage3_compressed",
            num_slots=8,
            slot_dim=32,
            num_heads=4,
            epochs=30,
            lr=3e-4,
            weight_decay=1e-4,
            variational=True,
            rate_weight=3e-3,
            slot_dropout=0.10,
            noise_std=0.04,
            state_loss_weight=0.20,
            style_loss_weight=0.50,
            style_adv_lambda=1.0,
            orth_weight=1e-2,
            distill_weight=1.0,
            distill_temp=2.0,
        ),
    ]


FIELD_SPECS = [
    ("city", ["Kyoto", "Lima", "Rabat", "Bergen", "Jaipur", "Porto", "Malmo", "Tucson"]),
    ("pet", ["ferret", "corgi", "iguana", "parrot", "otter", "rabbit", "gecko", "beagle"]),
    ("drink", ["oolong", "mocha", "mate", "lassi", "cider", "horchata", "kombucha", "espresso"]),
    ("hobby", ["pottery", "fencing", "birding", "origami", "sailing", "juggling", "knitting", "calligraphy"]),
    ("commute", ["tram", "subway", "bicycle", "scooter", "ferry", "bus", "train", "skateboard"]),
    ("instrument", ["cello", "banjo", "oboe", "sitar", "violin", "flute", "harp", "clarinet"]),
    ("project", ["amber", "teal", "scarlet", "indigo", "copper", "silver", "crimson", "violet"]),
    ("weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "holiday"]),
]

QUESTION_TEMPLATES = {
    "city": [
        "Which city is {name} based in?",
        "Where does {name} live now?",
        "What city is listed for {name}?",
    ],
    "pet": [
        "What pet does {name} keep?",
        "Which animal stays with {name} at home?",
        "What kind of pet is mentioned for {name}?",
    ],
    "drink": [
        "What drink does {name} usually order?",
        "Which beverage is {name}'s favorite?",
        "What is {name} most likely to sip?",
    ],
    "hobby": [
        "What hobby does {name} practice on weekends?",
        "Which pastime is associated with {name}?",
        "What craft or pastime does {name} enjoy?",
    ],
    "commute": [
        "How does {name} usually commute?",
        "What ride does {name} take most often?",
        "Which mode of transport is linked to {name}?",
    ],
    "instrument": [
        "Which instrument does {name} play?",
        "What instrument is in {name}'s notes?",
        "What musical instrument belongs to {name}'s profile?",
    ],
    "project": [
        "What project color is assigned to {name}?",
        "Which project tag does {name} carry?",
        "What color code appears in {name}'s project line?",
    ],
    "weekday": [
        "Which weekday is tied to {name}'s planning routine?",
        "What day appears in {name}'s schedule note?",
        "Which day is listed for {name}?",
    ],
}

PROFILE_STYLES = [
    {
        "lead": "Profile note for {name}:",
        "connector": "Also",
        "closing": "The note stays brief and matter-of-fact.",
        "order": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    {
        "lead": "Quick memo about {name}:",
        "connector": "By the way",
        "closing": "The wording sounds casual and upbeat.",
        "order": [3, 0, 5, 1, 6, 2, 7, 4],
    },
    {
        "lead": "Team directory entry for {name}:",
        "connector": "Additionally",
        "closing": "The entry reads formal and neatly organized.",
        "order": [4, 6, 0, 2, 7, 3, 1, 5],
    },
    {
        "lead": "{name}'s running profile says this:",
        "connector": "Fun fact",
        "closing": "The wording feels chatty and energetic.",
        "order": [1, 2, 0, 7, 5, 4, 3, 6],
    },
    {
        "lead": "Reference card for {name}:",
        "connector": "In the same card",
        "closing": "The tone is dry, clipped, and office-like.",
        "order": [6, 5, 4, 3, 2, 1, 0, 7],
    },
    {
        "lead": "Diary-style snapshot for {name}:",
        "connector": "Honestly",
        "closing": "The passage feels reflective rather than technical.",
        "order": [2, 7, 3, 0, 1, 6, 5, 4],
    },
]

FIELD_SENTENCES = {
    "city": "{name} is currently based in {value}.",
    "pet": "{name} keeps a {value} at home.",
    "drink": "{name}'s usual drink is {value}.",
    "hobby": "On free evenings, {name} spends time on {value}.",
    "commute": "{name} usually gets around by {value}.",
    "instrument": "{name} practices the {value}.",
    "project": "{name}'s project marker is {value}.",
    "weekday": "{name}'s standing planning day is {value}.",
}

FIRST_NAMES = [
    "Maya", "Arin", "Lena", "Noah", "Iris", "Jonah", "Talia", "Victor", "Zoe", "Milan",
    "Rina", "Damon", "Sara", "Owen", "Nadia", "Felix", "Clara", "Leo", "Mira", "Jules",
]
LAST_NAMES = [
    "Hart", "Vale", "Morris", "Quinn", "Sato", "Rivera", "Patel", "Berg", "Lopez", "Khan",
    "Meyer", "Silva", "Ibrahim", "Frost", "Dawson", "Chen", "Diaz", "Nolan", "Reed", "Park",
]


def make_answer_vocab() -> tuple[list[str], dict[str, int]]:
    values = []
    for _, field_values in FIELD_SPECS:
        values.extend(field_values)
    return values, {value: idx for idx, value in enumerate(values)}


ANSWER_VOCAB, ANSWER_TO_ID = make_answer_vocab()


def pick_name(sample_idx: int) -> str:
    first = FIRST_NAMES[sample_idx % len(FIRST_NAMES)]
    last = LAST_NAMES[(sample_idx // len(FIRST_NAMES)) % len(LAST_NAMES)]
    return f"{first} {last}"


def render_profile(name: str, value_ids: list[int], style_id: int) -> str:
    style = PROFILE_STYLES[style_id]
    ordered_sentences = []
    for field_idx in style["order"]:
        field_name, field_values = FIELD_SPECS[field_idx]
        value = field_values[value_ids[field_idx]]
        ordered_sentences.append(FIELD_SENTENCES[field_name].format(name=name, value=value))
    parts = [style["lead"].format(name=name), ordered_sentences[0]]
    for sentence in ordered_sentences[1:]:
        parts.append(f"{style['connector']}, {sentence[0].lower()}{sentence[1:]}")
    parts.append(style["closing"])
    return " ".join(parts)


def render_question(name: str, field_idx: int, phrasing_id: int) -> str:
    field_name, _ = FIELD_SPECS[field_idx]
    templates = QUESTION_TEMPLATES[field_name]
    return templates[phrasing_id % len(templates)].format(name=name)


def build_split(size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for idx in range(size):
        sample_idx = seed * 10000 + idx
        name = pick_name(sample_idx)
        value_ids = [rng.randrange(len(values)) for _, values in FIELD_SPECS]
        style_id = rng.randrange(len(PROFILE_STYLES))
        question_field = rng.randrange(len(FIELD_SPECS))
        phrasing_id = rng.randrange(3)
        answer = FIELD_SPECS[question_field][1][value_ids[question_field]]
        samples.append(
            {
                "sample_id": sample_idx,
                "name": name,
                "state_ids": value_ids,
                "style_id": style_id,
                "question_field": question_field,
                "answer_text": answer,
                "answer_id": ANSWER_TO_ID[answer],
                "profile_text": render_profile(name, value_ids, style_id),
                "question_text": render_question(name, question_field, phrasing_id),
            }
        )
    return samples


def build_dataset(cfg: DatasetConfig) -> dict[str, list[dict]]:
    return {
        "train": build_split(cfg.train_size, cfg.seed),
        "val": build_split(cfg.val_size, cfg.seed + 1),
        "test": build_split(cfg.test_size, cfg.seed + 2),
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


def load_qwen_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=-1) - 1
    row = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[row, lengths]


@torch.no_grad()
def extract_hidden_features(
    samples: list[dict],
    cfg: DatasetConfig,
    tokenizer,
    model,
    split_name: str,
) -> dict[str, torch.Tensor]:
    doc_hidden_chunks = []
    doc_mask_chunks = []
    question_hidden_chunks = []
    answer_ids = []
    field_ids = []
    style_ids = []
    state_ids = []

    total = len(samples)
    for start in range(0, total, cfg.extractor_batch_size):
        batch = samples[start : start + cfg.extractor_batch_size]
        doc_texts = [f"Read the profile carefully.\nProfile:\n{item['profile_text']}" for item in batch]
        question_texts = [f"Question:\n{item['question_text']}\nAnswer with one short phrase." for item in batch]

        doc_inputs = tokenizer(
            doc_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.max_doc_length,
        ).to(model.device)
        doc_outputs = model(**doc_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        doc_hidden_chunks.append(doc_outputs.hidden_states[-1].to(dtype=torch.float16).cpu())
        doc_mask_chunks.append(doc_inputs["attention_mask"].cpu())

        question_inputs = tokenizer(
            question_texts,
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
        question_hidden = last_token_pool(question_outputs.hidden_states[-1], question_inputs["attention_mask"])
        question_hidden_chunks.append(question_hidden.to(dtype=torch.float16).cpu())

        answer_ids.extend(item["answer_id"] for item in batch)
        field_ids.extend(item["question_field"] for item in batch)
        style_ids.extend(item["style_id"] for item in batch)
        state_ids.extend(item["state_ids"] for item in batch)

        if (start // cfg.extractor_batch_size) % 25 == 0:
            print(f"[{split_name}] extracted {min(start + len(batch), total)}/{total}")

    doc_hidden = torch.cat(doc_hidden_chunks, dim=0)
    doc_mask = torch.cat(doc_mask_chunks, dim=0)
    question_hidden = torch.cat(question_hidden_chunks, dim=0)
    return {
        "doc_hidden": doc_hidden,
        "doc_mask": doc_mask,
        "question_hidden": question_hidden,
        "answer_id": torch.tensor(answer_ids, dtype=torch.long),
        "question_field": torch.tensor(field_ids, dtype=torch.long),
        "style_id": torch.tensor(style_ids, dtype=torch.long),
        "state_ids": torch.tensor(state_ids, dtype=torch.long),
    }


def normalize_answer(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z]+", " ", lowered)
    return " ".join(lowered.split())


@torch.no_grad()
def run_generation_baseline(
    samples: list[dict],
    cfg: DatasetConfig,
    tokenizer,
    model,
) -> dict[str, float]:
    chosen = samples[: cfg.generation_eval_examples]
    full_correct = 0
    question_only_correct = 0

    normalized_values = {normalize_answer(value): value for value in ANSWER_VOCAB}

    for idx, item in enumerate(chosen):
        full_prompt = (
            "Answer with exactly the short phrase from the profile.\n"
            f"Profile:\n{item['profile_text']}\n"
            f"Question: {item['question_text']}\nAnswer:"
        )
        question_only_prompt = (
            "Answer with exactly one short phrase.\n"
            f"Question: {item['question_text']}\nAnswer:"
        )

        for prompt_name, prompt in [("full", full_prompt), ("question_only", question_only_prompt)]:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
            )
            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            norm = normalize_answer(text)
            matched = None
            for candidate_norm, original in normalized_values.items():
                if candidate_norm in norm:
                    matched = original
                    break
            if prompt_name == "full":
                full_correct += int(matched == item["answer_text"])
            else:
                question_only_correct += int(matched == item["answer_text"])
        if idx % 10 == 0:
            print(f"[generation] processed {idx + 1}/{len(chosen)}")

    total = len(chosen)
    return {
        "num_examples": total,
        "full_context_accuracy": full_correct / total,
        "question_only_accuracy": question_only_correct / total,
    }


class DocSlotBottleneck(nn.Module):
    def __init__(self, hidden_dim: int, num_slots: int, slot_dim: int, num_heads: int, variational: bool) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.variational = variational
        self.slot_queries = nn.Parameter(torch.randn(num_slots, slot_dim) / math.sqrt(slot_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=slot_dim,
            num_heads=num_heads,
            batch_first=True,
            kdim=hidden_dim,
            vdim=hidden_dim,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, 2 * slot_dim),
            nn.GELU(),
            nn.Linear(2 * slot_dim, slot_dim),
        )
        if variational:
            self.to_mu = nn.Linear(slot_dim, slot_dim)
            self.to_logvar = nn.Linear(slot_dim, slot_dim)

    def forward(
        self,
        doc_hidden: torch.Tensor,
        doc_mask: torch.Tensor,
        *,
        apply_channel: bool,
        slot_dropout: float,
        noise_std: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = doc_hidden.size(0)
        queries = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
        slots, _ = self.cross_attn(
            queries,
            doc_hidden,
            doc_hidden,
            key_padding_mask=(doc_mask == 0),
            need_weights=False,
        )
        slots = slots + self.ff(slots)

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
            keep = torch.rand(batch_size, self.num_slots, 1, device=doc_hidden.device) > slot_dropout
            slots = slots * keep

        if apply_channel and noise_std > 0.0:
            slots = slots + noise_std * torch.randn_like(slots)

        return slots, {"mu": mu, "logvar": logvar}


class QueryReceiver(nn.Module):
    def __init__(self, hidden_dim: int, slot_dim: int, num_heads: int, num_answers: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, slot_dim)
        self.cross_attn = nn.MultiheadAttention(slot_dim, num_heads, batch_first=True)
        self.readout = nn.Sequential(
            nn.LayerNorm(2 * slot_dim),
            nn.Linear(2 * slot_dim, 4 * slot_dim),
            nn.GELU(),
            nn.Linear(4 * slot_dim, num_answers),
        )

    def forward(self, question_hidden: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(question_hidden).unsqueeze(1)
        context, _ = self.cross_attn(query, slots, slots, need_weights=False)
        joint = torch.cat([query.squeeze(1), context.squeeze(1)], dim=-1)
        return self.readout(joint)


class StateHead(nn.Module):
    def __init__(self, num_fields: int, num_values_per_field: int, slot_dim: int) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.num_values_per_field = num_values_per_field
        self.net = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, 4 * slot_dim),
            nn.GELU(),
            nn.Linear(4 * slot_dim, num_fields * num_values_per_field),
        )

    def forward(self, pooled_slots: torch.Tensor) -> torch.Tensor:
        logits = self.net(pooled_slots)
        return logits.view(pooled_slots.size(0), self.num_fields, self.num_values_per_field)


class SlotFieldHead(nn.Module):
    def __init__(self, num_fields: int, num_values_per_field: int, slot_dim: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(slot_dim, num_values_per_field) for _ in range(num_fields)])

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        logits = [head(slots[:, idx]) for idx, head in enumerate(self.heads)]
        return torch.stack(logits, dim=1)


class StyleAdversary(nn.Module):
    def __init__(self, num_styles: int, slot_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, 4 * slot_dim),
            nn.GELU(),
            nn.Linear(4 * slot_dim, num_styles),
        )

    def forward(self, pooled_slots: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_slots)


class LatentHandoffModel(nn.Module):
    def __init__(self, hidden_dim: int, num_answers: int, dataset_cfg: DatasetConfig, stage_cfg: StageConfig) -> None:
        super().__init__()
        self.stage_cfg = stage_cfg
        self.dataset_cfg = dataset_cfg
        self.bottleneck = DocSlotBottleneck(
            hidden_dim=hidden_dim,
            num_slots=stage_cfg.num_slots,
            slot_dim=stage_cfg.slot_dim,
            num_heads=stage_cfg.num_heads,
            variational=stage_cfg.variational,
        )
        self.receiver = QueryReceiver(hidden_dim, stage_cfg.slot_dim, stage_cfg.num_heads, num_answers)
        self.state_head = StateHead(dataset_cfg.num_fields, len(FIELD_SPECS[0][1]), stage_cfg.slot_dim)
        self.slot_field_head = (
            SlotFieldHead(dataset_cfg.num_fields, len(FIELD_SPECS[0][1]), stage_cfg.slot_dim)
            if stage_cfg.slot_factor_weight > 0.0
            else None
        )
        self.style_adversary = (
            StyleAdversary(dataset_cfg.num_styles, stage_cfg.slot_dim)
            if stage_cfg.style_loss_weight > 0.0
            else None
        )

    def encode(
        self,
        doc_hidden: torch.Tensor,
        doc_mask: torch.Tensor,
        *,
        training: bool,
        override_slot_dropout: float | None = None,
        override_noise_std: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        slot_dropout = self.stage_cfg.slot_dropout if override_slot_dropout is None else override_slot_dropout
        noise_std = self.stage_cfg.noise_std if override_noise_std is None else override_noise_std
        apply_channel = training or override_slot_dropout is not None or override_noise_std is not None
        return self.bottleneck(
            doc_hidden,
            doc_mask,
            apply_channel=apply_channel,
            slot_dropout=slot_dropout,
            noise_std=noise_std,
        )

    def forward(
        self,
        doc_hidden: torch.Tensor,
        doc_mask: torch.Tensor,
        question_hidden: torch.Tensor,
        *,
        override_slot_dropout: float | None = None,
        override_noise_std: float | None = None,
    ) -> dict[str, torch.Tensor]:
        slots, stats = self.encode(
            doc_hidden,
            doc_mask,
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
            "answer_logits": self.receiver(question_hidden, slots),
            "state_logits": self.state_head(pooled),
        }
        if self.slot_field_head is not None:
            outputs["slot_field_logits"] = self.slot_field_head(slots)
        if self.style_adversary is not None:
            outputs["style_logits"] = self.style_adversary(grad_reverse(pooled, self.stage_cfg.style_adv_lambda))
        return outputs


def build_tensor_dataset(features: dict[str, torch.Tensor]) -> TensorDataset:
    return TensorDataset(
        features["doc_hidden"],
        features["doc_mask"],
        features["question_hidden"],
        features["answer_id"],
        features["question_field"],
        features["style_id"],
        features["state_ids"],
    )


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> list[torch.Tensor]:
    out = []
    for idx, tensor in enumerate(batch):
        dtype = torch.float32 if idx in {0, 2} else None
        if tensor.is_floating_point():
            out.append(tensor.to(device=device, dtype=torch.float32, non_blocking=True))
        else:
            out.append(tensor.to(device=device, non_blocking=True))
    return out


def kl_rate_loss(mu: torch.Tensor | None, logvar: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if mu is None or logvar is None:
        return torch.zeros((), device=device)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def orthogonality_loss(slots: torch.Tensor) -> torch.Tensor:
    normed = F.normalize(slots, dim=-1)
    gram = normed @ normed.transpose(1, 2)
    eye = torch.eye(gram.size(-1), device=gram.device).unsqueeze(0)
    return ((gram - eye) ** 2).mean()


def state_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: list[torch.Tensor],
    stage_cfg: StageConfig,
    device: torch.device,
    *,
    teacher_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    _, _, _, answer_id, _, style_id, state_ids = batch
    answer_loss = F.cross_entropy(outputs["answer_logits"], answer_id)
    total = answer_loss
    metrics = {"answer_loss": float(answer_loss.detach())}

    rate = kl_rate_loss(outputs["mu"], outputs["logvar"], device)
    if stage_cfg.rate_weight > 0.0:
        total = total + stage_cfg.rate_weight * rate
    metrics["kl_rate"] = float(rate.detach())

    state_loss = state_cross_entropy(outputs["state_logits"], state_ids)
    total = total + stage_cfg.state_loss_weight * state_loss
    metrics["state_loss"] = float(state_loss.detach())

    if "slot_field_logits" in outputs:
        slot_loss = state_cross_entropy(outputs["slot_field_logits"], state_ids)
        total = total + stage_cfg.slot_factor_weight * slot_loss
        metrics["slot_field_loss"] = float(slot_loss.detach())

    if "style_logits" in outputs:
        style_loss = F.cross_entropy(outputs["style_logits"], style_id)
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
    model: LatentHandoffModel,
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
        doc_hidden, doc_mask, question_hidden, answer_id, _, _, _ = batch
        outputs = model(
            doc_hidden,
            doc_mask,
            question_hidden,
            override_slot_dropout=override_slot_dropout,
            override_noise_std=override_noise_std,
        )
        pred = outputs["answer_logits"].argmax(dim=-1)
        correct += (pred == answer_id).sum().item()
        total += answer_id.numel()
        total_kl += float(kl_rate_loss(outputs["mu"], outputs["logvar"], device).detach()) * answer_id.size(0)
    return {"accuracy": correct / total, "avg_kl_rate": total_kl / total}


@torch.no_grad()
def collect_latent_features(
    model: LatentHandoffModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    features = []
    states = []
    styles = []
    for batch in loader:
        batch = move_batch(batch, device)
        doc_hidden, doc_mask, _, _, _, style_id, state_ids = batch
        slots, _ = model.encode(doc_hidden, doc_mask, training=False)
        features.append(slots.reshape(slots.size(0), -1).cpu())
        states.append(state_ids.cpu())
        styles.append(style_id.cpu())
    return torch.cat(features, dim=0), torch.cat(states, dim=0), torch.cat(styles, dim=0)


def train_state_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    *,
    max_train_samples: int = 1000,
    max_test_samples: int = 200,
    epochs: int = 30,
) -> dict[str, float]:
    train_x = train_x[:max_train_samples]
    train_y = train_y[:max_train_samples]
    test_x = test_x[:max_test_samples]
    test_y = test_y[:max_test_samples]

    num_fields = train_y.size(-1)
    num_values = len(FIELD_SPECS[0][1])
    probe = nn.Linear(train_x.size(-1), num_fields * num_values).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=5e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True, num_workers=0)

    for _ in range(epochs):
        probe.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = probe(xb).view(xb.size(0), num_fields, num_values)
            loss = state_cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    probe.eval()
    xb = test_x.to(device)
    yb = test_y.to(device)
    logits = probe(xb).view(xb.size(0), num_fields, num_values)
    pred = logits.argmax(dim=-1)
    per_field = (pred == yb).float().mean(dim=0).cpu()
    return {
        "mean_field_accuracy": float(per_field.mean().item()),
        "min_field_accuracy": float(per_field.min().item()),
        "max_field_accuracy": float(per_field.max().item()),
        "exact_match": float((pred == yb).all(dim=-1).float().mean().item()),
    }


def train_style_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    *,
    max_train_samples: int = 1000,
    max_test_samples: int = 200,
    epochs: int = 30,
) -> dict[str, float]:
    train_x = train_x[:max_train_samples]
    train_y = train_y[:max_train_samples]
    test_x = test_x[:max_test_samples]
    test_y = test_y[:max_test_samples]

    probe = nn.Linear(train_x.size(-1), len(PROFILE_STYLES)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=5e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True, num_workers=0)

    for _ in range(epochs):
        probe.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    probe.eval()
    xb = test_x.to(device)
    yb = test_y.to(device)
    pred = probe(xb).argmax(dim=-1)
    return {
        "accuracy": float((pred == yb).float().mean().item()),
        "chance": 1.0 / len(PROFILE_STYLES),
    }


def train_query_only_baseline(
    train_features: dict[str, torch.Tensor],
    test_features: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    model = nn.Sequential(
        nn.LayerNorm(train_features["question_hidden"].size(-1)),
        nn.Linear(train_features["question_hidden"].size(-1), 512),
        nn.GELU(),
        nn.Linear(512, len(ANSWER_VOCAB)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_loader = DataLoader(
        TensorDataset(train_features["question_hidden"], train_features["answer_id"]),
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )
    for _ in range(25):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    xb = test_features["question_hidden"].to(device=device, dtype=torch.float32)
    yb = test_features["answer_id"].to(device)
    pred = model(xb).argmax(dim=-1)
    return float((pred == yb).float().mean().item())


def train_stage(
    hidden_dim: int,
    dataset_cfg: DatasetConfig,
    stage_cfg: StageConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    teacher: LatentHandoffModel | None = None,
) -> tuple[LatentHandoffModel, dict[str, float]]:
    model = LatentHandoffModel(hidden_dim, len(ANSWER_VOCAB), dataset_cfg, stage_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage_cfg.lr, weight_decay=stage_cfg.weight_decay)

    best_state = None
    best_val = -1.0
    best_snapshot: dict[str, float] = {}

    for epoch in range(1, stage_cfg.epochs + 1):
        model.train()
        running_correct = 0
        running_total = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            doc_hidden, doc_mask, question_hidden, answer_id, _, _, _ = batch

            teacher_logits = None
            if teacher is not None and stage_cfg.distill_weight > 0.0:
                teacher.eval()
                with torch.no_grad():
                    teacher_outputs = teacher(doc_hidden, doc_mask, question_hidden)
                teacher_logits = teacher_outputs["answer_logits"]

            outputs = model(doc_hidden, doc_mask, question_hidden)
            loss, _ = compute_loss(outputs, batch, stage_cfg, device, teacher_logits=teacher_logits)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_correct += (outputs["answer_logits"].argmax(dim=-1) == answer_id).sum().item()
            running_total += answer_id.size(0)

        train_acc = running_correct / running_total
        val_metrics = evaluate_model(model, val_loader, device)
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_snapshot = {
                "epoch": epoch,
                "train_accuracy": train_acc,
                "val_accuracy": val_metrics["accuracy"],
                "val_avg_kl_rate": val_metrics["avg_kl_rate"],
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_snapshot


def summarize_stage(
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
    dataset_cfg: DatasetConfig,
    stage_cfgs: list[StageConfig],
    summaries: list[dict[str, float]],
    generation_baseline: dict[str, float],
    query_only_accuracy: float,
) -> None:
    stage1_size = stage_cfgs[0].num_slots * stage_cfgs[0].slot_dim
    lines = ["# Qwen3 Latent Handoff", ""]
    lines.append("## Task")
    lines.append(f"- Train/val/test sizes: {dataset_cfg.train_size}/{dataset_cfg.val_size}/{dataset_cfg.test_size}")
    lines.append(f"- Fields per profile: {dataset_cfg.num_fields}")
    lines.append(f"- Style templates: {dataset_cfg.num_styles}")
    lines.append("- Sender reads the full profile paragraph; receiver reads only the question.")
    lines.append("")
    lines.append("## Baselines")
    lines.append(f"- Query-only classifier accuracy: {query_only_accuracy:.4f}")
    lines.append(
        f"- Qwen direct generation full-context accuracy on {generation_baseline['num_examples']} examples: "
        f"{generation_baseline['full_context_accuracy']:.4f}"
    )
    lines.append(
        f"- Qwen direct generation question-only accuracy on {generation_baseline['num_examples']} examples: "
        f"{generation_baseline['question_only_accuracy']:.4f}"
    )
    lines.append("")
    lines.append("## Results")
    for stage_cfg, summary in zip(stage_cfgs, summaries, strict=True):
        compression = stage1_size / summary["message_floats"]
        lines.append(f"### {stage_cfg.name}")
        lines.append(f"- Message size: {summary['message_floats']} floats ({compression:.2f}x smaller than stage 1)")
        lines.append(f"- Clean answer accuracy: {summary['clean_accuracy']:.4f}")
        lines.append(f"- Robust answer accuracy: {summary['robust_accuracy']:.4f}")
        lines.append(f"- Avg KL rate: {summary['clean_avg_kl_rate']:.4f}")
        lines.append(f"- State probe mean field accuracy: {summary['state_probe_mean_field_accuracy']:.4f}")
        lines.append(f"- State probe exact match: {summary['state_probe_exact_match']:.4f}")
        lines.append(f"- Style probe accuracy: {summary['style_probe_accuracy']:.4f}")
        lines.append(f"- Best val accuracy: {summary['best_val_accuracy']:.4f} at epoch {int(summary['best_epoch'])}")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--extractor-batch-size", type=int, default=3)
    parser.add_argument("--generation-eval-examples", type=int, default=40)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--epoch-scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(11)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cfg = DatasetConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        extractor_batch_size=args.extractor_batch_size,
        train_batch_size=args.train_batch_size,
        generation_eval_examples=args.generation_eval_examples,
    )
    stage_cfgs = build_default_stages()
    if args.epoch_scale != 1.0:
        scaled = []
        for stage in stage_cfgs:
            updated = asdict(stage)
            updated["epochs"] = max(1, int(round(updated["epochs"] * args.epoch_scale)))
            scaled.append(StageConfig(**updated))
        stage_cfgs = scaled
    dataset = build_dataset(dataset_cfg)

    for split_name, samples in dataset.items():
        (output_dir / f"{split_name}.json").write_text(json.dumps(samples, indent=2))

    started = time.time()
    tokenizer, qwen_model = load_qwen_model(args.model_path)
    generation_baseline = run_generation_baseline(dataset["test"], dataset_cfg, tokenizer, qwen_model)
    save_json(output_dir / "generation_baseline.json", generation_baseline)

    feature_splits = {}
    for split_name, samples in dataset.items():
        feature_splits[split_name] = extract_hidden_features(samples, dataset_cfg, tokenizer, qwen_model, split_name)
        torch.save(feature_splits[split_name], output_dir / f"{split_name}_features.pt")

    hidden_dim = int(feature_splits["train"]["question_hidden"].size(-1))
    del qwen_model
    torch.cuda.empty_cache()

    device = torch.device(args.device)
    query_only_accuracy = train_query_only_baseline(feature_splits["train"], feature_splits["test"], device)

    train_loader = DataLoader(
        build_tensor_dataset(feature_splits["train"]),
        batch_size=dataset_cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        build_tensor_dataset(feature_splits["val"]),
        batch_size=dataset_cfg.train_batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        build_tensor_dataset(feature_splits["test"]),
        batch_size=dataset_cfg.train_batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_summaries = []
    teacher = None
    stage1_message_floats = stage_cfgs[0].num_slots * stage_cfgs[0].slot_dim

    for stage_cfg in stage_cfgs:
        print(f"==> Training {stage_cfg.name}")
        model, best_snapshot = train_stage(
            hidden_dim,
            dataset_cfg,
            stage_cfg,
            train_loader,
            val_loader,
            device,
            teacher=teacher,
        )
        clean_metrics = evaluate_model(model, test_loader, device)
        robust_metrics = evaluate_model(
            model,
            test_loader,
            device,
            override_slot_dropout=0.20,
            override_noise_std=0.06,
        )

        train_x, train_state, train_style = collect_latent_features(model, train_loader, device)
        test_x, test_state, test_style = collect_latent_features(model, test_loader, device)
        state_probe = train_state_probe(train_x, train_state, test_x, test_state, device)
        style_probe = train_style_probe(train_x, train_style, test_x, test_style, device)

        summary = summarize_stage(
            stage_cfg,
            best_snapshot,
            clean_metrics,
            robust_metrics,
            state_probe,
            style_probe,
            stage1_message_floats=stage1_message_floats,
        )
        all_summaries.append(summary)
        save_json(output_dir / f"{stage_cfg.name}_metrics.json", summary)
        torch.save(model.state_dict(), output_dir / f"{stage_cfg.name}.pt")

        teacher = model
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        print(
            f"{stage_cfg.name}: clean_acc={clean_metrics['accuracy']:.4f}, "
            f"robust_acc={robust_metrics['accuracy']:.4f}, "
            f"state_probe={state_probe['mean_field_accuracy']:.4f}, "
            f"style_probe={style_probe['accuracy']:.4f}"
        )

    write_summary(output_dir, dataset_cfg, stage_cfgs, all_summaries, generation_baseline, query_only_accuracy)
    save_json(
        output_dir / "run_metadata.json",
        {
            "dataset": asdict(dataset_cfg),
            "stages": [asdict(cfg) for cfg in stage_cfgs],
            "generation_baseline": generation_baseline,
            "query_only_classifier_accuracy": query_only_accuracy,
            "results": all_summaries,
            "runtime_seconds": time.time() - started,
        },
    )

    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
