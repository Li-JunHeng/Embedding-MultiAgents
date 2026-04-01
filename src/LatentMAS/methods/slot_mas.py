"""
SlotMAS: Slot-Attention Compressed Latent Communication for Multi-Agent Systems.

Key difference from LatentMAS: agents communicate through compressed slot
representations instead of full KV caches / hidden states.

Pipeline: Planner → compress → Critic → compress → Refiner → compress → Judger
Each agent receives ONLY the compressed slots from previous agents (not full KV).

Communication cost: num_slots × slot_dim × 2 bytes (fp16)
  e.g. 4 slots × 64 dim = 512 bytes  (vs ~100MB+ for full KV cache in LatentMAS)
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ModelWrapper
from prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
)
from utils import (
    extract_gsm8k_answer,
    extract_markdown_python_block,
    normalize_answer,
    run_with_timeout,
)

from . import default_agents


class SlotAttentionCompressor(nn.Module):
    """Compress variable-length hidden states into fixed-size slots."""

    def __init__(self, d_model: int, num_slots: int = 4, slot_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Cross-attention: learned slot queries attend over hidden states
        self.slot_queries = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

        # Compress: d_model → slot_dim (the actual bottleneck)
        self.compress = nn.Linear(d_model, slot_dim, bias=False)
        # Decompress: slot_dim → d_model (reconstruct for next agent's embedding space)
        self.decompress = nn.Linear(slot_dim, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = hidden_states.shape
        h = self.layer_norm(hidden_states)

        queries = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
        Q = self.q_proj(queries)
        K = self.k_proj(h)

        scale = math.sqrt(D)
        attn_weights = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / scale, dim=-1)
        slots = torch.bmm(attn_weights, hidden_states)  # (B, num_slots, D)

        compressed = self.compress(
            slots
        )  # (B, num_slots, slot_dim) — THIS is transmitted
        decoded = self.decompress(compressed)  # (B, num_slots, D) — fed to next agent
        return compressed, decoded

    def message_bytes(self) -> int:
        return self.num_slots * self.slot_dim * 2  # fp16


class SlotMASMethod:
    """
    Multi-agent with slot-attention compressed communication.

    Flow for each non-judger agent:
      1. Prepend decoded slots from previous agents as soft prefix
      2. Run forward pass to get hidden states
      3. Run latent reasoning steps (same as LatentMAS)
      4. Compress output hidden states → slots (the message)

    Flow for judger:
      1. Prepend all decoded slots as soft prefix
      2. Generate text answer

    Key: the compressed slots are the ONLY information passed between agents.
    No KV cache sharing. This is what makes it efficient.
    """

    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        num_slots: int = 4,
        slot_dim: int = 64,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = "slot_mas"
        self.task = args.task

        d_model = model.model.config.hidden_size
        first_param = next(model.model.parameters())
        self._dtype = first_param.dtype
        self._device = first_param.device

        # Use actual slot_dim (default 64), not d_model
        actual_slot_dim = slot_dim if slot_dim > 0 else 64
        self.compressor = (
            SlotAttentionCompressor(
                d_model=d_model,
                num_slots=num_slots,
                slot_dim=actual_slot_dim,
            )
            .to(self._device)
            .to(self._dtype)
        )

        # Load trained compressor weights if provided
        compressor_path = getattr(args, "compressor_path", None)
        if compressor_path:
            print(f"[SlotMAS] Loading trained compressor from {compressor_path}")
            state = torch.load(compressor_path, weights_only=True)
            self.compressor.load_state_dict(state)
        else:
            print(f"[SlotMAS] WARNING: Using random compressor weights (not trained)")

        self.compressor.eval()

        self.embedding_layer = model.model.get_input_embeddings()
        self._msg_bytes = self.compressor.message_bytes()
        print(
            f"[SlotMAS] num_slots={num_slots}, slot_dim={actual_slot_dim}, "
            f"d_model={d_model}, message={self._msg_bytes} bytes/agent, "
            f"compression={d_model * 2 / actual_slot_dim:.0f}x vs full hidden"
        )

    @torch.no_grad()
    def _agent_forward_and_compress(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prev_slots_decoded: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one agent: optional slot prefix → forward → latent steps → compress."""
        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        prompt_embeds = self.embedding_layer(input_ids)

        # Prepend decoded slots from previous agents as soft prefix
        if prev_slots_decoded is not None:
            prev_slots_decoded = prev_slots_decoded.to(prompt_embeds.device)
            inputs_embeds = torch.cat([prev_slots_decoded, prompt_embeds], dim=1)
            slot_mask = torch.ones(
                prev_slots_decoded.shape[0],
                prev_slots_decoded.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_mask = torch.cat([slot_mask, attention_mask], dim=1)
        else:
            inputs_embeds = prompt_embeds
            full_mask = attention_mask

        # Forward pass
        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_kv = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1:, :]

        # Latent reasoning steps
        all_hiddens = [last_hidden]
        for _ in range(self.latent_steps):
            latent_vec = self.model._apply_latent_realignment(
                last_hidden.squeeze(1), self.model.model
            )
            latent_embed = latent_vec.unsqueeze(1)
            past_len = past_kv[0][0].shape[-2]
            latent_mask = torch.ones(
                latent_embed.shape[0],
                past_len + 1,
                dtype=torch.long,
                device=self._device,
            )
            outputs = self.model.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_kv = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1:, :]
            all_hiddens.append(last_hidden)

        # Compress: slot attention over latent hidden sequence
        hidden_seq = torch.cat(all_hiddens, dim=1)  # (B, 1+latent_steps, D)
        compressed, decoded = self.compressor(hidden_seq)
        # compressed: (B, num_slots, slot_dim) — the transmitted message
        # decoded: (B, num_slots, D) — reconstructed embeddings for next agent
        return compressed, decoded

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        all_decoded_slots: List[torch.Tensor] = []

        for agent in self.agents:
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method="slot_mas",
                        args=self.args,
                    )
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method="slot_mas",
                        args=self.args,
                    )
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = (
                self.model.prepare_chat_batch(
                    batch_messages, add_generation_prompt=True
                )
            )

            if self.args.think:
                wrapped_prompts = [f"{p}<think>" for p in prompts]
            else:
                wrapped_prompts = prompts
            wrapped_enc = self.model.tokenizer(
                wrapped_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            wrapped_ids = wrapped_enc["input_ids"].to(self.model.device)
            wrapped_mask = wrapped_enc["attention_mask"].to(self.model.device)

            if agent.role != "judger":
                prev_decoded = None
                if all_decoded_slots:
                    prev_decoded = torch.cat(all_decoded_slots, dim=1)

                compressed, decoded = self._agent_forward_and_compress(
                    wrapped_ids,
                    wrapped_mask,
                    prev_decoded,
                )
                all_decoded_slots.append(decoded)

                for idx in range(batch_size):
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "latent_steps": self.latent_steps,
                            "compressed_msg_bytes": self._msg_bytes,
                            "output": "",
                        }
                    )
            else:
                # Judger: prepend all slots, then generate text
                prev_decoded = None
                if all_decoded_slots:
                    prev_decoded = torch.cat(all_decoded_slots, dim=1)

                wrapped_ids = wrapped_ids.to(self._device)
                wrapped_mask = wrapped_mask.to(self._device)
                prompt_embeds = self.embedding_layer(wrapped_ids)

                if prev_decoded is not None:
                    prev_decoded = prev_decoded.to(prompt_embeds.device)
                    inputs_embeds = torch.cat([prev_decoded, prompt_embeds], dim=1)
                    slot_mask = torch.ones(
                        prev_decoded.shape[0],
                        prev_decoded.shape[1],
                        dtype=wrapped_mask.dtype,
                        device=wrapped_mask.device,
                    )
                    full_mask = torch.cat([slot_mask, wrapped_mask], dim=1)
                else:
                    inputs_embeds = prompt_embeds
                    full_mask = wrapped_mask

                prefix_len = inputs_embeds.shape[1]
                gen_outputs = self.model.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.model.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
                for idx in range(batch_size):
                    gen_ids = gen_outputs.sequences[idx, prefix_len:]
                    text = self.model.tokenizer.decode(
                        gen_ids, skip_special_tokens=True
                    ).strip()
                    final_texts[idx] = text
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "output": text,
                            "total_msg_bytes": self._msg_bytes * len(all_decoded_slots),
                        }
                    )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ["mbppplus", "humanevalplus"]:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")
                if pred is None:
                    ok = False
                else:
                    ok, _ = run_with_timeout(pred + "\n" + gold, timeout=10)
            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    ok = int(pred) == int(gold)
                except (ValueError, TypeError):
                    ok = False
            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "msg_bytes_per_agent": self._msg_bytes,
                    "total_msg_bytes": self._msg_bytes * len(all_decoded_slots),
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
