from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import torch

from memory_bank import LatentMemoryAdapter, PerSampleMemoryBank
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


class MemoryMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        memory_dim: int = 256,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        if model.use_vllm:
            raise ValueError('memory_mas currently supports only the HuggingFace backend. Please rerun without --use_vllm.')

        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.memory_dim = memory_dim
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'memory_mas'
        self.task = args.task

        d_model = model.model.config.hidden_size
        first_param = next(model.model.parameters())
        self._dtype = first_param.dtype
        self._device = first_param.device
        mem_dev = getattr(args, "memory_device", None) if args is not None else None
        self._memory_device = torch.device(mem_dev) if mem_dev else self._device
        if self._memory_device != self._device:
            print(
                f"[memory_mas] Qwen/transformers on {self._device}, "
                f"LatentMemoryAdapter + memory bank on {self._memory_device}"
            )
        self.embedding_layer = model.model.get_input_embeddings()
        self.memory_adapter = LatentMemoryAdapter(d_model=d_model, memory_dim=memory_dim).to(self._memory_device).to(
            self._dtype
        )

        adapter_path = getattr(args, "adapter_path", None)
        if adapter_path:
            print(f"[memory_mas] Loading trained adapter from {adapter_path}")
            state = torch.load(adapter_path, map_location=self._memory_device, weights_only=True)
            self.memory_adapter.load_state_dict(state)
        else:
            print("[memory_mas] WARNING: Using random adapter weights (not trained)")

        self.memory_adapter.eval()

    def _build_messages(self, item: Dict, role: str) -> List[Dict]:
        if self.args.prompt == 'sequential':
            return build_agent_message_sequential_latent_mas(
                role=role,
                question=item['question'],
                context='',
                method=self.method_name,
                args=self.args,
            )
        if self.args.prompt == 'hierarchical':
            return build_agent_message_hierarchical_latent_mas(
                role=role,
                question=item['question'],
                context='',
                method=self.method_name,
                args=self.args,
            )
        raise ValueError(f'Unsupported prompt mode: {self.args.prompt}')

    def _prepare_prompt(self, item: Dict, role: str) -> tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        messages = self._build_messages(item, role)
        prompt = self.model.render_chat(messages, add_generation_prompt=True)
        if getattr(self.args, 'think', False):
            prompt = f'{prompt}<think>'
        encoded = self.model.tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=False,
        )
        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.model.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt, input_ids, attention_mask, tokens

    def _read_memory_prefix(
        self,
        bank: PerSampleMemoryBank,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if bank.is_empty():
            return None
        _, _, query_hidden = self.model.rollout_latent_sequence(
            input_ids,
            attention_mask=attention_mask,
            latent_steps=0,
        )
        return bank.read(query_hidden, self.memory_adapter)

    def _generate_with_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_embeds: Optional[torch.Tensor],
    ) -> str:
        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        prompt_embeds = self.embedding_layer(input_ids)

        if prefix_embeds is not None:
            prefix_embeds = prefix_embeds.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            prefix_mask = torch.ones(
                prefix_embeds.shape[0],
                prefix_embeds.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            inputs_embeds = prompt_embeds
            full_mask = attention_mask

        prefix_len = inputs_embeds.shape[1]
        outputs = self.model.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=self.judger_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.model.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        generated_ids = outputs.sequences[0, prefix_len:]
        return self.model.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _evaluate_prediction(self, item: Dict, final_text: str) -> tuple[str, str, bool]:
        if self.task in ['mbppplus', 'humanevalplus']:
            pred = extract_markdown_python_block(final_text)
            gold = item.get('gold', '')
            if pred is None:
                ok = False
            else:
                ok, _ = run_with_timeout(pred + '\n' + gold, timeout=10)
            return pred, gold, ok

        pred = normalize_answer(extract_gsm8k_answer(final_text))
        if self.task in ['aime2024', 'aime2025']:
            gold = str(item.get('gold', '')).strip()
            try:
                ok = int(pred) == int(gold)
            except (TypeError, ValueError):
                ok = False
            return pred, gold, ok

        gold = item.get('gold', '')
        ok = (pred == gold) if (pred and gold) else False
        return pred, gold, ok

    @torch.no_grad()
    def run_item(self, item: Dict) -> Dict:
        bank = PerSampleMemoryBank()
        agent_trace: List[Dict] = []
        final_text = ''

        for agent in self.agents:
            prompt, input_ids, attention_mask, input_tokens = self._prepare_prompt(item, agent.role)
            memory_tokens_available = bank.total_tokens()
            memory_prefix = self._read_memory_prefix(bank, input_ids, attention_mask)
            memory_read_used = memory_prefix is not None

            if agent.role != 'judger':
                _, hidden_seq, _ = self.model.rollout_latent_sequence(
                    input_ids,
                    attention_mask=attention_mask,
                    latent_steps=self.latent_steps,
                    prefix_embeds=memory_prefix,
                )
                hidden_for_bank = hidden_seq.to(device=self._memory_device, dtype=self._dtype)
                memory_tokens_written = bank.add(agent.role, hidden_for_bank, self.memory_adapter)
                trimmed_ids = input_ids[0][attention_mask[0].bool()].to('cpu').tolist()
                agent_trace.append(
                    {
                        'name': agent.name,
                        'role': agent.role,
                        'input': prompt,
                        'input_ids': trimmed_ids,
                        'input_tokens': input_tokens,
                        'latent_steps': self.latent_steps,
                        'memory_dim': self.memory_dim,
                        'memory_read_used': memory_read_used,
                        'memory_tokens_available': memory_tokens_available,
                        'memory_tokens_written': memory_tokens_written,
                        'output': '',
                    }
                )
            else:
                final_text = self._generate_with_prefix(input_ids, attention_mask, memory_prefix)
                trimmed_ids = input_ids[0][attention_mask[0].bool()].to('cpu').tolist()
                agent_trace.append(
                    {
                        'name': agent.name,
                        'role': agent.role,
                        'input': prompt,
                        'input_ids': trimmed_ids,
                        'input_tokens': input_tokens,
                        'memory_dim': self.memory_dim,
                        'memory_read_used': memory_read_used,
                        'memory_tokens_available': memory_tokens_available,
                        'memory_tokens_written': 0,
                        'output': final_text,
                    }
                )

        pred, gold, ok = self._evaluate_prediction(item, final_text)
        return {
            'question': item['question'],
            'gold': gold,
            'solution': item['solution'],
            'prediction': pred,
            'raw_prediction': final_text,
            'agents': agent_trace,
            'correct': ok,
            'memory_dim': self.memory_dim,
            'total_memory_tokens': bank.total_tokens(),
        }

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError('Batch size exceeds configured generate_bs')
        return [self.run_item(item) for item in items]
