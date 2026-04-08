from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryEntry:
    agent_role: str
    token_count: int
    compressed_tokens: torch.Tensor
    value_tokens: torch.Tensor


class LatentMemoryAdapter(nn.Module):
    def __init__(self, d_model: int, memory_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim
        self.layer_norm = nn.LayerNorm(d_model)
        self.memory_proj = nn.Linear(d_model, memory_dim, bias=False)

    def compress_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.memory_proj(self.layer_norm(hidden_states))

    def read(
        self,
        query_hidden: torch.Tensor,
        compressed_tokens: torch.Tensor,
        value_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if query_hidden.dim() == 2:
            query_hidden = query_hidden.unsqueeze(1)
        if compressed_tokens.dim() == 2:
            compressed_tokens = compressed_tokens.unsqueeze(0)
        if value_tokens.dim() == 2:
            value_tokens = value_tokens.unsqueeze(0)

        query_hidden = query_hidden.to(device=value_tokens.device, dtype=value_tokens.dtype)
        compressed_tokens = compressed_tokens.to(device=value_tokens.device, dtype=value_tokens.dtype)
        query_proj = self.compress_hidden_states(query_hidden)

        attn_scores = torch.matmul(query_proj, compressed_tokens.transpose(1, 2))
        attn_scores = attn_scores / math.sqrt(self.memory_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, value_tokens)


class PerSampleMemoryBank:
    def __init__(self) -> None:
        self.entries: List[MemoryEntry] = []

    def reset(self) -> None:
        self.entries.clear()

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def total_tokens(self) -> int:
        return sum(entry.token_count for entry in self.entries)

    def add(
        self,
        agent_role: str,
        hidden_seq: torch.Tensor,
        adapter: LatentMemoryAdapter,
    ) -> int:
        if hidden_seq.dim() == 3:
            if hidden_seq.shape[0] != 1:
                raise ValueError('PerSampleMemoryBank expects a single-sample hidden sequence')
            hidden_seq = hidden_seq.squeeze(0)
        if hidden_seq.dim() != 2:
            raise ValueError('hidden_seq must have shape [tokens, d_model]')

        hidden_seq = hidden_seq.detach()
        compressed = adapter.compress_hidden_states(hidden_seq.unsqueeze(0)).squeeze(0).detach()
        entry = MemoryEntry(
            agent_role=agent_role,
            token_count=int(hidden_seq.shape[0]),
            compressed_tokens=compressed.clone(),
            value_tokens=hidden_seq.clone(),
        )
        self.entries.append(entry)
        return entry.token_count

    def _stack_tokens(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_empty():
            raise ValueError('Cannot stack tokens from an empty memory bank')
        compressed = torch.cat([entry.compressed_tokens for entry in self.entries], dim=0)
        values = torch.cat([entry.value_tokens for entry in self.entries], dim=0)
        return compressed, values

    def read(
        self,
        query_hidden: torch.Tensor,
        adapter: LatentMemoryAdapter,
    ) -> Optional[torch.Tensor]:
        if self.is_empty():
            return None
        compressed, values = self._stack_tokens()
        return adapter.read(query_hidden, compressed, values)

    def stats(self) -> Dict[str, int]:
        return {
            'num_entries': len(self.entries),
            'total_tokens': self.total_tokens(),
        }
