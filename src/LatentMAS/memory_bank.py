from __future__ import annotations

import math
from dataclasses import dataclass, field
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


@dataclass
class L1Segment:
    segment_id: int
    agent_role: str
    agent_index: int
    start: int
    end: int
    summary_key: torch.Tensor
    compressed_tokens: torch.Tensor
    value_tokens: torch.Tensor
    cluster_id: int

    @property
    def token_count(self) -> int:
        return int(self.end - self.start)


@dataclass
class L2ThoughtCluster:
    cluster_id: int
    member_segment_ids: List[int]
    summary_key: torch.Tensor
    pooled_value: torch.Tensor
    kind: str
    priority_weight: float
    total_weight: float = 1.0


@dataclass
class L3AgentSummary:
    agent_role: str
    agent_index: int
    summary_key: torch.Tensor
    segment_ids: List[int] = field(default_factory=list)


ROLE_MAP: Dict[str, int] = {
    "planner": 0,
    "critic": 1,
    "refiner": 2,
    "judger": 3,
}
NUM_ROLES: int = len(ROLE_MAP)


def _cosine_score(query: torch.Tensor, key: torch.Tensor) -> float:
    query = query.reshape(-1)
    key = key.reshape(-1)
    return float(F.cosine_similarity(query.unsqueeze(0), key.unsqueeze(0), dim=-1).item())


def _topk(items: List[Tuple[float, object]], k: int) -> List[Tuple[float, object]]:
    if k <= 0:
        return []
    return sorted(items, key=lambda pair: pair[0], reverse=True)[:k]


class LatentMemoryAdapter(nn.Module):
    def __init__(self, d_model: int, memory_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim
        self.layer_norm = nn.LayerNorm(d_model)
        self.memory_proj = nn.Linear(d_model, memory_dim, bias=False)
        self.role_embeddings = nn.Embedding(NUM_ROLES, d_model)

    def compress_hidden_states(
        self,
        hidden_states: torch.Tensor,
        role_id: Optional[int] = None,
    ) -> torch.Tensor:
        h = self.layer_norm(hidden_states)
        if role_id is not None:
            role_emb = self.role_embeddings(torch.tensor(role_id, device=h.device))
            h = h + role_emb
        return self.memory_proj(h)

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
    def __init__(
        self,
        *,
        segment_length: int = 4,
        top_agents: int = 2,
        top_clusters: int = 4,
        top_segments: int = 4,
        max_prefix_tokens: int = 64,
        gate_scale: float = 4.0,
        merge_threshold: float = 0.92,
        difference_threshold: float = 0.55,
        difference_boost: float = 1.25,
        consensus_penalty: float = 0.85,
    ) -> None:
        self.segment_length = max(1, int(segment_length))
        self.top_agents = max(1, int(top_agents))
        self.top_clusters = max(1, int(top_clusters))
        self.top_segments = max(1, int(top_segments))
        self.max_prefix_tokens = max(1, int(max_prefix_tokens))
        self.gate_scale = float(gate_scale)
        self.merge_threshold = float(merge_threshold)
        self.difference_threshold = float(difference_threshold)
        self.difference_boost = float(difference_boost)
        self.consensus_penalty = float(consensus_penalty)

        self.entries: List[MemoryEntry] = []
        self.segments: List[L1Segment] = []
        self.segment_by_id: Dict[int, L1Segment] = {}
        self.clusters: Dict[int, L2ThoughtCluster] = {}
        self.agent_summaries: Dict[Tuple[str, int], L3AgentSummary] = {}
        self._next_segment_id = 0
        self._next_cluster_id = 0
        self._last_read_stats: Dict[str, object] = {}
        self._last_write_stats: Dict[str, object] = {}

    def reset(self) -> None:
        self.entries.clear()
        self.segments.clear()
        self.segment_by_id.clear()
        self.clusters.clear()
        self.agent_summaries.clear()
        self._next_segment_id = 0
        self._next_cluster_id = 0
        self._last_read_stats = {}
        self._last_write_stats = {}

    def is_empty(self) -> bool:
        return len(self.segments) == 0

    def total_tokens(self) -> int:
        return sum(segment.token_count for segment in self.segments)

    def _summary_key(
        self,
        hidden_states: torch.Tensor,
        adapter: LatentMemoryAdapter,
        agent_role: str,
    ) -> torch.Tensor:
        role_id = ROLE_MAP.get(agent_role)
        pooled = hidden_states.mean(dim=0, keepdim=True).unsqueeze(0)
        return (
            adapter.compress_hidden_states(pooled, role_id=role_id)
            .squeeze(0)
            .squeeze(0)
            .detach()
            .clone()
        )

    def _assign_cluster(
        self,
        segment_summary: torch.Tensor,
        segment_value: torch.Tensor,
    ) -> Tuple[int, bool]:
        if not self.clusters:
            cluster_id = self._next_cluster_id
            self._next_cluster_id += 1
            self.clusters[cluster_id] = L2ThoughtCluster(
                cluster_id=cluster_id,
                member_segment_ids=[],
                summary_key=segment_summary.clone(),
                pooled_value=segment_value.mean(dim=0).detach().clone(),
                kind="difference",
                priority_weight=self.difference_boost,
                total_weight=1.0,
            )
            return cluster_id, False

        best_cluster: Optional[L2ThoughtCluster] = None
        best_score = -1.0
        for cluster in self.clusters.values():
            score = _cosine_score(segment_summary, cluster.summary_key)
            if score > best_score:
                best_score = score
                best_cluster = cluster

        assert best_cluster is not None
        if best_score >= self.merge_threshold:
            cluster = best_cluster
            old_weight = cluster.total_weight
            new_weight = old_weight + 1.0
            cluster.summary_key = ((cluster.summary_key * old_weight) + segment_summary) / new_weight
            pooled_value = segment_value.mean(dim=0)
            cluster.pooled_value = ((cluster.pooled_value * old_weight) + pooled_value) / new_weight
            cluster.total_weight = new_weight
            cluster.kind = "consensus"
            cluster.priority_weight = self.consensus_penalty
            return cluster.cluster_id, True

        cluster_id = self._next_cluster_id
        self._next_cluster_id += 1
        if best_score <= self.difference_threshold:
            kind = "difference"
            priority = self.difference_boost
        else:
            kind = "neutral"
            priority = 1.0
        self.clusters[cluster_id] = L2ThoughtCluster(
            cluster_id=cluster_id,
            member_segment_ids=[],
            summary_key=segment_summary.clone(),
            pooled_value=segment_value.mean(dim=0).detach().clone(),
            kind=kind,
            priority_weight=priority,
            total_weight=1.0,
        )
        return cluster_id, False

    def _update_agent_summary(
        self,
        agent_role: str,
        agent_index: int,
        segment_ids: List[int],
    ) -> None:
        summary_key = torch.stack(
            [self.segment_by_id[segment_id].summary_key for segment_id in segment_ids], dim=0
        ).mean(dim=0)
        key = (agent_role, agent_index)
        if key in self.agent_summaries:
            self.agent_summaries[key].summary_key = summary_key.detach().clone()
            self.agent_summaries[key].segment_ids.extend(segment_ids)
        else:
            self.agent_summaries[key] = L3AgentSummary(
                agent_role=agent_role,
                agent_index=agent_index,
                summary_key=summary_key.detach().clone(),
                segment_ids=list(segment_ids),
            )

    def add(
        self,
        agent_role: str,
        hidden_seq: torch.Tensor,
        adapter: LatentMemoryAdapter,
        *,
        agent_index: int = 0,
    ) -> int:
        if hidden_seq.dim() == 3:
            if hidden_seq.shape[0] != 1:
                raise ValueError("PerSampleMemoryBank expects a single-sample hidden sequence")
            hidden_seq = hidden_seq.squeeze(0)
        if hidden_seq.dim() != 2:
            raise ValueError("hidden_seq must have shape [tokens, d_model]")

        hidden_seq = hidden_seq.detach()
        role_id = ROLE_MAP.get(agent_role)
        compressed = adapter.compress_hidden_states(
            hidden_seq.unsqueeze(0), role_id=role_id
        ).squeeze(0).detach()
        self.entries.append(
            MemoryEntry(
                agent_role=agent_role,
                token_count=int(hidden_seq.shape[0]),
                compressed_tokens=compressed.clone(),
                value_tokens=hidden_seq.clone(),
            )
        )

        written_segment_ids: List[int] = []
        consensus_merges = 0
        difference_segments = 0
        neutral_segments = 0

        for start in range(0, hidden_seq.shape[0], self.segment_length):
            end = min(start + self.segment_length, hidden_seq.shape[0])
            segment_values = hidden_seq[start:end].clone()
            segment_compressed = compressed[start:end].clone()
            segment_summary = self._summary_key(segment_values, adapter, agent_role)
            cluster_id, merged = self._assign_cluster(segment_summary, segment_values)
            segment = L1Segment(
                segment_id=self._next_segment_id,
                agent_role=agent_role,
                agent_index=agent_index,
                start=start,
                end=end,
                summary_key=segment_summary,
                compressed_tokens=segment_compressed,
                value_tokens=segment_values,
                cluster_id=cluster_id,
            )
            self._next_segment_id += 1
            self.segments.append(segment)
            self.segment_by_id[segment.segment_id] = segment
            self.clusters[cluster_id].member_segment_ids.append(segment.segment_id)
            written_segment_ids.append(segment.segment_id)
            if merged:
                consensus_merges += 1
            elif self.clusters[cluster_id].kind == "difference":
                difference_segments += 1
            else:
                neutral_segments += 1

        self._update_agent_summary(agent_role, agent_index, written_segment_ids)
        self._last_write_stats = {
            "agent_role": agent_role,
            "agent_index": agent_index,
            "segments_written": len(written_segment_ids),
            "consensus_merges": consensus_merges,
            "difference_segments": difference_segments,
            "neutral_segments": neutral_segments,
            "total_clusters": len(self.clusters),
        }
        return int(hidden_seq.shape[0])

    def _select_agent_summaries(self, query_key: torch.Tensor) -> List[Tuple[float, L3AgentSummary]]:
        scored = [
            (_cosine_score(query_key, summary.summary_key), summary)
            for summary in self.agent_summaries.values()
        ]
        return _topk(scored, self.top_agents)

    def _select_clusters(
        self,
        query_key: torch.Tensor,
        selected_agents: List[L3AgentSummary],
    ) -> List[Tuple[float, L2ThoughtCluster]]:
        if not selected_agents:
            return []
        allowed_segment_ids = {
            segment_id for summary in selected_agents for segment_id in summary.segment_ids
        }
        scored: List[Tuple[float, L2ThoughtCluster]] = []
        for cluster in self.clusters.values():
            if not any(segment_id in allowed_segment_ids for segment_id in cluster.member_segment_ids):
                continue
            score = _cosine_score(query_key, cluster.summary_key) * cluster.priority_weight
            scored.append((score, cluster))
        return _topk(scored, self.top_clusters)

    def _select_segments(
        self,
        query_key: torch.Tensor,
        selected_clusters: List[L2ThoughtCluster],
        allowed_segment_ids: Optional[set[int]] = None,
    ) -> List[Tuple[float, L1Segment]]:
        if not selected_clusters:
            return []
        scored: List[Tuple[float, L1Segment]] = []
        for cluster in selected_clusters:
            for segment_id in cluster.member_segment_ids:
                if allowed_segment_ids is not None and segment_id not in allowed_segment_ids:
                    continue
                segment = self.segment_by_id[segment_id]
                score = _cosine_score(query_key, segment.summary_key)
                scored.append((score, segment))
        deduped: Dict[int, Tuple[float, L1Segment]] = {}
        for score, segment in scored:
            prev = deduped.get(segment.segment_id)
            if prev is None or score > prev[0]:
                deduped[segment.segment_id] = (score, segment)
        return _topk(list(deduped.values()), self.top_segments)

    def read(
        self,
        query_hidden: torch.Tensor,
        adapter: LatentMemoryAdapter,
        *,
        return_stats: bool = False,
    ) -> Optional[torch.Tensor] | Tuple[Optional[torch.Tensor], Dict[str, object]]:
        if self.is_empty():
            if return_stats:
                return None, {
                    "selected_agents": [],
                    "selected_clusters": [],
                    "selected_segments": [],
                    "gate_weights": [],
                    "prefix_tokens": 0,
                    "consensus_clusters": 0,
                    "difference_clusters": 0,
                }
            return None

        if query_hidden.dim() == 3:
            query_hidden = query_hidden.squeeze(0)
        if query_hidden.dim() == 2:
            query_hidden = query_hidden[-1]
        if query_hidden.dim() != 1:
            raise ValueError("query_hidden must reduce to shape [d_model]")

        query_hidden = query_hidden.detach()
        query_key = (
            adapter.compress_hidden_states(query_hidden.unsqueeze(0).unsqueeze(0))
            .squeeze(0)
            .squeeze(0)
            .detach()
        )

        selected_agent_pairs = self._select_agent_summaries(query_key)
        selected_agents = [summary for _, summary in selected_agent_pairs]
        allowed_segment_ids = {
            segment_id for summary in selected_agents for segment_id in summary.segment_ids
        }
        selected_cluster_pairs = self._select_clusters(query_key, selected_agents)
        selected_clusters = [cluster for _, cluster in selected_cluster_pairs]
        selected_segment_pairs = self._select_segments(
            query_key,
            selected_clusters,
            allowed_segment_ids=allowed_segment_ids,
        )

        gated_segments: List[torch.Tensor] = []
        gate_weights: List[float] = []
        selected_segment_ids: List[int] = []
        selected_cluster_ids = [cluster.cluster_id for cluster in selected_clusters]
        prefix_tokens = 0

        for score, segment in selected_segment_pairs:
            cluster = self.clusters[segment.cluster_id]
            gate = torch.sigmoid(
                torch.tensor(self.gate_scale * score, device=segment.value_tokens.device)
            ).item()
            gate = max(0.0, min(1.5, gate * cluster.priority_weight))
            remaining = self.max_prefix_tokens - prefix_tokens
            if remaining <= 0:
                break
            value_tokens = segment.value_tokens[:remaining]
            gated_segments.append(value_tokens * gate)
            gate_weights.append(float(gate))
            selected_segment_ids.append(segment.segment_id)
            prefix_tokens += value_tokens.shape[0]

        prefix = None
        if gated_segments:
            prefix = torch.cat(gated_segments, dim=0).unsqueeze(0)

        read_stats = {
            "selected_agents": [
                {
                    "agent_role": summary.agent_role,
                    "agent_index": summary.agent_index,
                    "score": float(score),
                }
                for score, summary in selected_agent_pairs
            ],
            "selected_clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "kind": cluster.kind,
                    "priority_weight": float(cluster.priority_weight),
                    "score": float(score),
                }
                for score, cluster in selected_cluster_pairs
            ],
            "selected_segments": selected_segment_ids,
            "selected_cluster_ids": selected_cluster_ids,
            "gate_weights": gate_weights,
            "prefix_tokens": prefix_tokens,
            "consensus_clusters": sum(1 for cluster in self.clusters.values() if cluster.kind == "consensus"),
            "difference_clusters": sum(1 for cluster in self.clusters.values() if cluster.kind == "difference"),
        }
        self._last_read_stats = read_stats
        if return_stats:
            return prefix, read_stats
        return prefix

    def stats(self) -> Dict[str, int]:
        return {
            "num_entries": len(self.entries),
            "num_segments": len(self.segments),
            "num_clusters": len(self.clusters),
            "num_agent_summaries": len(self.agent_summaries),
            "total_tokens": self.total_tokens(),
        }

    @property
    def last_read_stats(self) -> Dict[str, object]:
        return dict(self._last_read_stats)

    @property
    def last_write_stats(self) -> Dict[str, object]:
        return dict(self._last_write_stats)
