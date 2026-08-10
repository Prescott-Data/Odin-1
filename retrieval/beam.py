from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import heapq, math, time
from .adapters import GraphAccessor, NodeId, RelId
from .confidence import EdgeConfidenceProvider, ConstantConfidence
from .budget import SearchBudget, BudgetTracker


@dataclass
class BeamParams:
    hop_limit: int = 3
    beam_width: int = 64
    max_paths: int = 200
    lambdas: Tuple[float, float, float, float] = (0.6, 0.2, 0.15, 0.05)
    allowed_relations: Optional[Set[RelId]] = None
    max_out_degree: Optional[int] = None


def default_recency(edge_timestamp: Optional[float], now_ts: Optional[float] = None, tau_days: float = 90.0) -> float:
    if edge_timestamp is None:
        return 1.0
    if now_ts is None:
        now_ts = time.time()
    dt_days = max(0.0, (now_ts - edge_timestamp) / (60 * 60 * 24))
    return math.exp(-dt_days / tau_days)


def safe_logp(x: float) -> float:
    return math.log(max(x, 1e-12))


def beam_search(
    accessor: GraphAccessor,
    community_id: str,
    seeds: List[NodeId],
    ppr_scores: List[Tuple[NodeId, float]],
    budget: Optional[SearchBudget] = None,
    beam_params: BeamParams = BeamParams(),
    conf_provider: EdgeConfidenceProvider = ConstantConfidence(0.8),
    edge_type_prior: Optional[Dict[RelId, float]] = None,
    edge_timestamp_lookup=None,
) -> Dict[str, object]:

    bt = BudgetTracker(budget or SearchBudget(max_paths=beam_params.max_paths))
    L1, L2, L3, L4 = beam_params.lambdas
    ppr = {n: p for n, p in ppr_scores}
    edge_type_prior = edge_type_prior or {}

    heap: List[Tuple[float, List[NodeId], List[Tuple[NodeId, RelId, NodeId]]]] = []
    for s in seeds:
        heapq.heappush(heap, (0.0, [s], []))

    best_paths = []

    def score_extension(u: NodeId, rel: RelId, v: NodeId) -> float:
        p1 = ppr.get(v, 1e-12)
        w_edge = edge_type_prior.get(rel, 1.0)
        c = conf_provider.confidence(u, rel, v)
        rec = default_recency(edge_timestamp_lookup(u, rel, v) if edge_timestamp_lookup else None)
        return L1 * safe_logp(p1) + L2 * safe_logp(w_edge) + L3 * safe_logp(c) + L4 * safe_logp(rec)

    early_stop_reason = None
    for hop in range(1, beam_params.hop_limit + 1):
        next_heap: List[Tuple[float, List[NodeId], List[Tuple[NodeId, RelId, NodeId]]]] = []
        while heap and not bt.over():
            logscore, path_nodes, path_edges = heapq.heappop(heap)
            u = path_nodes[-1]
            bt.tick_nodes(1)
            out_iter = accessor.iter_out(u)
            if beam_params.max_out_degree is not None:
                # Degree cap: take only first N neighbors
                out_iter = list(out_iter)[: beam_params.max_out_degree]
            for v, rel, _ in out_iter:
                if bt.over():
                    break
                bt.tick_edges(1)
                if v in path_nodes:
                    continue
                if beam_params.allowed_relations is not None and rel not in beam_params.allowed_relations:
                    continue
                inc = score_extension(u, rel, v)
                new_score = logscore + inc
                new_nodes = path_nodes + [v]
                new_edges = path_edges + [(u, rel, v)]
                heapq.heappush(next_heap, (new_score, new_nodes, new_edges))
                if len(next_heap) > beam_params.beam_width:
                    heapq.heappop(next_heap)
            if bt.timed_out():
                early_stop_reason = early_stop_reason or "timeout"
                break

        next_heap.sort(key=lambda x: x[0], reverse=True)
        for sc, nodes, edges in next_heap:
            best_paths.append((sc, nodes, edges))
            bt.tick_paths(1)
            if bt.over():
                break
        heap = next_heap
        if bt.over():
            if early_stop_reason is None:
                # Determine reason
                if bt.usage.nodes >= bt.budget.max_nodes:
                    early_stop_reason = "max_nodes"
                elif bt.usage.edges >= bt.budget.max_edges:
                    early_stop_reason = "max_edges"
                elif bt.usage.paths >= bt.budget.max_paths:
                    early_stop_reason = "max_paths"
                else:
                    early_stop_reason = "budget_exhausted"
            break

    best_paths.sort(key=lambda x: x[0], reverse=True)
    return {
        "paths": [
            {
                "score": float(sc),
                "nodes": ns,
                "edges": [{"u": u, "rel": r, "v": v} for (u, r, v) in es],
            }
            for sc, ns, es in best_paths[: beam_params.max_paths]
        ],
        "used_budget": bt.usage.__dict__,
        "trace": {"beam_width": beam_params.beam_width, "hop_limit": beam_params.hop_limit, "early_stop_reason": early_stop_reason},
    }


