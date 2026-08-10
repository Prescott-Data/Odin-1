from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import math

from .adapters import NodeId, RelId, GraphAccessor
from .confidence import EdgeConfidenceProvider, ConstantConfidence


@dataclass
class PathScoreConfig:
    relation_type_prior: Optional[Dict[RelId, float]] = None
    recency_tau_days: float = 30.0
    recency_clamp: Tuple[float, float] = (0.5, 1.0)
    power_mean_rho: float = 0.5
    # New signals for GNN/Bridge integration
    bridge_boost: float = 1.5  # Multiplier for bridge nodes
    affinity_boost: float = 1.5  # Multiplier for high-affinity cross-community edges
    use_bridge_scoring: bool = True



def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def combine_edge_confidence(raw: float, npll: Optional[float] = None, calibrated: Optional[float] = None,
                            weights: Tuple[float, float, float] = (0.5, 0.4, 0.1)) -> float:
    w_raw, w_npll, w_cal = weights
    terms: List[Tuple[float, float]] = [(w_raw, raw)]
    if npll is not None:
        terms.append((w_npll, npll))
    if calibrated is not None:
        terms.append((w_cal, calibrated))
    # Normalize weights present
    total_w = sum(w for w, _ in terms) or 1.0
    norm = [(w / total_w, p) for w, p in terms]
    s = sum(w * _logit(p) for w, p in norm)
    return _sigmoid(s)


def power_mean(values: Iterable[float], rho: float) -> float:
    values = list(values)
    if not values:
        return 0.0
    if rho == 0.0:
        # geometric mean
        return math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values))
    acc = sum((max(v, 1e-12)) ** rho for v in values) / len(values)
    return max(acc, 1e-12) ** (1.0 / rho)


def recency_factor(edge_timestamps: List[Optional[float]], now_ts: Optional[float], tau_days: float,
                   clamp: Tuple[float, float]) -> float:
    if not edge_timestamps:
        return 1.0
    lo, hi = clamp
    total = 1.0
    for ts in edge_timestamps:
        if ts is None or now_ts is None:
            continue
        dt_days = max(0.0, (now_ts - ts) / (60 * 60 * 24))
        total *= math.exp(-dt_days / max(tau_days, 1e-6))
    return min(max(total, lo), hi)


def path_score(
    path_edges: List[Tuple[NodeId, RelId, NodeId]],
    node_ppr: Dict[NodeId, float],
    conf_provider: EdgeConfidenceProvider,
    now_ts: Optional[float],
    edge_timestamp_lookup,
    cfg: PathScoreConfig = PathScoreConfig(),
    # New params
    node_bridge_scores: Optional[Dict[NodeId, float]] = None,
    edge_affinity_scores: Optional[Dict[Tuple[NodeId, NodeId], float]] = None,
) -> float:
    # Edge confidence term (product of effective confidences)
    edge_confs: List[float] = []
    type_priors: List[float] = []
    ts_list: List[Optional[float]] = []
    affinity_mults: List[float] = []  # New
    
    priors = cfg.relation_type_prior or {}
    for u, rel, v in path_edges:
        c = conf_provider.confidence(u, rel, v)
        edge_confs.append(max(min(c, 1.0), 1e-12))
        type_priors.append(max(priors.get(rel, 1.0), 1e-6))
        ts_list.append(edge_timestamp_lookup(u, rel, v) if edge_timestamp_lookup else None)
        
        # Affinity term
        if cfg.use_bridge_scoring and edge_affinity_scores:
            # Check (u,v) or (v,u)
            aff = edge_affinity_scores.get((u, v), edge_affinity_scores.get((v, u), 0.0))
            if aff > 0:
                affinity_mults.append(1.0 + (aff * cfg.affinity_boost))
            else:
                affinity_mults.append(1.0)
        else:
            affinity_mults.append(1.0)

    edge_term = 1.0
    for c in edge_confs:
        edge_term *= c

    # PPR term: power mean across unique nodes on the path
    path_nodes: List[NodeId] = [path_edges[0][0]] + [v for (_, _, v) in path_edges] if path_edges else []
    ppr_vals = [node_ppr.get(n, 1e-12) for n in path_nodes]
    ppr_term = power_mean(ppr_vals, cfg.power_mean_rho)

    # Bridge term: boost nodes that are bridges
    bridge_term = 1.0
    if cfg.use_bridge_scoring and node_bridge_scores:
        # We can average the bridge scores or take max
        # node_bridge_scores should be pre-processed factors (e.g. 1.0 to 2.0)
        b_vals = [node_bridge_scores.get(n, 1.0) for n in path_nodes]
        bridge_term = power_mean(b_vals, cfg.power_mean_rho)

    # Affinity term aggregation
    affinity_term = 1.0
    for a in affinity_mults:
        affinity_term *= a

    # Type priors multiply
    prior_term = 1.0
    for t in type_priors:
        prior_term *= t

    # Recency multiplicative factor (clamped)
    rec_term = recency_factor(ts_list, now_ts, cfg.recency_tau_days, cfg.recency_clamp)

    return float(edge_term * ppr_term * bridge_term * affinity_term * prior_term * rec_term)


def aggregate_evidence_strength(path_scores: List[float], top_k: int = 5) -> float:
    if not path_scores:
        return 0.0
    top = sorted(path_scores, reverse=True)[: top_k]
    prod = 1.0
    for s in top:
        prod *= max(0.0, 1.0 - s)
    return 1.0 - prod


@dataclass
class InsightScoreConfig:
    alpha: float = 0.5
    beta: float = 0.2
    gamma: float = 0.2
    delta: float = 0.1


def insight_score(
    evidence_strength: float,
    community_relevance: float,
    explanation_quality: float,
    business_impact_proxy: float,
    cfg: InsightScoreConfig = InsightScoreConfig(),
) -> float:
    # Normalize inputs into [0,1]
    e = min(max(evidence_strength, 0.0), 1.0)
    c = min(max(community_relevance, 0.0), 1.0)
    x = min(max(explanation_quality, 0.0), 1.0)
    b = min(max(business_impact_proxy, 0.0), 1.0)
    w_sum = max(cfg.alpha + cfg.beta + cfg.gamma + cfg.delta, 1e-9)
    a = cfg.alpha / w_sum
    bb = cfg.beta / w_sum
    g = cfg.gamma / w_sum
    d = cfg.delta / w_sum
    return float(a * e + bb * c + g * x + d * b)


def compute_community_relevance(nodes_in_insight: Iterable[NodeId], node_ppr: Dict[NodeId, float]) -> float:
    return float(sum(node_ppr.get(n, 0.0) for n in set(nodes_in_insight)))


def score_paths_and_insight(
    accessor: GraphAccessor,
    community_id: str,
    seeds: List[NodeId],
    node_ppr_scores: List[Tuple[NodeId, float]],
    candidate_paths: List[List[Tuple[NodeId, RelId, NodeId]]],
    conf_provider: EdgeConfidenceProvider = ConstantConfidence(0.8),
    now_ts: Optional[float] = None,
    edge_timestamp_lookup=None,
    path_cfg: PathScoreConfig = PathScoreConfig(),
    insight_cfg: InsightScoreConfig = InsightScoreConfig(),
    top_k_paths: int = 5,
) -> Dict[str, object]:
    node_ppr = {n: p for n, p in node_ppr_scores}
    
    # Pre-fetch bridge & affinity data if accessor supports it
    node_bridge_scores = {}
    edge_affinity_scores = {}
    
    if path_cfg.use_bridge_scoring:
        # Duck typing check for GlobalGraphAccessor capabilities
        has_bridge = hasattr(accessor, "is_bridge")
        has_affinity = hasattr(accessor, "get_affinity") and hasattr(accessor, "get_entity_community")
        
        if has_bridge or has_affinity:
            unique_nodes = set()
            for p in candidate_paths:
                unique_nodes.add(p[0][0] if p else "") 
                for _, _, v in p:
                    unique_nodes.add(v)
            if "" in unique_nodes: unique_nodes.remove("")
            
            node_communities = {}
            for n in unique_nodes:
                if has_bridge:
                    b_data = accessor.is_bridge(n)
                    if b_data:
                        # Normalize strength: 1 + log(1+strength) * scaling
                        strength = b_data.get("bridge_strength", 0)
                        if strength > 0:
                            # e.g., strength 10 -> log(11)~2.4 -> 1.24 multiplier boost (if boost=1.0)
                            node_bridge_scores[n] = 1.0 + (math.log(1 + strength) * 0.1 * path_cfg.bridge_boost)
                
                if has_affinity:
                    comm = accessor.get_entity_community(n)
                    if comm:
                        node_communities[n] = comm
            
            if has_affinity:
                for p in candidate_paths:
                    for u, _, v in p:
                        pair = (u, v)
                        if pair not in edge_affinity_scores and (v, u) not in edge_affinity_scores:
                            c_u = node_communities.get(u)
                            c_v = node_communities.get(v)
                            if c_u and c_v and c_u != c_v:
                                aff = accessor.get_affinity(c_u, c_v)
                                edge_affinity_scores[pair] = aff

    path_scores: List[float] = []
    scored_paths: List[Dict[str, object]] = []
    for edges in candidate_paths:
        ps = path_score(
            edges, node_ppr, conf_provider, now_ts, edge_timestamp_lookup, path_cfg,
            node_bridge_scores=node_bridge_scores,
            edge_affinity_scores=edge_affinity_scores
        )
        path_scores.append(ps)
        nodes = [edges[0][0]] + [e[2] for e in edges] if edges else []
        # Decomposition terms for transparency
        ppr_vals = [node_ppr.get(n, 1e-12) for n in nodes]
        priors = path_cfg.relation_type_prior or {}
        pri_list = [max(priors.get(r, 1.0), 1e-6) for (_, r, _) in edges]
        confs = [conf_provider.confidence(u, r, v) for (u, r, v) in edges]
        recs = [edge_timestamp_lookup(u, r, v) if edge_timestamp_lookup else None for (u, r, v) in edges]
        
        # New decomposition terms
        bridge_vals = [node_bridge_scores.get(n, 1.0) for n in nodes]
        aff_vals = []
        for u, r, v in edges:
            aff = edge_affinity_scores.get((u, v), edge_affinity_scores.get((v, u), 0.0))
            aff_vals.append(aff)

        scored_paths.append({
            "score": ps,
            "nodes": nodes,
            "edges": [{"u": u, "rel": r, "v": v} for (u, r, v) in edges],
            "decomp": {
                "ppr_values": ppr_vals,
                "type_priors": pri_list,
                "edge_confidences": confs,
                "edge_timestamps": recs,
                "bridge_scores": bridge_vals,
                "affinity_scores": aff_vals,
            },
        })
    es = aggregate_evidence_strength(path_scores, top_k=top_k_paths)
    nodes_in_insight = set(n for p in scored_paths for n in p["nodes"])
    comm_rel = compute_community_relevance(nodes_in_insight, node_ppr)
    ins = insight_score(es, comm_rel, explanation_quality=0.5, business_impact_proxy=0.5, cfg=insight_cfg)
    return {
        "paths": sorted(scored_paths, key=lambda x: x["score"], reverse=True),
        "evidence_strength": es,
        "community_relevance": comm_rel,
        "insight_score": ins,
    }


