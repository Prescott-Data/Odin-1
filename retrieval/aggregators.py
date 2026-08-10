"""
Aggregation utilities for insight-ready features in Odin KG Engine.
These functions analyze retrieval paths and generate quantitative patterns
for discovery triage and briefs-ready evidence.

Drop-in safe: timezone-aware, schema-noise guards, small-n gates, and
explainable scores.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime, timezone
import statistics
import logging
import math

logger = logging.getLogger(__name__)

# ----------------------------
# Relation filtering & guards
# ----------------------------

BAN_RELATIONS = {
    "is", "object", "value", "from", "to", "object type", "object value",
    "object ID", "relationship", "related to", "page", "table", "id",
    "chunkid", "imagemetadata"
}

# Allowed only if BOTH endpoints are NOT schema crumbs (see keep_relation)
COND_RELATIONS = {"contains", "includes"}

SCHEMA_LABELS = {"page", "table", "imagemetadata", "id", "chunkid", "object"}


def keep_relation(
    rel: Optional[str],
    u_label: Optional[str] = None,
    v_label: Optional[str] = None
) -> bool:
    """
    Return True if a relation should be kept for insight aggregation.
    Enforces banlist and conditional relations logic.
    """
    if not rel:
        return False
    rel = str(rel).strip().lower()

    if rel in BAN_RELATIONS:
        return False

    if rel in COND_RELATIONS:
        # Drop if either endpoint looks like schema/meta
        if (u_label and u_label.lower() in SCHEMA_LABELS) or (v_label and v_label.lower() in SCHEMA_LABELS):
            return False

    return True


# ----------------------------
# Timestamp & math utilities
# ----------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(ts: Any) -> Optional[datetime]:
    """
    Parse various timestamp formats to timezone-aware UTC datetime.
    Supports ISO strings (with or without Z) and unix epoch (int/float).
    """
    if ts is None:
        return None
    try:
        if isinstance(ts, str):
            # Normalize trailing Z to +00:00
            iso = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception as e:
        logger.debug(f"parse_timestamp failed for {ts!r}: {e}")
    return None


def geometric_mean(values: List[float]) -> float:
    """
    Numerically stable geometric mean on [1e-6, 1.0], returns 0.0 if empty.
    """
    if not values:
        return 0.0
    clamped = [max(1e-6, min(1.0, float(v))) for v in values]
    return math.exp(sum(math.log(v) for v in clamped) / len(clamped))


# ----------------------------
# Core aggregators
# ----------------------------

def extract_motifs(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cluster paths by relation sequence (motifs) and compute statistics.

    Returns:
        List of dicts with:
          - pattern: "rel1->rel2->..."
          - edge_count: int
          - path_count: int (unique paths supporting this motif)
          - avg_edge_conf: float
          - median_recency_days: float | None
    """
    motif_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"edge_confs": [], "timestamps": [], "path_ids": set(), "edge_count": 0}
    )

    for path_idx, path in enumerate(paths):
        edges = path.get("edges", []) or []
        if not edges:
            continue

        # Build motif relation sequence (filtered)
        rel_seq: List[str] = []
        filtered_edges: List[Dict[str, Any]] = []
        for e in edges:
            rel = e.get("relation", e.get("relationship"))
            u_label = e.get("u_label")
            v_label = e.get("v_label")
            if keep_relation(rel, u_label, v_label):
                rel_seq.append(str(rel).strip())
                filtered_edges.append(e)

        if not rel_seq:
            continue

        motif_pattern = "->".join(rel_seq)
        path_id = path.get("id", f"path_{path_idx}")

        # Collect stats
        for e in filtered_edges:
            motif_stats[motif_pattern]["edge_count"] += 1
            conf = e.get("confidence", e.get("weight", 1.0))
            try:
                conf = float(conf)
            except Exception:
                conf = 1.0
            motif_stats[motif_pattern]["edge_confs"].append(conf)

            ts = e.get("created_at", e.get("timestamp"))
            dt = parse_timestamp(ts)
            if dt:
                motif_stats[motif_pattern]["timestamps"].append(dt)

        motif_stats[motif_pattern]["path_ids"].add(path_id)

    motifs: List[Dict[str, Any]] = []
    now = _now_utc()

    for pattern, stats in motif_stats.items():
        # Median recency (days)
        rec_days: Optional[float] = None
        if stats["timestamps"]:
            try:
                days_ago = [max(0.0, (now - dt).total_seconds() / 86400.0) for dt in stats["timestamps"]]
                if days_ago:
                    rec_days = statistics.median(days_ago)
            except Exception as e:
                logger.warning(f"Error calculating recency for motif {pattern}: {e}")

        motifs.append(
            {
                "pattern": pattern,
                "edge_count": int(stats["edge_count"]),
                "path_count": int(len(stats["path_ids"])),
                "avg_edge_conf": round(statistics.mean(stats["edge_confs"]), 4) if stats["edge_confs"] else 0.0,
                "median_recency_days": rec_days,
            }
        )

    motifs.sort(key=lambda x: (x["edge_count"], x["path_count"]), reverse=True)
    return motifs


def calculate_relation_shares(paths: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate the distribution of relations across all edges in paths (filtered).

    Returns:
        Dict[relation] = { "count": int, "share": float }
    """
    relation_counts: Counter = Counter()
    total_edges = 0

    for path in paths:
        for e in path.get("edges", []) or []:
            rel = e.get("relation", e.get("relationship"))
            u_label = e.get("u_label")
            v_label = e.get("v_label")
            if not keep_relation(rel, u_label, v_label):
                continue
            relation_counts[str(rel).strip()] += 1
            total_edges += 1

    # Avoid division by zero
    denom = max(1, total_edges)
    shares = {rel: {"count": cnt, "share": round(cnt / denom, 4)} for rel, cnt in relation_counts.items()}
    return shares


def compute_provenance_coverage(paths: List[Dict[str, Any]]) -> float:
    """
    Fraction of edges that carry any provenance/document id.
    """
    prov_edges = 0
    total = 0
    for p in paths:
        for e in p.get("edges", []) or []:
            total += 1
            prov = e.get("provenance")
            src = e.get("source_doc")
            has = False
            if isinstance(prov, dict):
                has = bool(prov.get("document_id"))
            elif isinstance(prov, list):
                # any item with document_id
                has = any(bool(it.get("document_id")) for it in prov if isinstance(it, dict))
            has = has or bool(src)
            prov_edges += int(has)
    return (prov_edges / total) if total else 0.0


def compute_recency_score(paths: List[Dict[str, Any]], half_life_days: float = 30.0) -> float:
    """
    Recency score in [0,1] as average of exp(-age/half_life) over edges with timestamps.
    """
    if half_life_days <= 0:
        return 0.0
    now = _now_utc()
    vals: List[float] = []
    for p in paths:
        for e in p.get("edges", []) or []:
            ts = e.get("created_at", e.get("timestamp"))
            dt = parse_timestamp(ts)
            if not dt:
                continue
            age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
            vals.append(math.exp(-age_days / half_life_days))
    return statistics.mean(vals) if vals else 0.0


def estimate_label_coverage(paths: List[Dict[str, Any]]) -> float:
    """
    Estimate fraction of labeled endpoints across edges (0..1).
    """
    labeled = 0
    total = 0
    for p in paths:
        for e in p.get("edges", []) or []:
            total += 2
            labeled += int(bool(e.get("u_label")))
            labeled += int(bool(e.get("v_label")))
    return (labeled / total) if total else 0.0


def compute_motif_density(motifs: List[Dict[str, Any]], topk: int = 2) -> float:
    """
    Share of edges that fall into the top-K motifs (by edge_count).
    """
    if not motifs:
        return 0.0
    counts = [int(m.get("edge_count", m.get("count", 0))) for m in motifs]
    counts.sort(reverse=True)
    top = sum(counts[:topk])
    total = sum(counts)
    return (top / total) if total else 0.0


def extract_snippet_anchors(
    paths: List[Dict[str, Any]],
    max_per_path: int = 2,
    max_total: int = 16
) -> List[Dict[str, Any]]:
    """
    Extract provenance anchors from top paths, preferring doc diversity.

    Returns:
        List of { path_idx, edge_id, document_id, span? }
    """
    anchors: List[Dict[str, Any]] = []
    seen_docs = set()

    for path_idx, path in enumerate(paths[:10]):  # focus on top-10 paths
        if len(anchors) >= max_total:
            break
        edges = path.get("edges", []) or []
        path_local: List[Dict[str, Any]] = []

        for e in edges:
            # Skip edges without allowed relations to keep anchors on-semantic
            rel = e.get("relation", e.get("relationship"))
            if not keep_relation(rel, e.get("u_label"), e.get("v_label")):
                continue

            prov = e.get("provenance")
            doc_id = None
            span = None

            if isinstance(prov, dict):
                doc_id = prov.get("document_id")
                span = prov.get("char_span")
            elif isinstance(prov, list) and prov:
                # take first with document_id
                for it in prov:
                    if isinstance(it, dict) and it.get("document_id"):
                        doc_id = it["document_id"]
                        span = it.get("char_span")
                        break

            if not doc_id:
                doc_id = e.get("source_doc")

            if doc_id and doc_id not in seen_docs:
                anchor = {
                    "path_idx": path_idx,
                    "edge_id": e.get("_id", e.get("id")),
                    "document_id": doc_id,
                }
                if span is not None:
                    anchor["span"] = span
                path_local.append(anchor)

            if len(path_local) >= max_per_path or len(anchors) + len(path_local) >= max_total:
                break

        # prefer per-path diversity
        anchors.extend(path_local[:max_per_path])

    return anchors


def surprise_vs_priors(relation_shares: Dict[str, Dict[str, Any]], priors: Dict[str, float]) -> Dict[str, float]:
    """
    Absolute deviation of observed share vs prior for each relation.
    """
    s: Dict[str, float] = {}
    priors = priors or {}
    for rel, stats in relation_shares.items():
        obs = float(stats.get("share", 0.0))
        prior = float(priors.get(rel, 0.0))
        s[rel] = abs(obs - prior)
    return s


def compute_baseline_comparison(
    current_aggregates: Dict[str, Any],
    baseline_aggregates: Optional[Dict[str, Any]] = None,
    min_baseline_edges: int = 10
) -> Dict[str, Any]:
    """
    Compare current aggregates with baseline to compute deltas & relative changes.
    Adds note if baseline support is low.
    """
    if not baseline_aggregates:
        return {"surprise": {}, "deltas": {}, "relative_changes": {}, "note": "no_baseline"}

    comparison = {"surprise": {}, "deltas": {}, "relative_changes": {}, "pp_change": {}}

    cur_shares = current_aggregates.get("relation_share", {})
    base_shares = baseline_aggregates.get("relation_share", {})

    base_total_edges = sum(v.get("count", 0) for v in base_shares.values())
    if base_total_edges < min_baseline_edges:
        comparison["note"] = "baseline_low_support"

    for rel in set(cur_shares.keys()) | set(base_shares.keys()):
        c = cur_shares.get(rel, {})
        b = base_shares.get(rel, {})
        c_share = float(c.get("share", 0.0))
        b_share = float(b.get("share", 0.0))
        delta = c_share - b_share
        comparison["surprise"][rel] = round(delta, 4)
        comparison["pp_change"][rel] = round(100.0 * delta, 2)
        if b_share > 0:
            comparison["relative_changes"][rel] = round((c_share - b_share) / b_share, 4)

    # Motif deltas (edge_count-based)
    cur_motifs = {m["pattern"]: int(m.get("edge_count", m.get("count", 0))) for m in current_aggregates.get("motifs", [])}
    base_motifs = {m["pattern"]: int(m.get("edge_count", m.get("count", 0))) for m in baseline_aggregates.get("motifs", [])}
    motif_deltas = {}
    for pattern in set(cur_motifs) | set(base_motifs):
        motif_deltas[pattern] = cur_motifs.get(pattern, 0) - base_motifs.get(pattern, 0)
    comparison["deltas"]["motifs"] = motif_deltas

    return comparison


def generate_aggregates(
    paths: List[Dict[str, Any]],
    baseline_paths: Optional[List[Dict[str, Any]]] = None,
    *,
    priors: Optional[Dict[str, float]] = None,
    half_life_days: float = 30.0,
    topk_motif_density: int = 2,
    min_current_edges: int = 20,
    min_baseline_edges: int = 10
) -> Dict[str, Any]:
    """
    Generate comprehensive aggregates for insight generation & discovery triage.

    Returns a dict with:
      - motifs, relation_share, snippet_anchors
      - baseline (if provided) and comparison
      - summary: totals, motif_density, label_coverage, provenance, recency, low_support
      - surprise_priors (if priors provided)
      - dominant_relation snapshot (share, prior, surprise)
    """
    motifs = extract_motifs(paths)
    relation_share = calculate_relation_shares(paths)
    anchors = extract_snippet_anchors(paths)

    # Summary metrics
    total_edges = sum(v["count"] for v in relation_share.values())
    prov = compute_provenance_coverage(paths)
    recency = compute_recency_score(paths, half_life_days=half_life_days)
    label_cov = estimate_label_coverage(paths)
    motif_dens = compute_motif_density(motifs, topk=topk_motif_density)

    aggregates: Dict[str, Any] = {
        "motifs": motifs,
        "relation_share": relation_share,
        "snippet_anchors": anchors,
        "summary": {
            "total_paths": len(paths),
            "unique_motifs": len(motifs),
            "unique_relations": len(relation_share),
            "total_edges": total_edges,
            "provenance": round(prov, 4),
            "recency": round(recency, 4),
            "label_coverage": round(label_cov, 4),
            "motif_density": round(motif_dens, 4),
            "has_baseline": baseline_paths is not None,
            "low_support": total_edges < min_current_edges
        }
    }

    # Baseline comparison (optional)
    if baseline_paths is not None:
        base_agg = {
            "motifs": extract_motifs(baseline_paths),
            "relation_share": calculate_relation_shares(baseline_paths)
        }
        aggregates["baseline"] = base_agg
        aggregates["comparison"] = compute_baseline_comparison(
            {"motifs": motifs, "relation_share": relation_share},
            base_agg,
            min_baseline_edges=min_baseline_edges
        )

    # Priors surprise (optional)
    if priors:
        s = surprise_vs_priors(relation_share, priors)
        aggregates["surprise_priors"] = {k: round(v, 4) for k, v in s.items()}

    # Dominant relation snapshot
    if relation_share:
        dominant_rel = max(relation_share.items(), key=lambda kv: kv[1]["share"])[0]
        dom = {"relation": dominant_rel, "share": relation_share[dominant_rel]["share"]}
        if priors:
            dom["prior"] = float(priors.get(dominant_rel, 0.0))
            dom["surprise_vs_prior"] = round(abs(dom["share"] - dom["prior"]), 4)
        if "comparison" in aggregates:
            dom["delta_vs_baseline"] = aggregates["comparison"]["surprise"].get(dominant_rel, 0.0)
        aggregates["dominant_relation"] = dom

    return aggregates


# ----------------------------
# Insight & triage scoring
# ----------------------------

def decompose_insight_score(
    paths: List[Dict[str, Any]],
    *,
    evidence_strength: float,
    community_relevance: float,
    insight_score: float
) -> Dict[str, Any]:
    """
    Decompose an insight score into drivers.

    Returns:
        {
          "value": float, "label": "High|Medium|Low",
          "drivers": {...},
          "quality_gate": {...}
        }
    """
    # Path-level strengths
    path_geo_confs: List[float] = []
    edge_confs_all: List[float] = []
    ppr_scores: List[float] = []

    for path in paths:
        edges = path.get("edges", []) or []
        e_confs = []
        for e in edges:
            c = e.get("confidence", e.get("weight", 1.0))
            try:
                e_confs.append(float(c))
                edge_confs_all.append(float(c))
            except Exception:
                edge_confs_all.append(1.0)
        if e_confs:
            path_geo_confs.append(geometric_mean(e_confs))

        if "ppr_score" in path:
            try:
                ppr_scores.append(float(path["ppr_score"]))
            except Exception:
                pass

    drivers = {
        "path_strength": round(statistics.mean(path_geo_confs), 3) if path_geo_confs else 0.0,
        "ppr_strength": round(statistics.mean(ppr_scores), 3) if ppr_scores else 0.0,
        "edge_conf_strength": round(statistics.mean(edge_confs_all), 3) if edge_confs_all else 0.0,
        "evidence_strength_f": round(float(evidence_strength), 3),
        "community_relevance_f": round(float(community_relevance), 3),
        "insight_score_f": round(float(insight_score), 3)
    }

    if insight_score >= 0.7:
        label = "High"
    elif insight_score >= 0.4:
        label = "Medium"
    else:
        label = "Low"

    quality_gate = {
        "meets_evidence_floor": evidence_strength >= 0.5,
        "has_strong_paths": drivers["path_strength"] >= 0.4,
        "recommendation": "proceed" if insight_score >= 0.4 else "gather_more_evidence"
    }

    return {"value": round(insight_score, 2), "label": label, "drivers": drivers, "quality_gate": quality_gate}


def compute_triage_score(
    *,
    provenance: float,
    recency: float,
    surprise: float,
    motif_density: float,
    controllability: float,
    label_coverage: Optional[float] = None,
    low_support: bool = False
) -> Tuple[int, Dict[str, float]]:
    """
    Compute triage score per contract:
      score = 25*prov + 25*rec + 25*surprise + 15*motif + 10*control
    All inputs expected in [0,1]. Returns (score_int_0_100, components_dict).

    Guards:
      - If label_coverage < 0.8, cap motif_density at 0.3 and subtract 15 points.
      - If low_support, subtract 40% of the score.
    """
    # Coerce bounds
    clamp = lambda x: max(0.0, min(1.0, float(x)))
    provenance = clamp(provenance)
    recency = clamp(recency)
    surprise = clamp(surprise)
    motif_density = clamp(motif_density)
    controllability = clamp(controllability)
    label_coverage = clamp(label_coverage) if label_coverage is not None else None

    penalty = 0.0
    if label_coverage is not None and label_coverage < 0.8:
        motif_density = min(motif_density, 0.3)
        penalty += 15.0

    base_score = (
        25.0 * provenance
        + 25.0 * recency
        + 25.0 * surprise
        + 15.0 * motif_density
        + 10.0 * controllability
    )

    if low_support:
        base_score *= 0.6  # subtract 40%

    score = max(0.0, min(100.0, base_score - penalty))
    components = {
        "provenance": round(provenance, 4),
        "recency": round(recency, 4),
        "surprise": round(surprise, 4),
        "motif_density": round(motif_density, 4),
        "controllability": round(controllability, 4),
        "label_coverage": round(label_coverage, 4) if label_coverage is not None else None,
        "penalty": round(penalty, 2),
        "low_support": bool(low_support)
    }
    return int(round(score)), components


# ----------------------------
# Convenience: one-shot features for Discovery
# ----------------------------

def build_opportunity_features(
    paths: List[Dict[str, Any]],
    baseline_paths: Optional[List[Dict[str, Any]]] = None,
    *,
    priors: Optional[Dict[str, float]] = None,
    half_life_days: float = 30.0,
    controllability: float = 1.0,
    min_current_edges: int = 20,
    min_baseline_edges: int = 10
) -> Dict[str, Any]:
    """
    Convenience wrapper to produce everything Discovery needs to compute triage.

    Returns dict:
      {
        "aggregates": { ... },               # from generate_aggregates
        "triage": {
           "score": 0-100,
           "components": {...},
           "dominant_relation": {...}
        }
      }
    """
    aggs = generate_aggregates(
        paths,
        baseline_paths,
        priors=priors,
        half_life_days=half_life_days,
        min_current_edges=min_current_edges,
        min_baseline_edges=min_baseline_edges,
    )

    # Surprise component selection:
    # Prefer priors-based surprise on dominant relation; fall back to baseline delta abs().
    surprise_component = 0.0
    dom = aggs.get("dominant_relation") or {}
    if priors and "surprise_vs_prior" in dom:
        surprise_component = abs(float(dom.get("surprise_vs_prior", 0.0)))
    elif "comparison" in aggs and "delta_vs_baseline" in dom:
        surprise_component = abs(float(dom.get("delta_vs_baseline", 0.0)))

    # Compute triage
    summary = aggs["summary"]
    score, comps = compute_triage_score(
        provenance=summary["provenance"],
        recency=summary["recency"],
        surprise=surprise_component,
        motif_density=summary["motif_density"],
        controllability=controllability,
        label_coverage=summary["label_coverage"],
        low_support=summary["low_support"]
    )

    triage = {
        "score": score,
        "components": comps,
        "dominant_relation": dom
    }

    return {"aggregates": aggs, "triage": triage}


# ----------------------------
# Optional: basic self-test (remove in prod if undesired)
# ----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Minimal synthetic example
    paths_example = [
        {
            "id": "p1",
            "edges": [
                {"u_label": "ClaimForm", "v_label": "Document", "relation": "supporting_documentation",
                 "confidence": 0.9, "created_at": "2025-07-20T12:00:00Z", "provenance": {"document_id": "docA"}},
                {"u_label": "Document", "v_label": "OfficerSignature", "relation": "signed_by",
                 "confidence": 0.95, "created_at": "2025-07-21T12:00:00Z", "provenance": {"document_id": "docB"}},
            ],
            "ppr_score": 0.12
        },
        {
            "id": "p2",
            "edges": [
                {"u_label": "AssessmentReport", "v_label": "Document", "relation": "attached",
                 "confidence": 0.85, "created_at": "2025-07-22T15:30:00Z", "provenance": {"document_id": "docC"}},
            ],
            "ppr_score": 0.08
        }
    ]

    priors_example = {"supporting_documentation": 0.30, "signed_by": 0.10, "attached": 0.20}

    out = build_opportunity_features(paths_example, priors=priors_example, controllability=1.0)
    from pprint import pprint
    pprint(out)
