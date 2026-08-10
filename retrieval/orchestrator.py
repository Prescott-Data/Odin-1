from __future__ import annotations
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Optional, Callable, Any
from collections import defaultdict

from .adapters import GraphAccessor, NodeId
from .budget import SearchBudget, BudgetTracker
from .ppr.engines import PushPPREngine, MonteCarloPPREngine, PPRParams
from .ppr.bippr import BiPPREngine
from .ppr.indexes import RandomWalkIndex
from .beam import beam_search, BeamParams
from .confidence import EdgeConfidenceProvider, ConstantConfidence, NPLLConfidence
from .ppr_cache import PPRCache, _key as ppr_cache_key
from .scoring import (
    PathScoreConfig,
    InsightScoreConfig,
    score_paths_and_insight,
)
from .metrics import Timer, MetricsLogger, RetrievalMetrics
from .adapters import OverlayAccessor
from .linker import CoherenceLinker, LinkerConfig, Mention
from .utils.pii_redaction import redact_dict
from .writers.base import PersistenceWriter

# NEW: triage-ready aggregators (with guards & priors surprise)
from .aggregators import build_opportunity_features, decompose_insight_score


@dataclass
class OrchestratorParams:
    community_id: str
    # PPR
    alpha: float = 0.15
    eps: float = 1e-4
    num_walks: int = 5000
    walk_len: int = 40
    topn: int = 200
    # Beam/paths
    hop_limit: int = 3
    beam_width: int = 64
    max_paths: int = 200
    # Scoring
    path_cfg: PathScoreConfig = field(default_factory=PathScoreConfig)
    insight_cfg: InsightScoreConfig = field(default_factory=InsightScoreConfig)


class RetrievalOrchestrator:
    """
    Orchestrates:
      1) Mixed PPR (push + MC) with optional anchor personalization & cache
      2) Beam path enumeration with edge confidences (NPLL if configured)
      3) Path & insight scoring
      4) Aggregation → motifs / relation shares / priors surprise / anchors
      5) Optional baseline (shifted time window) for deltas
      6) Triage score + ICS decomposition for explainability

    Quality guards:
      - Requires NPLL if ODIN_REQUIRE_NPLL=true
      - Normalizes raw path edge schema for aggregators
      - Suppresses low-support metrics automatically
      - Optional single-step dynamic scaling when support is low
    """

    def __init__(
        self,
        accessor: GraphAccessor,
        ppr_cache: Optional[PPRCache] = None,
        edge_confidence: Optional[EdgeConfidenceProvider] = None,
        edge_timestamp_lookup: Optional[Callable[[NodeId, str, NodeId], Optional[float]]] = None,
        walk_index: Optional[RandomWalkIndex] = None,
        redact_pii_in_trace: bool = True,
    ):
        self.A = accessor
        self.cache = ppr_cache or PPRCache()
        self.conf = edge_confidence or ConstantConfidence(0.8)

        # Enforce NPLL when requested by env
        if os.getenv("ODIN_REQUIRE_NPLL", "false").lower() in ("true", "1", "yes"):
            if not isinstance(self.conf, NPLLConfidence):
                raise RuntimeError(
                    "ODIN_REQUIRE_NPLL is enabled but edge_confidence is not NPLLConfidence. "
                    "Provide NPLLConfidence(npll_model)."
                )

        self.ts_lookup = edge_timestamp_lookup
        self.walk_index = walk_index
        self.redact_pii_in_trace = redact_pii_in_trace

    # ----------------------------
    # Public API
    # ----------------------------

    def retrieve(
        self,
        seeds: List[NodeId],
        params: OrchestratorParams,
        budget: Optional[SearchBudget] = None,
        now_ts: Optional[float] = None,
        include_baseline: bool = False,
        anchor_prior: Optional[Dict[NodeId, float]] = None,
        beam_override: Optional[BeamParams] = None,
        dynamic_scale: bool = True,
        min_paths_for_confidence: int = 8,
    ) -> Dict[str, object]:
        """
        Core retrieval + aggregation + triage pipeline.
        Optionally performs a single dynamic scale-up pass if support is low.
        """
        primary = self._retrieve_once(
            seeds=seeds,
            params=params,
            budget=budget,
            now_ts=now_ts,
            include_baseline=include_baseline,
            anchor_prior=anchor_prior,
            beam_override=beam_override,
        )

        # Single-step dynamic expansion if needed (low support or no paths)
        need_scale = dynamic_scale and (
            len(primary.get("paths", [])) < min_paths_for_confidence
            or (primary.get("aggregates", {}).get("summary", {}).get("low_support") is True)
        )

        if not need_scale:
            return primary

        # Expand hop/beam a notch, reuse same PPR
        scaled_params = replace(
            params,
            hop_limit=min(params.hop_limit + 1, 4),
            beam_width=min(params.beam_width * 2, 256),
            max_paths=min(params.max_paths * 2, 400),
        )

        scaled = self._retrieve_once(
            seeds=seeds,
            params=scaled_params,
            budget=budget,
            now_ts=now_ts,
            include_baseline=include_baseline,
            anchor_prior=anchor_prior,
            beam_override=beam_override,
        )

        # Choose better result by (insight_score, then #paths)
        def _key(r: Dict[str, Any]) -> Tuple[float, int]:
            return (float(r.get("insight_score", 0.0)), len(r.get("paths", [])))

        return scaled if _key(scaled) > _key(primary) else primary

    def score_candidates(
        self,
        source: NodeId,
        targets: List[NodeId],
        params: OrchestratorParams,
        budget: Optional[SearchBudget] = None,
    ) -> Dict[str, object]:
        """
        Efficiently score a given candidate set via BiPPR.
        """
        bt = BudgetTracker(budget or SearchBudget())
        bippr = BiPPREngine(graph=self.A, alpha=params.alpha, rmax=params.eps)
        walks = max(1000, min(params.num_walks, bt.left().max_edges))
        scores = bippr.score(source=source, targets=targets, walks=walks)
        return {
            "candidate_scores": scores,
            "used_budget": bt.usage.__dict__,
            "trace": {"engine": "bippr", "walks": walks},
        }

    def link_and_retrieve(
        self,
        mentions: List[Mention],
        base_accessor: GraphAccessor,
        params: OrchestratorParams,
        seeds_from_linked: int = 3,
        budget: Optional[SearchBudget] = None,
        now_ts: Optional[float] = None,
        persistence_writer: Optional[PersistenceWriter] = None,
    ) -> Dict[str, object]:
        linker_cfg = getattr(self, "linker_cfg", LinkerConfig())
        linker = CoherenceLinker(linker_cfg)
        linked = linker.link(mentions)

        overlay = OverlayAccessor(base_accessor, params.community_id)
        seeds: List[NodeId] = []
        for m in mentions:
            if m.mention_id in linked:
                ent = linked[m.mention_id]["entity_id"]
                conf = linked[m.mention_id]["link_confidence"]
                overlay.add_edge(m.surface, "mentions", ent, conf)
                seeds.append(ent)
        seeds = list(dict.fromkeys(seeds))[: seeds_from_linked]

        prev_A = self.A
        try:
            self.A = overlay
            res = self.retrieve(seeds=seeds, params=params, budget=budget, now_ts=now_ts)
            res["trace"]["linked_entities"] = linked

            if persistence_writer is not None:
                for m in mentions:
                    if m.mention_id in linked:
                        ent = linked[m.mention_id]["entity_id"]
                        conf = linked[m.mention_id]["link_confidence"]
                        try:
                            persistence_writer.maybe_write_link(
                                src_entity=ent,
                                rel="linked_from_text",
                                dst_entity=ent,
                                confidence=conf,
                                metadata={"mention_id": m.mention_id},
                            )
                        except Exception:
                            pass

            return res
        finally:
            self.A = prev_A

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _retrieve_once(
        self,
        seeds: List[NodeId],
        params: OrchestratorParams,
        budget: Optional[SearchBudget],
        now_ts: Optional[float],
        include_baseline: bool,
        anchor_prior: Optional[Dict[NodeId, float]],
        beam_override: Optional[BeamParams],
    ) -> Dict[str, object]:
        bt = BudgetTracker(budget or SearchBudget(max_paths=params.max_paths))
        tm = Timer()

        # Personalization vector (seeds + optional anchors)
        personalization: Dict[NodeId, float] = defaultdict(float)
        for s in seeds:
            personalization[s] += 1.0
        if anchor_prior:
            for n, p in anchor_prior.items():
                personalization[n] += float(p)

        total_mass = sum(personalization.values())
        if total_mass > 0:
            personalization = {n: v / total_mass for n, v in personalization.items()}
        else:
            # Fallback: uniform over community if no seeds/anchors
            all_nodes = list(self.A.nodes(params.community_id))
            if not all_nodes:
                return self._empty_result(bt, tm)
            uniform = 1.0 / len(all_nodes)
            personalization = {n: uniform for n in all_nodes}

        # ---- PPR (with cache) ----
        prior_hash = str(hash(frozenset(anchor_prior.items()))) if anchor_prior else ""
        key = ppr_cache_key(params.community_id, seeds, params.alpha, engine="push+mc", prior_hash=prior_hash)
        cached = self.cache.get(key)

        if cached is not None:
            ppr_scores = cached
            ppr_trace = {"engine": "cache", "cache_hit": True}
            tm.mark("ppr_done")
        else:
            push = PushPPREngine(self.A, params.community_id)
            mc = MonteCarloPPREngine(self.A, params.community_id, walk_index=self.walk_index)
            pp = PPRParams(alpha=params.alpha, eps=params.eps, num_walks=0, walk_len=0, topn=params.topn)
            mm = PPRParams(alpha=params.alpha, eps=params.eps, num_walks=params.num_walks, walk_len=params.walk_len, topn=params.topn)

            pr_push = push.run(list(personalization.keys()), pp, bt.left(), personalization=personalization)
            pr_mc = mc.run(list(personalization.keys()), mm, bt.left(), personalization=personalization)

            push_mass = max(pr_push.mass, 1e-12)
            resid = max(1.0 - push_mass, 0.0)

            combined: Dict[NodeId, float] = {}
            for n, p in pr_push.scores:
                combined[n] = combined.get(n, 0.0) + p
            for n, p in pr_mc.scores:
                combined[n] = combined.get(n, 0.0) + p * resid

            Z = sum(combined.values()) or 1.0
            ppr_scores = sorted(((n, v / Z) for n, v in combined.items()), key=lambda kv: kv[1], reverse=True)[: params.topn]
            ppr_trace = {"engine": "push+mc", "push": pr_push.trace, "mc": pr_mc.trace, "cache_hit": False}
            self.cache.put(key, ppr_scores)
            tm.mark("ppr_done")

        # ---- Beam search (paths) ----
        beam = beam_override or BeamParams(hop_limit=params.hop_limit, beam_width=params.beam_width, max_paths=params.max_paths)
        path_out = beam_search(
            accessor=self.A,
            community_id=params.community_id,
            seeds=seeds,
            ppr_scores=ppr_scores,
            budget=bt.left(),
            beam_params=beam,
            conf_provider=self.conf,
        )
        tm.mark("beam_done")

        # ---- Scoring ----
        candidate_paths = [[(e["u"], e["rel"], e["v"]) for e in p.get("edges", [])] for p in path_out.get("paths", [])]
        scored = score_paths_and_insight(
            accessor=self.A,
            community_id=params.community_id,
            seeds=seeds,
            node_ppr_scores=ppr_scores,
            candidate_paths=candidate_paths,
            conf_provider=self.conf,
            now_ts=now_ts,
            edge_timestamp_lookup=self.ts_lookup,
            path_cfg=params.path_cfg,
            insight_cfg=params.insight_cfg,
        )
        tm.mark("scoring_done")

        # ---- Normalize edges for aggregators (add relation/labels/ts/provenance keys) ----
        paths_norm = self._normalize_paths_for_aggregators(scored.get("paths", []))

        # ---- Baseline retrieval (optional; shifted window) ----
        baseline_paths_norm: Optional[List[Dict[str, Any]]] = None
        if include_baseline and getattr(self.A, "time_window", None):
            baseline_paths_norm = self._get_shifted_baseline_paths(
                seeds=seeds,
                params=params,
                scorer_output_like=scored,
            )

        # ---- Priors (if accessor exposes them) ----
        try:
            priors: Dict[str, float] = getattr(self.A, "get_edge_type_priors")(params.community_id)  # type: ignore
        except Exception:
            priors = {}

        # ---- Half-life (~1.5x window length clamped 14–60) ----
        half_life_days = self._half_life_from_accessor_window(default=30.0)

        # ---- Aggregates + triage (includes motif density, priors surprise, anchors, low-support flags) ----
        features = build_opportunity_features(
            paths=paths_norm,
            baseline_paths=baseline_paths_norm,
            priors=priors,
            half_life_days=half_life_days,
            controllability=1.0,  # default; Discovery can override per-op
        )

        # ---- Insight score decomposition for explainability (uses scored metrics) ----
        ics_decomposed = decompose_insight_score(
            paths=paths_norm,
            evidence_strength=float(scored.get("evidence_strength", 0.0)),
            community_relevance=float(scored.get("community_relevance", 0.0)),
            insight_score=float(scored.get("insight_score", 0.0)),
        )

        result = {
            "topk_ppr": ppr_scores,
            "paths": paths_norm,  # normalized, aggregator-friendly
            "evidence_strength": scored.get("evidence_strength", 0.0),
            "community_relevance": scored.get("community_relevance", 0.0),
            "insight_score": scored.get("insight_score", 0.0),
            "aggregates": features["aggregates"],
            "triage": features["triage"],
            "ics": ics_decomposed,
            "used_budget": bt.usage.__dict__ | {"beam": path_out.get("used_budget", {})},
            "trace": {
                "ppr": ppr_trace,
                "beam": path_out.get("trace", {}),
                "params": params,
                "timings_ms": tm.marks | {"total": tm.elapsed_ms()},
                "linker_cfg": getattr(self, "linker_cfg", None),
            },
        }

        # Optional external metrics logger
        if hasattr(self, "metrics_logger") and isinstance(self.metrics_logger, MetricsLogger):
            esr = path_out.get("trace", {}).get("early_stop_reason")
            self.metrics_logger.log(
                RetrievalMetrics(
                    query_id=None,
                    community_id=params.community_id,
                    seeds_count=len(seeds),
                    ppr_mass=sum(p for _, p in ppr_scores),
                    topk=len(ppr_scores),
                    used_budget=result["used_budget"],
                    latency_ms=tm.elapsed_ms(),
                    early_stop_reason=esr,
                    engine=ppr_trace.get("engine", "unknown"),
                    notes={"timings": tm.marks},
                )
            )

        # PII redaction for trace
        if self.redact_pii_in_trace:
            result["trace"] = redact_dict(result["trace"])

        return result

    # ---- schema normalization & utilities ----

    def _normalize_paths_for_aggregators(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure each edge has aggregator-friendly fields:
          relation, u_label, v_label, confidence, created_at/timestamp, provenance{document_id?}
        """
        out: List[Dict[str, Any]] = []

        def _label(n: Any) -> Optional[str]:
            try:
                # Accessor MAY expose label(...) or node_label(...)
                if hasattr(self.A, "label"):
                    return self.A.label(n)  # type: ignore
                if hasattr(self.A, "node_label"):
                    return self.A.node_label(n)  # type: ignore
            except Exception:
                pass
            return None

        for p in paths:
            norm_edges: List[Dict[str, Any]] = []
            for e in p.get("edges", []):
                u = e.get("u")
                v = e.get("v")
                rel = e.get("rel") or e.get("relation") or e.get("relationship")

                # Timestamp: prefer provided; else ts_lookup fallback (seconds)
                ts = e.get("created_at", e.get("timestamp"))
                if ts is None and self.ts_lookup and (u is not None and v is not None and rel is not None):
                    try:
                        ts_lookup_val = self.ts_lookup(u, str(rel), v)
                        ts = ts_lookup_val
                    except Exception:
                        ts = None

                # Provenance
                prov = e.get("provenance")
                if prov is None:
                    # Common alternates
                    doc_id = e.get("source_doc") or e.get("doc_id")
                    if doc_id:
                        prov = {"document_id": doc_id}

                norm_edges.append(
                    {
                        "u": u,
                        "v": v,
                        "relation": rel,
                        "u_label": e.get("u_label") or _label(u),
                        "v_label": e.get("v_label") or _label(v),
                        "confidence": e.get("confidence", e.get("weight", 1.0)),
                        "created_at": ts,  # ISO or epoch; aggregator can parse both
                        "provenance": prov,
                        "_id": e.get("_id", e.get("id")),
                    }
                )

            out.append({"id": p.get("id"), "score": p.get("score"), "edges": norm_edges})

        return out

    def _get_shifted_baseline_paths(
        self,
        seeds: List[NodeId],
        params: OrchestratorParams,
        scorer_output_like: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Run a light baseline retrieval by shifting self.A.time_window back by its duration.
        Returns normalized paths (aggregator schema).
        """
        original_window = getattr(self.A, "time_window", None)
        if not original_window:
            return []

        start_ts, end_ts = original_window
        start_f = self._coerce_ts(start_ts)
        end_f = self._coerce_ts(end_ts)
        if start_f is None or end_f is None or end_f <= start_f:
            return []

        duration = end_f - start_f
        baseline_window = (max(0.0, start_f - duration), start_f)

        # Temporarily swap window
        self.A.time_window = baseline_window  # type: ignore[attr-defined]
        try:
            baseline_budget = SearchBudget(max_paths=max(20, params.max_paths // 10), max_edges=1000)
            # Slim params for baseline
            base_params = replace(params, max_paths=baseline_budget.max_paths, beam_width=max(16, params.beam_width // 2))
            baseline_res = self._retrieve_once(
                seeds=seeds,
                params=base_params,
                budget=baseline_budget,
                now_ts=None,
                include_baseline=False,
                anchor_prior=None,
                beam_override=None,
            )
            return baseline_res.get("paths", [])
        finally:
            self.A.time_window = original_window  # type: ignore[attr-defined]

    def _half_life_from_accessor_window(self, default: float = 30.0) -> float:
        """
        ~1.5× window length in days, clamped to [14, 60].
        Falls back to `default` if no valid window on accessor.
        """
        w = getattr(self.A, "time_window", None)
        if not w:
            return default
        s, e = w
        s_f = self._coerce_ts(s)
        e_f = self._coerce_ts(e)
        if s_f is None or e_f is None or e_f <= s_f:
            return default
        days = (e_f - s_f) / 86400.0
        hl = max(14.0, min(60.0, 1.5 * days))
        return hl

    @staticmethod
    def _coerce_ts(ts: Any) -> Optional[float]:
        """
        Coerce ISO string or numeric epoch into float epoch seconds.
        """
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            try:
                # Allow trailing Z
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.timestamp()
            except Exception:
                return None
        return None

    @staticmethod
    def _empty_result(bt: BudgetTracker, tm: Timer) -> Dict[str, object]:
        return {
            "topk_ppr": [],
            "paths": [],
            "evidence_strength": 0.0,
            "community_relevance": 0.0,
            "insight_score": 0.0,
            "aggregates": {
                "motifs": [],
                "relation_share": {},
                "snippet_anchors": [],
                "summary": {
                    "total_paths": 0,
                    "unique_motifs": 0,
                    "unique_relations": 0,
                    "total_edges": 0,
                    "provenance": 0.0,
                    "recency": 0.0,
                    "label_coverage": 0.0,
                    "motif_density": 0.0,
                    "has_baseline": False,
                    "low_support": True,
                },
            },
            "triage": {"score": 0, "components": {}, "dominant_relation": {}},
            "ics": {"value": 0.0, "label": "Low", "drivers": {}, "quality_gate": {"meets_evidence_floor": False, "has_strong_paths": False, "recommendation": "gather_more_evidence"}},
            "used_budget": bt.usage.__dict__,
            "trace": {"timings_ms": tm.marks | {"total": tm.elapsed_ms()}},
        }
