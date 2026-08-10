from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from .adapters import NodeId


@dataclass
class LinkerConfig:
    candidates_per_mention: int = 10
    coherence_iterations: int = 1
    persist_threshold: float = 0.8
    w_candidate: float = 0.6
    w_prior: float = 0.3
    w_coherence: float = 0.1


@dataclass
class Mention:
    mention_id: str
    surface: str
    normalized: Optional[str]
    span: Tuple[int, int]
    context: Optional[str]
    llm_confidence: float
    candidates: List[Tuple[NodeId, float]]  # (entity_id, candidate_score)


class CoherenceLinker:
    """
    Skeleton linker that accepts LLM mentions with candidates and returns linked entities.
    Coherence/ranking by graph priors (to be plugged-in): use PPR/anchors in orchestrator.
    """

    def __init__(self, cfg: LinkerConfig):
        self.cfg = cfg

    def link(
        self,
        mentions: List[Mention],
        entity_prior: Optional[Dict[NodeId, float]] = None,
        coherence_fn: Optional[callable] = None,
    ) -> Dict[str, Dict[str, object]]:
        pri = entity_prior or {}
        # Initialize by local best per mention
        assignment: Dict[str, Tuple[NodeId, float]] = {}
        for m in mentions:
            cs = sorted(m.candidates, key=lambda x: x[1], reverse=True)[: self.cfg.candidates_per_mention]
            if not cs:
                continue
            ent, score = cs[0]
            assignment[m.mention_id] = (ent, float(score))

        # Iterative coherence re-weighting (greedy)
        for _ in range(max(1, self.cfg.coherence_iterations)):
            linked_entities = [e for (_, (e, _)) in assignment.items()]
            for m in mentions:
                cs = sorted(m.candidates, key=lambda x: x[1], reverse=True)[: self.cfg.candidates_per_mention]
                best_ent, best_val = None, -1e9
                for ent, cand_score in cs:
                    prior = pri.get(ent, 0.0)
                    coh = 0.0
                    if coherence_fn and linked_entities:
                        coh = sum(coherence_fn(ent, le) for le in linked_entities) / max(1, len(linked_entities))
                    val = (
                        self.cfg.w_candidate * cand_score
                        + self.cfg.w_prior * prior
                        + self.cfg.w_coherence * coh
                    )
                    if val > best_val:
                        best_val = val
                        best_ent = ent
                if best_ent is not None:
                    assignment[m.mention_id] = (best_ent, float(best_val))

        # Produce results with normalized confidence in [0,1]
        # Here we map the composite score through min-max over chosen candidates for a rough normalization
        vals = [v for (_, v) in assignment.values()]
        vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
        rng = max(vmax - vmin, 1e-9)
        results: Dict[str, Dict[str, object]] = {}
        for mid, (ent, val) in assignment.items():
            norm = (val - vmin) / rng
            results[mid] = {"entity_id": ent, "link_confidence": float(norm)}
        return results


