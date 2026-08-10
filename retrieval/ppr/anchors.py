from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

from ..adapters import GraphAccessor, NodeId
from .engines import PushPPREngine, PPRParams


@dataclass
class APPRAnchorParams:
    alpha: float = 0.15
    eps: float = 1e-4
    topn: int = 200


class APPRAnchors:
    """
    Build per-community APPR anchor sets using push-based PPR.
    Intended for offline or periodic refresh to warm personalization priors.
    """

    def __init__(self, accessor: GraphAccessor):
        self.A = accessor
        self.cache: Dict[Tuple[str, str], List[Tuple[NodeId, float]]] = {}

    def build_for_community(
        self,
        community_id: str,
        seed_set: List[NodeId],
        params: APPRAnchorParams = APPRAnchorParams(),
    ) -> List[Tuple[NodeId, float]]:
        key = (community_id, ",".join(sorted(map(str, seed_set))))
        if key in self.cache:
            return self.cache[key]
        engine = PushPPREngine(self.A, community_id)
        p = PPRParams(alpha=params.alpha, eps=params.eps, topn=params.topn)
        res = engine.run(seeds=seed_set, params=p)
        self.cache[key] = res.scores
        return res.scores


