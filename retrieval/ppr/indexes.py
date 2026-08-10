from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional
import random

NodeId = int


@dataclass
class WalkIndexConfig:
    omega: int = 10     # walks per node (cap)
    rmax: float = 1e-3  # residual threshold knob (for capacity heuristics)
    alpha: float = 0.15 # teleport
    seed: int = 42


class RandomWalkIndex:
    """
    FIRM-style random-walk index (skeleton):
    - stores short geometric walks per node to accelerate SSPPR queries
    - supports O(1) expected-time updates under random arrival model (sketch)
    """

    def __init__(self, cfg: WalkIndexConfig):
        self.cfg = cfg
        self.walks: Dict[NodeId, List[List[NodeId]]] = {}
        random.seed(cfg.seed)

    def build(self, graph, nodes: Optional[Iterable[NodeId]] = None):
        nodes = nodes or graph.nodes()
        for u in nodes:
            self.walks[u] = self._sample_walks(graph, u, self.cfg.omega)

    def _sample_walks(self, graph, u: NodeId, k: int) -> List[List[NodeId]]:
        walks: List[List[NodeId]] = []
        for _ in range(k):
            path = [u]
            v = u
            while True:
                if random.random() < self.cfg.alpha:
                    break
                nbrs = list(graph.out_neighbors(v))
                if not nbrs:
                    break
                v = random.choice(nbrs)
                path.append(v)
            walks.append(path)
        return walks

    def on_edge_insert(self, graph, u: NodeId, v: NodeId):
        if u not in self.walks:
            return
        W = self.walks[u]
        target = max(1, int(graph.out_degree(u) * self.cfg.rmax * self.cfg.omega))
        while len(W) < target:
            W.append(self._sample_walks(graph, u, 1)[0])
        while len(W) > target and W:
            W.pop()

    def on_edge_delete(self, graph, u: NodeId, v: NodeId):
        if u not in self.walks:
            return
        W = self.walks[u]
        for _ in range(min(2, len(W))):
            if W:
                W.pop()
        target = max(1, int(graph.out_degree(u) * self.cfg.rmax * self.cfg.omega))
        while len(W) < target:
            W.append(self._sample_walks(graph, u, 1)[0])

    def sample_hits(self, source: NodeId) -> Dict[NodeId, int]:
        counts: Dict[NodeId, int] = {}
        for w in self.walks.get(source, []):
            for x in w:
                counts[x] = counts.get(x, 0) + 1
        return counts


