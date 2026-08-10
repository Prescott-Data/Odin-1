from __future__ import annotations
from typing import Dict, Iterable, Tuple, List
from collections import defaultdict, deque
import random

from ..adapters import GraphAccessor, NodeId

class BiPPREngine:
    """
    Bidirectional PPR for source→target scoring:
    - Reverse push from targets builds fingerprints
    - Forward random walks from source intersect with reverse mass
    """

    def __init__(self, graph: GraphAccessor, alpha=0.15, rmax=1e-4):
        self.G = graph
        self.alpha, self.rmax = alpha, rmax

    def _reverse_push(self, targets: Iterable[NodeId]) -> Dict[NodeId, float]:
        p_t, r_t = defaultdict(float), defaultdict(float)
        q = deque()
        tgt = list(targets)
        if not tgt:
            return p_t
        mass = 1.0 / len(tgt)
        for t in tgt:
            r_t[t] = mass
            q.append(t)
        while q:
            u = q.popleft()
            if r_t[u] / max(1, self.G.in_degree(u)) <= self.rmax:
                continue
            push = (1 - self.alpha) * r_t[u]
            p_t[u] += self.alpha * r_t[u]
            r_t[u] = 0.0
            deg = self.G.in_degree(u)
            if deg == 0:
                continue
            share = push / deg
            # BiPPR needs in_neighbors, which needs to be in GraphAccessor and Mock
            for v in self.G.in_neighbors(u):
                r_t[v] += share
                if r_t[v] / max(1, self.G.in_degree(v)) > self.rmax:
                    q.append(v)
        return p_t

    def score(self, source: NodeId, targets: List[NodeId], walks=5000) -> List[Tuple[NodeId, float]]:
        fp = self._reverse_push(targets)
        hits = defaultdict(int)
        for _ in range(walks):
            u = source
            while True:
                hits[u] += 1
                if random.random() < self.alpha:
                    break
                nbrs = list(self.G.out_neighbors(u))
                if not nbrs:
                    break
                u = random.choice(nbrs)
        Z = float(sum(hits.values()) or 1)
        return sorted(((t, fp.get(t, 0.0) * hits.get(t, 0) / Z) for t in targets), key=lambda kv: kv[1], reverse=True)
