from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable

from ..adapters import GraphAccessor, NodeId


@dataclass
class GlobalPRParams:
    alpha: float = 0.15
    tol: float = 1e-8
    max_iter: int = 100


class GlobalPR:
    """
    Power-iteration PageRank over nodes visible to the GraphAccessor in a given community.
    Handles dangling nodes by redistributing mass to personalization (uniform over nodes).
    """

    def __init__(self, accessor: GraphAccessor, community_id: str):
        self.A = accessor
        self.cid = community_id
        self.pr: Dict[NodeId, float] = {}

    def fit(self, params: GlobalPRParams = GlobalPRParams()) -> Dict[NodeId, float]:
        nodes = list(self.A.nodes(self.cid))
        if not nodes:
            self.pr = {}
            return self.pr
        n = len(nodes)
        idx: Dict[NodeId, int] = {u: i for i, u in enumerate(nodes)}
        pr_prev = [1.0 / n] * n
        alpha = params.alpha
        teleport = 1.0 / n

        # Precompute out-neighbors indices
        out_idx: Dict[int, list[int]] = {}
        for u in nodes:
            ui = idx[u]
            nbrs = [idx[v] for v, _, _ in self.A.iter_out(u)]
            out_idx[ui] = nbrs

        for _ in range(params.max_iter):
            pr = [0.0] * n
            dangling_mass = 0.0
            for ui in range(n):
                nbrs = out_idx[ui]
                if not nbrs:
                    dangling_mass += (1.0 - alpha) * pr_prev[ui]
                    pr[ui] += alpha * pr_prev[ui]
                    continue
                share = (1.0 - alpha) * pr_prev[ui] / len(nbrs)
                pr[ui] += alpha * pr_prev[ui]
                for vj in nbrs:
                    pr[vj] += share
            # Redistribute dangling to teleport set (uniform personalization)
            if dangling_mass > 0:
                add = dangling_mass * teleport
                pr = [x + add for x in pr]

            # Teleportation to uniform as well
            pr = [alpha * teleport + (1.0 - alpha) * (x - alpha * teleport) for x in pr]

            # Normalize and check convergence (L1)
            s = sum(pr) or 1.0
            pr = [x / s for x in pr]
            diff = sum(abs(pr[i] - pr_prev[i]) for i in range(n))
            pr_prev = pr
            if diff < params.tol:
                break

        self.pr = {u: pr_prev[idx[u]] for u in nodes}
        return self.pr


