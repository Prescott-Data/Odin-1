from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import random
from ..adapters import GraphAccessor, NodeId
from ..budget import SearchBudget, BudgetTracker


@dataclass
class PPRParams:
    alpha: float = 0.2
    eps: float = 1e-6
    num_walks: int = 2000
    walk_len: int = 50
    topn: int = 200


@dataclass
class PPRResult:
    scores: List[Tuple[NodeId, float]]
    mass: float
    used_budget: Dict[str, int]
    trace: Dict[str, object]


def build_alias_table(weighted_neighbors: List[Tuple[NodeId, float]]):
    if not weighted_neighbors:
        return [], [], []
    total = sum(max(0.0, w) for _, w in weighted_neighbors) or 1.0
    probs = [(n, w / total) for n, w in weighted_neighbors]
    n = len(probs)
    scaled = [p * n for _, p in probs]
    alias, prob = [0] * n, [0.0] * n
    small, large = [], []
    for i, sp in enumerate(scaled):
        (small if sp < 1 else large).append(i)
    while small and large:
        s, l = small.pop(), large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] - (1 - prob[s])
        (small if scaled[l] < 1 else large).append(l)
    for i in small + large:
        prob[i] = 1.0
        alias[i] = i
    nodes = [n for n, _ in probs]
    return nodes, prob, alias


def alias_draw(nodes, prob, alias):
    if not nodes:
        return None
    i = random.randrange(len(nodes))
    return nodes[i] if random.random() < prob[i] else nodes[alias[i]]


class PushPPREngine:
    def __init__(self, accessor: GraphAccessor, community_id: str):
        self.A = accessor
        self.cid = community_id

    def run(self, seeds: List[NodeId], params: PPRParams, budget: Optional[SearchBudget] = None, personalization: Optional[Dict[NodeId, float]] = None) -> PPRResult:
        bt = BudgetTracker(budget or SearchBudget())
        p: Dict[NodeId, float] = defaultdict(float)
        r: Dict[NodeId, float] = defaultdict(float)
        q: deque[NodeId] = deque()

        if personalization is None:
            # Default behavior if no personalization is provided (uniform over seeds)
            seeds = seeds or []
            if not seeds:
                return PPRResult([], 0.0, bt.usage.__dict__, {"engine": "push", "iters": 0, "cache_hit": False})
            init_mass = 1.0 / len(seeds)
            for s in seeds:
                r[s] += init_mass
                q.append(s)
        else:
            # Use provided personalization vector
            for s, mass in personalization.items():
                r[s] += mass
                q.append(s)
            if not personalization:
                return PPRResult([], 0.0, bt.usage.__dict__, {"engine": "push", "iters": 0, "cache_hit": False})

        early_stop_reason = None
        iters = 0
        while q and not bt.over():
            u = q.popleft()
            iters += 1
            ru = r[u]
            if ru <= 0:
                continue
            p[u] += params.alpha * ru
            residual = (1 - params.alpha) * ru
            r[u] = 0.0

            nbrs = list(self.A.iter_out(u))
            deg = len(nbrs)
            if deg == 0:
                continue
            share = residual / deg
            for v, _, _ in nbrs:
                r[v] += share
                bt.tick_edges(1)
                if r[v] / max(1, self.A.degree(v)) > params.eps:
                    q.append(v)
            bt.tick_nodes(1)
            if bt.timed_out():
                early_stop_reason = "timeout"
                break

        if not early_stop_reason and bt.over():
            early_stop_reason = "budget_exhausted"

        items = sorted(p.items(), key=lambda kv: kv[1], reverse=True)[: params.topn]
        mass = sum(p.values())
        return PPRResult(scores=items, mass=mass, used_budget=bt.usage.__dict__, trace={"engine": "push", "iters": iters, "cache_hit": False, "early_stop_reason": early_stop_reason})


class MonteCarloPPREngine:
    def __init__(self, accessor: GraphAccessor, community_id: str, walk_index=None):
        self.A = accessor
        self.cid = community_id
        self._alias_cache: Dict[NodeId, Tuple[List[NodeId], List[float], List[int]]] = {}
        self.walk_index = walk_index

    def _alias_for(self, u: NodeId):
        if u in self._alias_cache:
            return self._alias_cache[u]
        nbrs = list(self.A.iter_out(u))
        table = build_alias_table([(v, w) for v, _, w in nbrs])
        self._alias_cache[u] = table
        return table

    def run(self, seeds: List[NodeId], params: PPRParams, budget: Optional[SearchBudget] = None, personalization: Optional[Dict[NodeId, float]] = None) -> PPRResult:
        bt = BudgetTracker(budget or SearchBudget())
        if not seeds and (personalization is None or not personalization):
            return PPRResult([], 0.0, bt.usage.__dict__, {"engine": "mc", "iters": 0, "cache_hit": False})

        hits: Dict[NodeId, int] = defaultdict(int)
        # Optional pre-hit sampling from walk index to save MC effort
        if self.walk_index is not None:
            for s in seeds:
                for v, c in self.walk_index.sample_hits(s).items():
                    hits[v] += int(c)
        
        # Prepare for weighted random choice if personalization is provided
        personalization_nodes = list(personalization.keys()) if personalization else seeds
        personalization_weights = list(personalization.values()) if personalization else [1.0] * len(seeds)
        # Normalize weights for random.choices if provided
        total_personalization_weight = sum(personalization_weights)
        if total_personalization_weight == 0:
            # Fallback to uniform if all weights are zero
            personalization_weights = [1.0] * len(personalization_nodes)
            total_personalization_weight = float(len(personalization_nodes))
        normalized_personalization_weights = [w / total_personalization_weight for w in personalization_weights]

        early_stop_reason = None
        for _ in range(params.num_walks):
            if bt.over():
                early_stop_reason = "budget_exhausted"
                break
            
            # Start walk from personalized distribution if available, else uniform from seeds
            if personalization:
                u = random.choices(personalization_nodes, weights=normalized_personalization_weights, k=1)[0]
            else:
                u = random.choice(seeds)

            for _ in range(params.walk_len):
                if bt.over(): # Check budget before each step
                    early_stop_reason = "budget_exhausted"
                    break
                hits[u] += 1
                bt.tick_nodes(1)
                if random.random() < params.alpha:
                    # Teleport back to personalized distribution if available, else uniform from seeds
                    if personalization:
                        u = random.choices(personalization_nodes, weights=normalized_personalization_weights, k=1)[0]
                    else:
                        u = random.choice(seeds)
                    continue
                nodes, prob, alias = self._alias_for(u)
                if not nodes:
                    # If dangling, teleport back to personalized distribution if available, else uniform from seeds
                    if personalization:
                        u = random.choices(personalization_nodes, weights=normalized_personalization_weights, k=1)[0]
                    else:
                        u = random.choice(seeds)
                    break
                u = alias_draw(nodes, prob, alias)
                bt.tick_edges(1)
                if bt.timed_out():
                    early_stop_reason = "timeout"
                    break

        total = float(sum(hits.values()) or 1.0)
        scores = sorted(((n, c / total) for n, c in hits.items()), key=lambda kv: kv[1], reverse=True)[: params.topn]
        return PPRResult(scores=scores, mass=1.0, used_budget=bt.usage.__dict__, trace={"engine": "mc", "iters": params.num_walks, "cache_hit": False, "early_stop_reason": early_stop_reason})


class BiPPREngine:
    """
    Bidirectional PPR for source→target scoring:
    - Reverse push from targets builds fingerprints
    - Forward random walks from source intersect with reverse mass
    """

    def __init__(self, graph: GraphAccessor, alpha=0.15, rmax=1e-4):
        self.G = graph
        self.alpha, self.rmax = alpha, rmax

    def _reverse_push(self, targets: List[NodeId]) -> Dict[NodeId, float]:
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


