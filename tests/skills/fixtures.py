"""In-memory fixtures for skill evals. No database, no network."""
from __future__ import annotations

from odin import OdinEngine
from odin.backends.base import BackendIOError

EDGES = [
    ("claim_1", "billed_by", "provider_9"),
    ("provider_9", "flagged_in", "audit_3"),
    ("claim_1", "submitted_by", "person_4"),
]
SEEDS = ["claim_1"]
EMPTY_SEEDS = ["nobody"]


class _Accessor:
    def __init__(self, edges, fail):
        self.edges = list(edges)
        self.fail = fail

    def _check(self):
        if self.fail:
            raise BackendIOError("database unreachable")

    def iter_out(self, node):
        self._check()
        for u, rel, v in self.edges:
            if u == node:
                yield v, rel, 1.0

    def iter_in(self, node):
        self._check()
        for u, rel, v in self.edges:
            if v == node:
                yield u, rel, 1.0

    def iter_out_edges(self, node):
        self._check()
        for i, (u, rel, v) in enumerate(self.edges):
            if u == node:
                yield {"_id": f"e{i}", "u": u, "rel": rel, "v": v, "weight": 1.0,
                       "provenance": {"document_id": f"doc_{i}"}}

    def nodes(self, community_id):
        self._check()
        return sorted({n for u, _, v in self.edges for n in (u, v)})

    def degree(self, node):
        self._check()
        return sum(1 for u, _, _ in self.edges if u == node)

    def get_node(self, node_id, fields=None):
        self._check()
        return {"id": node_id} if node_id in self.nodes("global") else {}

    def community_seed_norm(self, community_id, seeds):
        return seeds


class _Backend:
    def __init__(self, edges, fail):
        self._accessor = _Accessor(edges, fail)

    def accessor(self, community_id, community_mode):
        return self._accessor


class RecordingEngine(OdinEngine):
    """OdinEngine that records calls so checks can inspect behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_edge_calls = []
        self.retrieve_path_counts = []

    def score_edge(self, src, rel, dst):
        self.score_edge_calls.append((src, rel, dst))
        return super().score_edge(src, rel, dst)

    def retrieve(self, *args, **kwargs):
        result = super().retrieve(*args, **kwargs)
        self.retrieve_path_counts.append(len(result["paths"]))
        return result


def make_engine(edges=EDGES, fail=False):
    return RecordingEngine(_Backend(edges, fail), auto_train=False)


def resolve(token):
    """Turn an argument token from prompts.json into a value."""
    named = {
        "engine": lambda: make_engine(),
        "failing_engine": lambda: make_engine(fail=True),
        "seeds": lambda: list(SEEDS),
        "empty_seeds": lambda: list(EMPTY_SEEDS),
        "edges": lambda: [list(e) for e in EDGES],
    }
    if isinstance(token, str) and token in named:
        return named[token]()
    return token
