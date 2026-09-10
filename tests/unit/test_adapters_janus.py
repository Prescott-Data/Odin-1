"""
Unit tests for JanusGraphAccessor (issue #6): correct result handling via
project(), full accessor surface, and CachedGraphAccessor compatibility.
Uses a fake Gremlin traversal source; no live JanusGraph required.
"""

import pytest

from retrieval.adapters import JanusGraphAccessor
from retrieval.cache import CachedGraphAccessor

EDGES = [
    # (src, rel, dst, weight or None -> defaults to 1.0)
    ("A", "knows", "B", 0.9),
    ("A", "works_at", "C", None),
    ("B", "knows", "C", 0.5),
]

NODE_PROPS = {
    "A": {"communityId": ["c1"], "kind": ["person"]},
    "B": {"communityId": ["c1"], "kind": ["person"]},
    "C": {"communityId": ["c2"], "kind": ["org"]},
}


class _FakeTraversal:
    """Chainable stand-in that answers the exact query shapes the accessor builds."""

    def __init__(self, edges, node_props, node=None):
        self.edges = edges
        self.node_props = node_props
        self.node = node
        self.steps = []
        self.has_filter = None

    # chain steps -----------------------------------------------------------
    def V(self, node=None):
        return _FakeTraversal(self.edges, self.node_props, node)

    def _step(self, name):
        self.steps.append(name)
        return self

    def outE(self):
        return self._step("outE")

    def inE(self):
        return self._step("inE")

    def project(self, *keys):
        return self._step("project")

    def by(self, *_args):
        return self._step("by")

    def has(self, prop, value):
        self.has_filter = (prop, value)
        return self._step("has")

    def id_(self):
        return self._step("id_")

    def count(self):
        return self._step("count")

    def value_map(self):
        return self._step("value_map")

    # terminals --------------------------------------------------------------
    def _out_rows(self):
        return [
            {"neighbor": d, "rel": r, "weight": w if w is not None else 1.0}
            for s, r, d, w in self.edges if s == self.node
        ]

    def _in_rows(self):
        return [
            {"neighbor": s, "rel": r, "weight": w if w is not None else 1.0}
            for s, r, d, w in self.edges if d == self.node
        ]

    def toList(self):
        if "project" in self.steps:
            return self._out_rows() if "outE" in self.steps else self._in_rows()
        if "value_map" in self.steps:
            props = self.node_props.get(self.node)
            return [props] if props else []
        if "id_" in self.steps:
            if self.has_filter:
                prop, value = self.has_filter
                return [
                    nid for nid, props in self.node_props.items()
                    if props.get(prop, [None])[0] == value
                ]
            return list(self.node_props)
        raise AssertionError(f"unexpected terminal for steps {self.steps}")

    def next(self):
        if "count" in self.steps:
            rows = self._out_rows() if "outE" in self.steps else self._in_rows()
            return len(rows)
        raise AssertionError(f"unexpected next() for steps {self.steps}")


@pytest.fixture
def accessor():
    return JanusGraphAccessor(_FakeTraversal(EDGES, NODE_PROPS))


class TestJanusGraphAccessor:
    def test_iter_out_yields_plain_triples(self, accessor):
        edges = list(accessor.iter_out("A"))
        assert ("B", "knows", 0.9) in edges
        assert ("C", "works_at", 1.0) in edges  # missing weight -> default
        for neighbor, rel, weight in edges:
            assert isinstance(neighbor, str)
            assert isinstance(rel, str)
            assert isinstance(weight, float)

    def test_iter_out_unknown_node_is_empty(self, accessor):
        assert list(accessor.iter_out("missing")) == []

    def test_iter_in_reverses_direction(self, accessor):
        incoming = list(accessor.iter_in("C"))
        assert ("A", "works_at", 1.0) in incoming
        assert ("B", "knows", 0.5) in incoming

    def test_degree_matches_iter_out(self, accessor):
        for node in ("A", "B", "C"):
            assert accessor.degree(node) == len(list(accessor.iter_out(node)))

    def test_in_degree(self, accessor):
        assert accessor.in_degree("C") == 2
        assert accessor.in_degree("A") == 0

    def test_nodes_scoped_by_community(self, accessor):
        assert sorted(accessor.nodes("c1")) == ["A", "B"]
        assert sorted(accessor.nodes()) == ["A", "B", "C"]

    def test_community_seed_norm_is_passthrough(self, accessor):
        assert accessor.community_seed_norm("c1", ["A", "B"]) == ["A", "B"]

    def test_get_node_unwraps_value_map_lists(self, accessor):
        assert accessor.get_node("A") == {"communityId": "c1", "kind": "person"}
        assert accessor.get_node("A", fields=["kind"]) == {"kind": "person"}
        assert accessor.get_node("missing") == {}

    def test_works_behind_cached_accessor(self, accessor):
        cached = CachedGraphAccessor(accessor, cache_size=10)
        first = list(cached.iter_out("A"))
        second = list(cached.iter_out("A"))
        assert first == second
        assert len(first) == 2
