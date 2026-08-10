import pytest
from unittest.mock import Mock
from collections import defaultdict
from typing import Dict, List, Tuple, Set

from retrieval.ppr.engines import PushPPREngine, MonteCarloPPREngine, BiPPREngine, PPRParams, PPRResult
from retrieval.ppr.indexes import RandomWalkIndex, WalkIndexConfig
from retrieval.budget import SearchBudget, BudgetTracker
from retrieval.adapters import GraphAccessor, NodeId # Import NodeId

# Mock GraphAccessor for isolated testing
class MockGraphAccessor(GraphAccessor):
    def __init__(self, graph_data: Dict[NodeId, List[Tuple[NodeId, str, float]]], community_nodes: Set[NodeId]):
        self._graph = graph_data
        self._community_nodes = community_nodes
        self._in_graph = defaultdict(list)
        for u, edges in graph_data.items():
            for v, r, w in edges:
                self._in_graph[v].append((u, r, w))

    def iter_out(self, node: NodeId) -> List[Tuple[NodeId, str, float]]:
        return self._graph.get(node, [])
    
    def out_neighbors(self, node: NodeId) -> List[NodeId]:
        return [v for v, _, _ in self.iter_out(node)]

    def in_neighbors(self, node: NodeId) -> List[NodeId]:
        return [u for u, _, _ in self._in_graph.get(node, [])]

    def nodes(self, community_id: str) -> List[NodeId]:
        # For testing, we'll just return all community nodes regardless of the ID
        return list(self._community_nodes)

    def degree(self, node: NodeId) -> int:
        return len(self._graph.get(node, []))
    
    def in_degree(self, node: NodeId) -> int:
        return len(self._in_graph.get(node, []))

    def community_of(self, node: NodeId) -> str:
        # Placeholder, not directly used by PPR engines here
        return "test_community"


@pytest.fixture
def small_graph_data():
    return {
        "A": [("B", "rel1", 1.0), ("C", "rel2", 1.0)],
        "B": [("C", "rel3", 1.0)],
        "C": [("A", "rel4", 1.0)],
        "D": [] # Dangling node
    }

@pytest.fixture
def small_community_nodes(small_graph_data):
    return set(small_graph_data.keys())

@pytest.fixture
def mock_accessor(small_graph_data, small_community_nodes):
    return MockGraphAccessor(small_graph_data, small_community_nodes)

@pytest.fixture
def ppr_params():
    return PPRParams(alpha=0.1, eps=1e-5, num_walks=1000, walk_len=10, topn=10)


class TestPushPPREngine:
    def test_basic_run(self, mock_accessor, ppr_params):
        engine = PushPPREngine(mock_accessor, "test_community")
        result = engine.run(seeds=["A"], params=ppr_params)

        assert isinstance(result, PPRResult)
        assert len(result.scores) <= ppr_params.topn
        assert sum(s for _, s in result.scores) <= 1.0 + ppr_params.eps # Allow for float inaccuracies
        assert all(s >= 0 for _, s in result.scores)

    def test_personalization(self, mock_accessor, ppr_params):
        engine = PushPPREngine(mock_accessor, "test_community")
        personalization = {"A": 0.8, "B": 0.2}
        result = engine.run(seeds=[], params=ppr_params, personalization=personalization)

        assert isinstance(result, PPRResult)
        # A and B should have higher scores than C (unless C gets a lot from B)
        a_score = next((s for n, s in result.scores if n == "A"), 0.0)
        b_score = next((s for n, s in result.scores if n == "B"), 0.0)
        c_score = next((s for n, s in result.scores if n == "C"), 0.0)
        assert a_score > 0
        assert b_score > 0
        assert c_score >= 0

    def test_empty_seeds(self, mock_accessor, ppr_params):
        engine = PushPPREngine(mock_accessor, "test_community")
        result = engine.run(seeds=[], params=ppr_params)
        assert len(result.scores) == 0
        assert result.mass == 0.0

    def test_dangling_node(self, small_graph_data, small_community_nodes, ppr_params):
        # For push PPR, dangling nodes retain their mass if alpha > 0 and no personalization redirects it
        # With a personalization vector, mass from dangling nodes would redistribute to that personalization.
        # In our current PushPPREngine, dangling nodes simply stop pushing residual, retaining alpha*ru mass.
        # We'll assert that D appears in scores, but its mass should be low if not seeded.

        engine = PushPPREngine(MockGraphAccessor(small_graph_data, small_community_nodes), "test_community")
        result = engine.run(seeds=["D"], params=ppr_params)
        d_score = next((s for n, s in result.scores if n == "D"), 0.0)
        assert d_score > 0 # D should have some mass from its own seed

    def test_budget_adherence(self, mock_accessor, ppr_params):
        engine = PushPPREngine(mock_accessor, "test_community")
        budget = SearchBudget(max_nodes=1, max_edges=1)
        result = engine.run(seeds=["A"], params=ppr_params, budget=budget)
        assert result.used_budget["nodes"] <= budget.max_nodes +1 # Allow for a slight overflow on tick
        assert result.used_budget["edges"] <= budget.max_edges +1
        assert result.trace.get("early_stop_reason") # Should be stopped by budget


class TestMonteCarloPPREngine:
    def test_basic_run(self, mock_accessor, ppr_params):
        engine = MonteCarloPPREngine(mock_accessor, "test_community")
        result = engine.run(seeds=["A"], params=ppr_params)

        assert isinstance(result, PPRResult)
        assert len(result.scores) <= ppr_params.topn
        assert sum(s for _, s in result.scores) <= 1.0 + ppr_params.eps
        assert all(s >= 0 for _, s in result.scores)

    def test_personalization(self, mock_accessor, ppr_params):
        engine = MonteCarloPPREngine(mock_accessor, "test_community")
        personalization = {"A": 0.8, "B": 0.2}
        result = engine.run(seeds=[], params=ppr_params, personalization=personalization)

        assert isinstance(result, PPRResult)
        a_score = next((s for n, s in result.scores if n == "A"), 0.0)
        b_score = next((s for n, s in result.scores if n == "B"), 0.0)
        assert a_score > 0
        assert b_score > 0
        # Due to MC randomness, difficult to assert specific score relations without many walks

    def test_with_walk_index(self, mock_accessor, ppr_params):
        mock_walk_index = Mock(spec=RandomWalkIndex)
        mock_walk_index.sample_hits.return_value = {"A": 50, "B": 20}
        engine = MonteCarloPPREngine(mock_accessor, "test_community", walk_index=mock_walk_index)
        result = engine.run(seeds=["A"], params=ppr_params)
        mock_walk_index.sample_hits.assert_called_once_with("A")
        # Walk index hits should contribute to scores
        a_score = next((s for n, s in result.scores if n == "A"), 0.0)
        b_score = next((s for n, s in result.scores if n == "B"), 0.0)
        assert a_score > 0
        assert b_score > 0

    def test_budget_adherence(self, mock_accessor, ppr_params):
        engine = MonteCarloPPREngine(mock_accessor, "test_community")
        budget = SearchBudget(max_nodes=10, max_edges=1)
        result = engine.run(seeds=["A"], params=ppr_params, budget=budget)
        assert result.used_budget["edges"] <= budget.max_edges +1 # Can tick one more edge before checking
        assert result.trace.get("early_stop_reason")


class TestBiPPREngine:
    def test_basic_score(self, mock_accessor, ppr_params):
        engine = BiPPREngine(mock_accessor, alpha=ppr_params.alpha, rmax=ppr_params.eps)
        # Target C should get score via B then A from source A
        scores = engine.score(source="A", targets=["C"], walks=1000)

        assert isinstance(scores, list)
        assert len(scores) == 1
        assert scores[0][0] == "C"
        assert scores[0][1] > 0.0 # Should have some score

    def test_multiple_targets(self, mock_accessor, ppr_params):
        engine = BiPPREngine(mock_accessor, alpha=ppr_params.alpha, rmax=ppr_params.eps)
        scores = engine.score(source="A", targets=["B", "C", "D"], walks=1000)
        
        assert len(scores) == 3
        assert any(n == "B" for n, _ in scores)
        assert any(n == "C" for n, _ in scores)
        assert any(n == "D" for n, _ in scores)

        # C should likely be higher than B due to A->B->C path (and C->A->B loop contributes)
        score_b = next(s for n, s in scores if n == "B")
        score_c = next(s for n, s in scores if n == "C")
        # D is a dangling node, so it should have zero score from A
        score_d = next(s for n, s in scores if n == "D")

        assert score_d < ppr_params.eps # Should be very close to zero

    def test_no_path_to_target(self, small_graph_data, small_community_nodes, ppr_params):
        # Create a graph where A cannot reach X
        isolated_graph_data = small_graph_data.copy()
        isolated_graph_data["X"] = []
        isolated_community_nodes = small_community_nodes.union({"X"})
        isolated_accessor = MockGraphAccessor(isolated_graph_data, isolated_community_nodes)

        engine = BiPPREngine(isolated_accessor, alpha=ppr_params.alpha, rmax=ppr_params.eps)
        scores = engine.score(source="A", targets=["X"], walks=1000)
        
        assert len(scores) == 1
        assert scores[0][0] == "X"
        assert scores[0][1] < ppr_params.eps # Should be very close to zero
