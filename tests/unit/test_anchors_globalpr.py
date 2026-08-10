import pytest
from unittest.mock import Mock
import math

from retrieval.ppr.anchors import APPRAnchors, APPRAnchorParams
from retrieval.ppr.global_pr import GlobalPR, GlobalPRParams
from retrieval.ppr.engines import PPRParams # for PushPPREngine dependency
from retrieval.budget import SearchBudget
from retrieval.adapters import GraphAccessor, NodeId
from test_ppr_engines import MockGraphAccessor, small_graph_data, small_community_nodes, mock_accessor


@pytest.fixture
def large_graph_data():
    return {
        "A": [("B", "r", 1.0), ("C", "r", 1.0)],
        "B": [("D", "r", 1.0)],
        "C": [("E", "r", 1.0)],
        "D": [("F", "r", 1.0)],
        "E": [("F", "r", 1.0)],
        "F": [("A", "r", 1.0), ("G", "r", 1.0)],
        "G": [] # Dangling node
    }

@pytest.fixture
def large_community_nodes(large_graph_data):
    return set(large_graph_data.keys())

@pytest.fixture
def large_mock_accessor(large_graph_data, large_community_nodes):
    return MockGraphAccessor(large_graph_data, large_community_nodes)


class TestAPPRAnchors:
    def test_build_for_community_basic(self, mock_accessor, small_community_nodes):
        anchors_engine = APPRAnchors(mock_accessor)
        params = APPRAnchorParams(topn=2, alpha=0.15, eps=1e-4)
        
        # Use node 'A' as a seed for anchors
        anchor_scores = anchors_engine.build_for_community(community_id="test_community", seed_set=["A"], params=params)
        
        assert isinstance(anchor_scores, list)
        assert len(anchor_scores) <= params.topn
        assert all(s >= 0 for _, s in anchor_scores)
        # A should be a top anchor or highly ranked
        assert any(n == "A" for n, _ in anchor_scores)
        assert anchor_scores[0][1] > 0 # Top score should be positive

    def test_build_for_community_caching(self, mock_accessor, small_community_nodes):
        anchors_engine = APPRAnchors(mock_accessor)
        params = APPRAnchorParams(topn=5, alpha=0.15)
        seed_set = ["A", "B"]

        first_run = anchors_engine.build_for_community("test_community", seed_set, params)
        # Modify mock_accessor to ensure subsequent calls don't hit the underlying graph if cached
        mock_accessor.iter_out = Mock(return_value=[]) 

        second_run = anchors_engine.build_for_community("test_community", seed_set, params)
        assert first_run == second_run
        # Assert that the underlying PushPPREngine was not called again via iter_out
        mock_accessor.iter_out.assert_not_called()

    def test_empty_seed_set(self, mock_accessor):
        anchors_engine = APPRAnchors(mock_accessor)
        params = APPRAnchorParams(topn=5)
        anchor_scores = anchors_engine.build_for_community("test_community", [], params)
        assert len(anchor_scores) == 0


class TestGlobalPR:
    def test_fit_basic(self, large_mock_accessor, large_community_nodes):
        pr_engine = GlobalPR(large_mock_accessor, "test_community")
        params = GlobalPRParams(alpha=0.15, tol=1e-8, max_iter=100)
        scores = pr_engine.fit(params)
        
        assert isinstance(scores, dict)
        assert set(scores.keys()) == large_community_nodes
        assert pytest.approx(sum(scores.values())) == 1.0 # PageRank scores sum to 1
        assert all(s >= 0 for s in scores.values())

    def test_fit_dangling_node_redistribution(self, large_mock_accessor, large_community_nodes):
        pr_engine = GlobalPR(large_mock_accessor, "test_community")
        params = GlobalPRParams(alpha=0.15, tol=1e-8, max_iter=100)
        scores = pr_engine.fit(params)

        # Node G is dangling. Its mass should be redistributed, so it should still have a score.
        assert "G" in scores
        assert scores["G"] > 0
        # Without specific ground truth, we primarily check non-zero and sum to 1

    def test_fit_convergence(self, large_mock_accessor):
        pr_engine = GlobalPR(large_mock_accessor, "test_community")
        params = GlobalPRParams(alpha=0.15, tol=1e-4, max_iter=5) # Lower tolerance and max_iter for faster test
        scores = pr_engine.fit(params)
        # Assert that it did converge (or at least ran iterations)
        assert len(scores) > 0
        assert pytest.approx(sum(scores.values())) == 1.0

    def test_fit_empty_graph(self, mock_accessor):
        empty_accessor = MockGraphAccessor({}, set())
        pr_engine = GlobalPR(empty_accessor, "empty_community")
        scores = pr_engine.fit()
        assert len(scores) == 0
