import pytest
from unittest.mock import Mock
import math
from datetime import datetime, timedelta

from retrieval.scoring import (
    path_score, PathScoreConfig,
    insight_score, InsightScoreConfig,
    aggregate_evidence_strength,
    combine_edge_confidence, power_mean, recency_factor
)
from retrieval.adapters import NodeId, RelId
from retrieval.confidence import EdgeConfidenceProvider

# Mock implementations
class MockConfidenceProvider(EdgeConfidenceProvider):
    def __init__(self, confidences: dict = None, default_conf: float = 0.8):
        self._confidences = confidences or {}
        self._default = default_conf

    def confidence(self, u: NodeId, rel: RelId, v: NodeId) -> float:
        return self._confidences.get((u, rel, v), self._default)

def mock_edge_timestamp_lookup(u: NodeId, rel: RelId, v: NodeId, timestamps: dict = None):
    timestamps = timestamps or {}
    return timestamps.get((u, rel, v), None)


# Fixtures
@pytest.fixture
def default_path_score_cfg():
    return PathScoreConfig()

@pytest.fixture
def custom_path_score_cfg():
    return PathScoreConfig(
        relation_type_prior={"high_prior": 2.0, "low_prior": 0.5},
        recency_tau_days=10.0,
        recency_clamp=(0.1, 0.9),
        power_mean_rho=0.8
    )

@pytest.fixture
def default_insight_score_cfg():
    return InsightScoreConfig()

@pytest.fixture
def mock_now_ts():
    return datetime.now().timestamp()


class TestPathScorer:
    def test_path_score_basic(self, default_path_score_cfg, mock_now_ts):
        path_edges = [("A", "rel1", "B"), ("B", "rel2", "C")]
        node_ppr = {"A": 0.5, "B": 0.3, "C": 0.1}
        conf_provider = MockConfidenceProvider(default_conf=0.9)
        
        # Simple timestamp lookup: all edges happened 5 days ago
        five_days_ago = (datetime.now() - timedelta(days=5)).timestamp()
        timestamps = {e: five_days_ago for e in path_edges}
        ts_lookup_fn = lambda u,r,v: mock_edge_timestamp_lookup(u,r,v,timestamps)

        score = path_score(
            path_edges,
            node_ppr,
            conf_provider,
            mock_now_ts,
            ts_lookup_fn,
            default_path_score_cfg
        )
        assert score > 0
        assert isinstance(score, float)

    def test_path_score_empty_path(self, default_path_score_cfg, mock_now_ts):
        score = path_score([], {}, MockConfidenceProvider(), mock_now_ts, None, default_path_score_cfg)
        assert score == 0.0

    def test_path_score_zero_ppr(self, default_path_score_cfg, mock_now_ts):
        path_edges = [("A", "rel1", "B")]
        node_ppr = {"A": 0.0, "B": 0.0}
        conf_provider = MockConfidenceProvider(default_conf=0.9)
        score = path_score(path_edges, node_ppr, conf_provider, mock_now_ts, None, default_path_score_cfg)
        assert score < 1e-10  # Should be very close to zero due to min(v, 1e-12) in power_mean

    def test_path_score_custom_config(self, custom_path_score_cfg, mock_now_ts):
        path_edges = [("A", "high_prior", "B"), ("B", "low_prior", "C")]
        node_ppr = {"A": 0.8, "B": 0.7, "C": 0.6}
        conf_provider = MockConfidenceProvider(confidences={("A","high_prior","B"):0.95, ("B","low_prior","C"):0.7})
        
        now = datetime.now()
        old_ts = (now - timedelta(days=20)).timestamp()
        timestamps = {("A","high_prior","B"): old_ts, ("B","low_prior","C"): old_ts}
        ts_lookup_fn = lambda u,r,v: mock_edge_timestamp_lookup(u,r,v,timestamps)

        score = path_score(
            path_edges,
            node_ppr,
            conf_provider,
            mock_now_ts,
            ts_lookup_fn,
            custom_path_score_cfg
        )
        assert score > 0

    def test_combine_edge_confidence(self):
        # Test with raw, npll, calibrated
        c = combine_edge_confidence(raw=0.8, npll=0.9, calibrated=0.7, weights=(0.5, 0.4, 0.1))
        assert isinstance(c, float)
        assert 0 < c < 1
        # Test with only raw
        c_raw = combine_edge_confidence(raw=0.8, weights=(1.0, 0.0, 0.0))
        assert abs(c_raw - 0.8) < 1e-6

    def test_power_mean(self):
        assert power_mean([0.1, 0.2, 0.3], rho=1.0) == pytest.approx(0.2)
        assert power_mean([0.1, 0.2, 0.3], rho=0.0) == pytest.approx(math.pow(0.1*0.2*0.3, 1/3)) # geometric mean
        assert power_mean([], rho=0.5) == 0.0

    def test_recency_factor(self, mock_now_ts):
        now = datetime.fromtimestamp(mock_now_ts)
        ts_list = [
            (now - timedelta(days=10)).timestamp(),
            (now - timedelta(days=60)).timestamp(),
            None # Should be ignored
        ]
        # With default tau=30 days
        factor = recency_factor(ts_list, mock_now_ts, tau_days=30.0, clamp=(0.0, 1.0))
        assert isinstance(factor, float)
        assert 0 < factor < 1.0

        # Test clamping
        factor_clamped_high = recency_factor(ts_list, mock_now_ts, tau_days=1000.0, clamp=(0.95, 1.0))
        assert factor_clamped_high == pytest.approx(0.95)
        factor_clamped_low = recency_factor(ts_list, mock_now_ts, tau_days=0.1, clamp=(0.0, 0.05))
        assert factor_clamped_low < 1e-10 # Should be effectively zero and thus clamped by the lower bound


class TestInsightScorer:
    def test_insight_score_basic(self, default_insight_score_cfg):
        score = insight_score(
            evidence_strength=0.8,
            community_relevance=0.7,
            explanation_quality=0.6,
            business_impact_proxy=0.5,
            cfg=default_insight_score_cfg
        )
        assert isinstance(score, float)
        assert 0 < score < 1

    def test_insight_score_weights_normalize(self):
        cfg = InsightScoreConfig(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)
        score = insight_score(0.5, 0.5, 0.5, 0.5, cfg=cfg)
        assert score == pytest.approx(0.5) # Should be 0.5 due to normalization

    def test_aggregate_evidence_strength(self):
        # 1 - (1-s1)(1-s2)... formula
        assert aggregate_evidence_strength([0.1, 0.2, 0.3]) == pytest.approx(1 - (0.9 * 0.8 * 0.7))
        assert aggregate_evidence_strength([0.5, 0.5], top_k=1) == pytest.approx(0.5)
        assert aggregate_evidence_strength([]) == 0.0

    def test_insight_score_inputs_clamped(self, default_insight_score_cfg):
        score = insight_score(
            evidence_strength=1.5, # Should be clamped to 1.0
            community_relevance=-0.5, # Should be clamped to 0.0
            explanation_quality=0.5,
            business_impact_proxy=0.5,
            cfg=default_insight_score_cfg
        )
        # With default weights (0.5,0.2,0.2,0.1), expected: 0.5*1 + 0.2*0 + 0.2*0.5 + 0.1*0.5 = 0.5 + 0 + 0.1 + 0.05 = 0.65
        expected_score = (0.5 * 1.0) + (0.2 * 0.0) + (0.2 * 0.5) + (0.1 * 0.5)
        assert score == pytest.approx(expected_score / (0.5+0.2+0.2+0.1)) # normalized by sum of weights
