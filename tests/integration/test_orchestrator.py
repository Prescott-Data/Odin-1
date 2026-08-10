import pytest
from unittest.mock import Mock, MagicMock
import time

from retrieval.orchestrator import RetrievalOrchestrator, OrchestratorParams
from retrieval.adapters import KGCommunityAccessor, OverlayAccessor
from retrieval.confidence import NPLLConfidence, ConstantConfidence
from retrieval.linker import CoherenceLinker, LinkerConfig, Mention
from retrieval.writers import ArangoWriter
from retrieval.metrics import MetricsLogger, JSONLSink
from retrieval.beam import BeamParams

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.kg_data_generator import KGDataGenerator
from typing import List, Tuple
from retrieval.adapters import NodeId, RelId
from retrieval.confidence import EdgeConfidenceProvider

# Fixtures
@pytest.fixture
def test_kg():
    generator = KGDataGenerator(seed=42)
    return generator.generate_community_kg(2, 5, 5, 2)[0] # 2 communities, 5 entities each, 5 intra-facts, 2 inter-facts

@pytest.fixture
def simple_test_kg():
    generator = KGDataGenerator(seed=42)
    return generator.generate_path_kg(5)

class MockNPLLConfidence(EdgeConfidenceProvider):
    def __init__(self, default_conf: float = 0.8):
        self._default = default_conf
        self._cache = {}

    def confidence_batch(self, edges: List[Tuple[NodeId, RelId, NodeId]]) -> List[float]:
        for u, r, v in edges:
            if (u, r, v) not in self._cache:
                self._cache[(u, r, v)] = self._default
        return [self._cache.get((u, r, v), self._default) for u, r, v in edges]

    def confidence(self, u: NodeId, rel: RelId, v: NodeId) -> float:
        return self.confidence_batch([(u, rel, v)])[0]

@pytest.fixture
def mock_npll_confidence():
    return MockNPLLConfidence(0.8)

@pytest.fixture
def mock_arango_writer():
    mock_writer = Mock(spec=ArangoWriter)
    mock_writer.maybe_write_link.return_value = True
    return mock_writer

@pytest.fixture
def mock_metrics_logger():
    mock_sink = Mock(spec=JSONLSink)
    return MetricsLogger(mock_sink)


class TestRetrievalOrchestratorIntegration:
    def test_retrieve_end_to_end(self, test_kg, mock_npll_confidence, mock_metrics_logger):
        all_nodes = {e.name for e in test_kg.entities}
        accessor = KGCommunityAccessor(test_kg, "C0", all_nodes)
        orc = RetrievalOrchestrator(
            accessor=accessor,
            edge_confidence=mock_npll_confidence,
        )
        orc.metrics_logger = mock_metrics_logger

        params = OrchestratorParams(community_id="C0")
        seeds = ["E0_0", "E0_1"]
        result = orc.retrieve(seeds=seeds, params=params, now_ts=time.time())

        assert "insight_score" in result
        assert "paths" in result
        assert "topk_ppr" in result
        assert isinstance(result["insight_score"], float)
        assert mock_metrics_logger.sink.write.called  # Ensure metrics are logged (may be called multiple times due to caching)

    def test_retrieve_with_anchor_prior(self, test_kg, mock_npll_confidence):
        all_nodes = {e.name for e in test_kg.entities}
        accessor = KGCommunityAccessor(test_kg, "C0", all_nodes)
        orc = RetrievalOrchestrator(accessor, edge_confidence=mock_npll_confidence)
        
        params = OrchestratorParams(community_id="C0")
        seeds = ["E0_0"]
        # E0_4 is not directly connected to E0_0 but is in the community.
        # Strong prior for E0_4 should make it appear higher in PPR.
        anchor_prior = {"E0_4": 1.0}
        
        result = orc.retrieve(seeds, params, anchor_prior=anchor_prior, now_ts=time.time())
        
        assert any(n == "E0_4" for n, _ in result["topk_ppr"])
        
    def test_link_and_retrieve_end_to_end(self, test_kg, mock_npll_confidence, mock_arango_writer, mock_metrics_logger):
        all_nodes = {e.name for e in test_kg.entities}
        base_accessor = KGCommunityAccessor(test_kg, "C0", all_nodes)
        orc = RetrievalOrchestrator(
            accessor=base_accessor,
            edge_confidence=mock_npll_confidence,
        )
        orc.metrics_logger = mock_metrics_logger
        
        mentions = [
            Mention("m1", "Entity Zero", "entity zero", (0,11), "...", 0.9, [("E0_0", 0.8), ("E1_0", 0.1)])
        ]
        params = OrchestratorParams(community_id="C0")
        
        result = orc.link_and_retrieve(
            mentions, 
            base_accessor=base_accessor, 
            params=params,
            now_ts=time.time(),
            persistence_writer=mock_arango_writer
        )

        assert "insight_score" in result
        assert result["trace"]["linked_entities"]["m1"]["entity_id"] == "E0_0"
        mock_arango_writer.maybe_write_link.assert_called_once()
        assert mock_metrics_logger.sink.write.called  # May be called multiple times due to caching

    def test_beam_params_filters(self, simple_test_kg, mock_npll_confidence):
        all_nodes = {e.name for e in simple_test_kg.entities}
        accessor = KGCommunityAccessor(simple_test_kg, "C0", all_nodes)
        orc = RetrievalOrchestrator(accessor, edge_confidence=mock_npll_confidence)
        
        params = OrchestratorParams(community_id="C0")
        seeds = ["P0"]
        
        # This should find paths
        result = orc.retrieve(seeds, params, now_ts=time.time())
        assert len(result["paths"]) > 0
        
        # This should find no paths
        beam_params_restrictive = BeamParams(allowed_relations={"non_existent_rel"})
        result_filtered = orc.retrieve(seeds, params, now_ts=time.time(), beam_override=beam_params_restrictive)
        assert len(result_filtered["paths"]) == 0

    def test_retrieve_empty_graph(self):
        from npll.core.knowledge_graph import KnowledgeGraph
        empty_kg = KnowledgeGraph()
        accessor = KGCommunityAccessor(empty_kg, "C0", set())
        orc = RetrievalOrchestrator(accessor)
        params = OrchestratorParams(community_id="C0")
        result = orc.retrieve(seeds=[], params=params)
        
        assert result["insight_score"] == 0.0
        assert len(result["paths"]) == 0
        assert len(result["topk_ppr"]) == 0
