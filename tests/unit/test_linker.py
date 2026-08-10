import pytest
from unittest.mock import Mock
import math

from retrieval.linker import CoherenceLinker, LinkerConfig, Mention, NodeId

# Fixtures
@pytest.fixture
def default_linker_config():
    return LinkerConfig(
        candidates_per_mention=3,
        coherence_iterations=1,
        persist_threshold=0.8,
        w_candidate=0.6,
        w_prior=0.3,
        w_coherence=0.1
    )

@pytest.fixture
def simple_mentions():
    return [
        Mention(
            mention_id="m1", surface="Apple", normalized="apple", span=(0,5), context="...Apple CEO...", llm_confidence=0.9,
            candidates=[("Apple_Company", 0.8), ("Apple_Fruit", 0.2)]
        ),
        Mention(
            mention_id="m2", surface="Tim Cook", normalized="tim cook", span=(10,18), context="...Tim Cook spoke...", llm_confidence=0.85,
            candidates=[("Tim_Cook_CEO", 0.9), ("Tim_Cook_Chef", 0.1)]
        )
    ]

@pytest.fixture
def complex_mentions_with_overlap():
    return [
        Mention(
            mention_id="m1", surface="AI", normalized="ai", span=(0,2), context="...AI research...", llm_confidence=0.9,
            candidates=[("Artificial_Intelligence", 0.7), ("Adobe_Illustrator", 0.3)]
        ),
        Mention(
            mention_id="m2", surface="Machine Learning", normalized="machine learning", span=(5,19), context="...Machine Learning models...", llm_confidence=0.8,
            candidates=[("Machine_Learning", 0.9)]
        ),
        Mention(
            mention_id="m3", surface="Deep Learning", normalized="deep learning", span=(25,38), context="...Deep Learning frameworks...", llm_confidence=0.82,
            candidates=[("Deep_Learning", 0.8), ("Deep_Blue_Chess", 0.2)]
        )
    ]

class TestCoherenceLinker:
    def test_basic_linking(self, default_linker_config, simple_mentions):
        linker = CoherenceLinker(default_linker_config)
        results = linker.link(simple_mentions)

        assert "m1" in results
        assert "m2" in results
        assert results["m1"]["entity_id"] == "Apple_Company"
        assert results["m2"]["entity_id"] == "Tim_Cook_CEO"
        assert 0.0 <= results["m1"]["link_confidence"] <= 1.0

    def test_linking_with_entity_prior(self, default_linker_config, simple_mentions):
        linker_config = default_linker_config
        linker_config.coherence_iterations = 0 # No coherence for this test, only prior
        linker_config.w_candidate = 0.2 # Lower candidate weight to let prior win
        linker_config.w_prior = 0.8     # Increase prior weight
        linker = CoherenceLinker(linker_config)
        
        # Strong prior for Apple_Fruit, which should override the candidate score for m1
        entity_prior = {"Apple_Fruit": 1.0, "Apple_Company": 0.1}
        results = linker.link(simple_mentions, entity_prior=entity_prior)

        # Expect Apple_Fruit to be picked due to strong prior, even if cand score lower
        assert results["m1"]["entity_id"] == "Apple_Fruit"
        assert results["m2"]["entity_id"] == "Tim_Cook_CEO"

    def test_linking_with_coherence_fn(self, default_linker_config, complex_mentions_with_overlap):
        linker_config = default_linker_config
        linker_config.coherence_iterations = 2
        linker_config.w_coherence = 0.5 # Increase coherence weight
        linker = CoherenceLinker(linker_config)

        # Mock coherence function: high coherence for AI-related entities
        def mock_coherence_fn(e1: NodeId, e2: NodeId) -> float:
            if all(x in ["Artificial_Intelligence", "Machine_Learning", "Deep_Learning"] for x in [e1, e2]):
                return 0.9
            return 0.1

        results = linker.link(complex_mentions_with_overlap, coherence_fn=mock_coherence_fn)

        # With coherence, all should resolve to AI concepts
        assert results["m1"]["entity_id"] == "Artificial_Intelligence"
        assert results["m2"]["entity_id"] == "Machine_Learning"
        assert results["m3"]["entity_id"] == "Deep_Learning"

    def test_empty_mentions(self, default_linker_config):
        linker = CoherenceLinker(default_linker_config)
        results = linker.link([])
        assert len(results) == 0

    def test_no_candidates(self, default_linker_config):
        mentions = [
            Mention(mention_id="m1", surface="Test", normalized="test", span=(0,4), context="...", llm_confidence=0.7,
                    candidates=[]
            )
        ]
        linker = CoherenceLinker(default_linker_config)
        results = linker.link(mentions)
        assert "m1" not in results # Should not link if no candidates

    def test_link_confidence_normalization(self, default_linker_config, simple_mentions):
        linker = CoherenceLinker(default_linker_config)
        results = linker.link(simple_mentions)
        for mid, data in results.items():
            assert 0.0 <= data["link_confidence"] <= 1.0

    def test_single_mention_multiple_candidates(self, default_linker_config):
        mentions = [
            Mention(
                mention_id="m1", surface="Foo", normalized="foo", span=(0,3), context="...Foo bar...", llm_confidence=0.8,
                candidates=[("Foo_A", 0.1), ("Foo_B", 0.9), ("Foo_C", 0.5)]
            )
        ]
        linker_config = default_linker_config
        linker_config.coherence_iterations = 0 # No coherence to keep it simple
        linker = CoherenceLinker(linker_config)
        results = linker.link(mentions)
        assert results["m1"]["entity_id"] == "Foo_B"
