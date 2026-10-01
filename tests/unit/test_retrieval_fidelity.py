import torch
from unittest.mock import patch

from npll.bootstrap import (KnowledgeBootstrapper, create_snapshot_initialized_model,
                            scoring_initialization_seed)
from npll.core.knowledge_graph import load_knowledge_graph_from_triples
from npll.utils.config import NPLLConfig
from odin.backends.base import TrainingSnapshot
from retrieval.beam import BeamParams, beam_search
from retrieval.budget import SearchBudget
from retrieval.confidence import ConstantConfidence, NPLLConfidence
from retrieval.orchestrator import RetrievalOrchestrator
from retrieval.scoring import score_paths_and_insight
from tests.utils.backend_fakes import MemorySource, MemoryStore


def small_model(snapshot, kg, rules):
    config = NPLLConfig(entity_embedding_dim=8, relation_embedding_dim=8,
                        scoring_hidden_dim=8, rule_embedding_dim=8, device="cpu")
    model = create_snapshot_initialized_model(snapshot, kg, rules, config)
    with torch.no_grad():
        model.mln.rule_weights.fill_(0.5)
    return model


def test_snapshot_seed_preserves_every_score_after_random_reinitialization():
    triples = [("nodes/a", "relation_%03d" % index, "nodes/b") for index in range(101)]
    snapshot = TrainingSnapshot(tuple(triples))
    kg = load_knowledge_graph_from_triples(triples)
    rules = KnowledgeBootstrapper(MemorySource(), MemoryStore())._generate_smart_rules(kg)
    model = small_model(snapshot, kg, rules)
    expected = NPLLConfidence(model).confidence_batch(triples)
    torch.manual_seed(937)
    reloaded = small_model(snapshot, kg, rules)
    assert NPLLConfidence(reloaded).confidence_batch(triples) == expected
    assert scoring_initialization_seed(snapshot) == scoring_initialization_seed(snapshot)


def test_bootstrap_persists_small_deterministic_state_and_reloads_exact_scores():
    source = MemorySource([("nodes/a", "r1", "nodes/b"), ("nodes/b", "r2", "nodes/c")])
    store = MemoryStore()
    config = NPLLConfig(entity_embedding_dim=8, relation_embedding_dim=8,
                        scoring_hidden_dim=8, rule_embedding_dim=8, device="cpu")
    with patch("npll.bootstrap.get_config", return_value=config):
        trained = KnowledgeBootstrapper(source, store).ensure_model_ready()
        torch.manual_seed(784)
        reloaded = KnowledgeBootstrapper(source, store).ensure_model_ready()
    assert reloaded.source == "cached_weights"
    assert NPLLConfidence(trained.model).confidence_batch(source.triples) == \
        NPLLConfidence(reloaded.model).confidence_batch(source.triples)
    state = store.load("npll_current").document["inference_state"]
    assert set(state) == {"config", "initialization_seed", "scorer_training"}


class ParallelAccessor:
    def iter_out_edges(self, node):
        if node == "nodes/start":
            for index in range(151):
                yield {"_id": f"edges/{index}", "u": node, "v": "nodes/end",
                       "rel": "requires", "weight": 1.0,
                       "provenance": {"source_refs": [{"text": "evidence " * 2000 + str(index)}]}}


def test_parallel_assertions_and_long_tail_evidence_survive_scoring_and_normalization():
    accessor = ParallelAccessor()
    confidence = ConstantConfidence(0.37)
    ppr = [("nodes/start", 0.6), ("nodes/end", 0.4)]
    beam = beam_search(accessor, "global", ["nodes/start"], ppr,
                       budget=SearchBudget(max_paths=200, max_nodes=1000, max_edges=1000),
                       beam_params=BeamParams(hop_limit=1, beam_width=200, max_paths=200),
                       conf_provider=confidence)
    scored = score_paths_and_insight(accessor, "global", ["nodes/start"], ppr,
                                    [path["edges"] for path in beam["paths"]], confidence)
    orchestrator = RetrievalOrchestrator(accessor, edge_confidence=confidence)
    result = orchestrator._normalize_paths_for_aggregators(scored["paths"])
    assert len(result) == 151
    by_id = {path["edges"][0]["_id"]: path for path in result}
    for original in accessor.iter_out_edges("nodes/start"):
        path = by_id[original["_id"]]
        edge = path["edges"][0]
        assert edge["provenance"] == original["provenance"]
        assert edge["relation"] == "requires"
        assert edge["confidence"] == 0.37
        assert path["decomp"]["edge_confidences"] == [0.37]


def test_arango_metadata_uses_only_mapped_fields_and_excludes_nested_vectors():
    from retrieval.adapters_arango import EdgeView, edge_record
    assertion = {"_id": "edges/a", "created_at": "unmapped", "timestamp": "unmapped",
                 "source_doc": "unmapped", "embedding": [1, 2],
                 "nested": [{"npll_embedding": [3], "text": "tail evidence"}],
                 "predicate": "Submitted By", "observed_at": "2026-10-01"}
    edge = EdgeView("nodes/b", "Submitted By", 1.0, "edges/a", None, None,
                    None, None, None, None, [], assertion)
    record = edge_record("nodes/a", edge)
    assert record["created_at"] is None
    assert "assertion" not in record
    assert record["provenance"]["assertion"]["created_at"] == "unmapped"
    assert record["provenance"]["assertion"]["nested"] == [{"text": "tail evidence"}]
    assert record["excluded_vector_fields"] == ["provenance.assertion.embedding",
                                                "provenance.assertion.nested[0].npll_embedding"]
    mapped = edge_record("nodes/a", edge._replace(timestamp="2026-10-01"))
    mapped["confidence"] = 0.4
    normalized = RetrievalOrchestrator(object())._normalize_paths_for_aggregators(
        [{"edges": [mapped]}])[0]["edges"][0]
    assert normalized["created_at"] == "2026-10-01"
    assert normalized["relation"] == "Submitted By"


def test_arango_validity_filter_never_guesses_fields():
    import pytest
    from retrieval.adapters_arango import ArangoCommunityAccessor
    from tests.utils.backend_fakes import FakeArango
    db = FakeArango()
    args = dict(community_id="global", nodes_collection="Nodes", edges_collection="Edges",
                relation_property="predicate", current_only=True, as_of="2026-10-01")
    with pytest.raises(ValueError, match="configured validity"):
        list(ArangoCommunityAccessor(db, **args).iter_out("Nodes/a"))
    accessor = ArangoCommunityAccessor(db, edge_valid_from_property="starts' at", **args)
    list(accessor.iter_out("Nodes/a"))
    assert "starts' at" not in db.queries[-1]
    assert db.query_arguments[-1]["bind_vars"]["valid_from_field"] == "starts' at"


def test_snapshot_scorer_learns_arbitrary_relations_and_replays_without_vectors():
    from npll.bootstrap import create_snapshot_trained_model
    triples = [("Records/a", "submitted_by", "Records/b"),
               ("Records/b", "approved_by", "Records/a")]
    snapshot = TrainingSnapshot(tuple(triples))
    kg = load_knowledge_graph_from_triples(snapshot.triples)
    rules = KnowledgeBootstrapper(MemorySource(), MemoryStore())._generate_smart_rules(kg)
    config = NPLLConfig(entity_embedding_dim=8, relation_embedding_dim=8,
                        scoring_hidden_dim=8, rule_embedding_dim=8, device="cpu")
    initial = create_snapshot_initialized_model(snapshot, kg, rules, config)
    model = create_snapshot_trained_model(snapshot, kg, rules, config)
    scorer = NPLLConfidence(model)
    positive = scorer.confidence("Records/a", "submitted_by", "Records/b")
    negative = scorer.confidence("Records/a", "submitted_by", "Records/a")
    assert positive > negative
    assert positive != NPLLConfidence(initial).confidence(*triples[0])
    losses = model.scorer_training["loss_history"]
    assert len(losses) == config.scorer_epochs
    assert losses[-1] < losses[0]
    torch.manual_seed(1987)
    replayed = create_snapshot_trained_model(snapshot, kg, rules, config)
    assert replayed.scorer_training == model.scorer_training
    assert NPLLConfidence(replayed).confidence_batch(triples) == scorer.confidence_batch(triples)
    assert model.scorer_training["excluded_vector_fields"]
