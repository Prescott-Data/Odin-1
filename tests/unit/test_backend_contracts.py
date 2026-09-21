"""Regression tests for snapshot identity and complete, atomic persistence."""

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest
import torch

from npll.bootstrap import KnowledgeBootstrapper, TrainingReport
from retrieval.backends.arango import ArangoBackend, ArangoModelStore, ArangoTripleSource
from retrieval.backends.base import (
    MODEL_KEY, BackendError, BackendIOError, CorruptModelError, ModelConflictError,
    TrainingSnapshot, validate_model_artifact,
)
from tests.unit.test_training_telemetry import make_training_result
from tests.utils.backend_fakes import (
    FakeArango, MemorySource, MemoryStore, model_artifact,
)


def test_snapshot_canonicalizes_order_but_preserves_all_triples():
    triples = [["Entity/é", "Works At", "Entity/B"], ["Entity/B", "has_type", "Person"]]
    snapshot = TrainingSnapshot(triples)
    assert snapshot == TrainingSnapshot(list(reversed(triples)))
    triples[0][1] = "changed"
    assert ("Entity/é", "Works At", "Entity/B") in snapshot.triples
    with pytest.raises(FrozenInstanceError):
        snapshot.triples = ()
    assert len(TrainingSnapshot(snapshot.triples * 2).triples) == 4
    assert TrainingSnapshot(snapshot.triples * 2).data_hash != snapshot.data_hash


@pytest.mark.parametrize("mutated", [
    [("A", "r", "C"), ("A", "has_type", "Person")],
    [("A", "r", "B"), ("A", "has_type", "Organization")],
    [("A", "R", "B"), ("A", "has_type", "Person")],
])
def test_same_count_endpoint_relation_and_type_changes_invalidate(mutated):
    original = TrainingSnapshot((("A", "r", "B"), ("A", "has_type", "Person")))
    assert TrainingSnapshot(mutated).data_hash != original.data_hash


def test_arango_materializes_one_snapshot_and_retains_tail_evidence():
    triples = [("Entities/A", "Relation %03d" % i, "Entities/B") for i in range(101)]
    triples.append(("Entities/A", "has_type", "Person"))
    db = FakeArango(triples)
    snapshot = ArangoTripleSource(db).snapshot()
    assert len(db.queries) == 1
    assert "source._id" in db.queries[0] and '"has_type"' in db.queries[0]
    assert snapshot == TrainingSnapshot(triples)
    db.triples[-1] = ("Entities/A", "has_type", "Organization")
    assert ("Entities/A", "Relation 100", "Entities/B") in snapshot.triples
    assert snapshot.data_hash != ArangoTripleSource(db).snapshot().data_hash


def test_extraction_errors_do_not_return_empty_or_partial_snapshots():
    db = FakeArango()

    def interrupted(query):
        yield ("A", "r", "B")
        raise OSError("cursor interrupted")

    db.aql.execute = interrupted
    with pytest.raises(BackendIOError, match="complete"):
        ArangoTripleSource(db).snapshot()
    db.aql.execute = lambda query: [("A", None, "B")]
    # Invalid identities are a data-validity failure, distinct from transport I/O.
    with pytest.raises(BackendError, match="invalid triple identities"):
        ArangoTripleSource(db).snapshot()


def test_arango_lossless_large_artifact_and_atomic_replacement():
    db = FakeArango()
    store = ArangoModelStore(db, namespace="graph/community")
    artifact = model_artifact(relation_count=101, history_length=120)
    first = store.save(MODEL_KEY, artifact, expected_revision=None)
    loaded = store.load(MODEL_KEY)
    assert loaded.document == artifact
    assert loaded.revision == first
    assert loaded.document["schema_snapshot"]["relation_names"][-1] == "relation_100"
    assert loaded.document["training_report"]["elbo_history"][-1] == -119
    assert TrainingReport.from_dict(loaded.document["training_report"]).to_dict() == artifact["training_report"]
    del artifact["evidence"]
    second = store.save(MODEL_KEY, artifact, expected_revision=first)
    assert second != first
    assert store.load(MODEL_KEY).document == artifact
    assert "evidence" not in store.load(MODEL_KEY).document


def test_concurrent_creation_and_replacement_reject_stale_writers():
    db = FakeArango()
    first = ArangoModelStore(db, namespace="graph/community")
    second = ArangoModelStore(db, namespace="graph/community")
    artifact = model_artifact()
    assert first.load(MODEL_KEY) is None
    assert second.load(MODEL_KEY) is None
    revision = first.save(MODEL_KEY, artifact, expected_revision=None)
    with pytest.raises(ModelConflictError):
        second.save(MODEL_KEY, artifact, expected_revision=None)
    read_by_second = second.load(MODEL_KEY)
    replacement = model_artifact()
    replacement["rule_weights"] = [0.75]
    first.save(MODEL_KEY, replacement, expected_revision=revision)
    with pytest.raises(ModelConflictError):
        second.save(MODEL_KEY, artifact, expected_revision=read_by_second.revision)
    assert first.load(MODEL_KEY).document == replacement


def test_backend_namespaces_separate_communities_modes_and_databases():
    db = FakeArango()
    a = ArangoBackend(db, community_id="a", community_mode="mapping").model_store()
    b = ArangoBackend(db, community_id="b", community_mode="mapping").model_store()
    unscoped = ArangoBackend(db, community_id="a").model_store()
    a.save(MODEL_KEY, model_artifact(), expected_revision=None)
    assert b.load(MODEL_KEY) is None
    assert unscoped.load(MODEL_KEY) is None
    other = FakeArango()
    other.name = "other_graph"
    assert ArangoBackend(other).model_store().namespace != ArangoBackend(db).model_store().namespace


def test_absence_corruption_and_backend_failure_are_distinct():
    db = FakeArango()
    store = ArangoModelStore(db, namespace="graph")
    assert store.load(MODEL_KEY) is None
    store.save(MODEL_KEY, model_artifact(), expected_revision=None)
    collection = next(iter(db.collections.values()))
    envelope = next(iter(collection.docs.values()))
    del envelope["artifact"]["training_report"]
    with pytest.raises(CorruptModelError, match="report"):
        store.load(MODEL_KEY)
    with patch.object(db, "has_collection", side_effect=OSError("unavailable")):
        with pytest.raises(BackendIOError):
            store.load(MODEL_KEY)
        with pytest.raises(BackendIOError):
            store.save(MODEL_KEY, model_artifact(), expected_revision=None)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(rule_weights=[float("nan")]),
    lambda d: d["schema_snapshot"]["relation_names"].pop(),
    lambda d: d["training_report"]["elbo_history"].pop(),
    lambda d: d.update(evidence={"unsupported_tuple": (1, 2)}),
    lambda d: d.update(version="2.1"),
    lambda d: d.update(rules=[]),
])
def test_corrupt_artifacts_are_rejected_before_writing(mutation):
    document = model_artifact()
    mutation(document)
    with pytest.raises(CorruptModelError):
        validate_model_artifact(document)


def test_rule_generation_change_is_staleness_not_corruption():
    """A code change to rule generation retrains instead of crashing startup."""
    source = MemorySource([("A", "r", "B")])
    store = MemoryStore()
    bootstrap = KnowledgeBootstrapper(source, store)
    with patch("npll.bootstrap.create_initialized_npll_model") as create, \
         patch("npll.bootstrap.create_trainer") as trainer:
        create.return_value.mln.rule_weights = torch.nn.Parameter(torch.tensor([0.5]))
        trainer.return_value.train.return_value = make_training_result()
        assert bootstrap.ensure_model_ready().source == "trained"
        assert bootstrap.ensure_model_ready().source == "cached_weights"

        original = bootstrap._generate_smart_rules

        def changed_generation(kg):
            rules = original(kg)
            for rule in rules:
                rule.confidence = 0.99  # simulate a rule-generation code change
            return rules

        with patch.object(bootstrap, "_generate_smart_rules", side_effect=changed_generation):
            result = bootstrap.ensure_model_ready()
    assert result.source == "trained"
    assert store.saves == 2


def test_bootstrap_uses_one_snapshot_even_if_graph_changes_during_load():
    source = MemorySource([("A", "r", "B"), ("B", "r", "C")])
    expected = TrainingSnapshot(source.triples)
    store = MemoryStore()
    bootstrap = KnowledgeBootstrapper(source, store)

    def change_graph(key):
        source.triples[:] = [("A", "r", "C"), ("C", "r", "B")]
        return None

    with patch.object(store, "load", side_effect=change_graph), \
         patch("npll.bootstrap.create_initialized_npll_model") as create, \
         patch("npll.bootstrap.create_trainer") as trainer:
        create.return_value.mln.rule_weights = torch.nn.Parameter(torch.tensor([0.5]))
        trainer.return_value.train.return_value = make_training_result()
        result = bootstrap.ensure_model_ready()
        kg = create.call_args.args[0]
    assert source.calls == 1
    assert result.data_hash == expected.data_hash
    assert {(t.head.name, t.relation.name, t.tail.name) for t in kg.known_facts} == set(expected.triples)
    assert store.load(MODEL_KEY).document["data_hash"] == expected.data_hash


def test_bootstrap_preserves_more_than_50_relations_and_retrains_on_mutation():
    source = MemorySource([("A", "r%03d" % i, "B") for i in range(101)])
    store = MemoryStore()
    bootstrap = KnowledgeBootstrapper(source, store)
    with patch("npll.bootstrap.create_initialized_npll_model") as create, \
         patch("npll.bootstrap.create_trainer") as trainer:
        create.return_value.mln.rule_weights = torch.nn.Parameter(torch.tensor([0.5]))
        trainer.return_value.train.return_value = make_training_result()
        first = bootstrap.ensure_model_ready()
        assert bootstrap.ensure_model_ready().source == "cached_weights"
        source.triples[0] = ("B", "r000", "A")
        second = bootstrap.ensure_model_ready()
    assert second.source == "trained"
    assert first.data_hash != second.data_hash
    artifact = store.load(MODEL_KEY).document
    assert artifact["schema_snapshot"]["relation_names"] == ["r%03d" % i for i in range(101)]
    assert artifact["schema_snapshot"]["fact_count"] == 101
    assert store.saves == 2


@pytest.mark.parametrize("error", [BackendIOError, CorruptModelError, ModelConflictError])
@pytest.mark.parametrize("operation", ["load", "save"])
def test_bootstrap_propagates_store_errors(error, operation):
    store = MemoryStore()
    bootstrap = KnowledgeBootstrapper(MemorySource([("A", "r", "B")]), store)
    with patch.object(store, operation, side_effect=error("failure")), \
         patch("npll.bootstrap.create_initialized_npll_model") as create, \
         patch("npll.bootstrap.create_trainer") as trainer:
        create.return_value.mln.rule_weights = torch.nn.Parameter(torch.tensor([0.5]))
        trainer.return_value.train.return_value = make_training_result()
        with pytest.raises(error):
            bootstrap.ensure_model_ready(force_retrain=True)


def test_force_retrain_uses_revision_read_before_training():
    store = MemoryStore()
    source = MemorySource([("A", "r", "B")])
    bootstrap = KnowledgeBootstrapper(source, store)
    with patch("npll.bootstrap.create_initialized_npll_model") as create, \
         patch("npll.bootstrap.create_trainer") as trainer:
        create.return_value.mln.rule_weights = torch.nn.Parameter(torch.tensor([0.5]))
        trainer.return_value.train.return_value = make_training_result()
        bootstrap.ensure_model_ready()

        def competing_training():
            current = store.load(MODEL_KEY)
            store.save(MODEL_KEY, current.document, expected_revision=current.revision)
            return make_training_result()

        trainer.return_value.train.side_effect = competing_training
        with pytest.raises(ModelConflictError):
            bootstrap.ensure_model_ready(force_retrain=True)


@pytest.mark.parametrize("method", ["_initialize_intelligence", "retrain_model"])
def test_engine_does_not_hide_backend_failure(method):
    from odin.engine import OdinEngine

    engine = OdinEngine.__new__(OdinEngine)
    engine.backend = ArangoBackend(FakeArango())
    with patch("odin.engine.KnowledgeBootstrapper") as bootstrap:
        bootstrap.return_value.ensure_model_ready.side_effect = BackendIOError("unavailable")
        with pytest.raises(BackendIOError):
            if method == "_initialize_intelligence":
                engine._initialize_intelligence(True)
            else:
                engine.retrain_model()
