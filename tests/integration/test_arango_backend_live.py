"""Live backend contract checks in a new, disposable Arango database.

Set ODIN_TEST_ARANGO_URL and ODIN_TEST_ARANGO_PASSWORD. The test account must
be able to create databases. Only the UUID-named database created here is
removed; existing databases are never modified.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from retrieval.backends.arango import ArangoBackend
from retrieval.backends.base import MODEL_KEY, ModelConflictError, TrainingSnapshot
from tests.utils.backend_fakes import model_artifact

pytestmark = pytest.mark.skipif(
    not os.environ.get("ODIN_TEST_ARANGO_URL"), reason="ODIN_TEST_ARANGO_URL not set",
)


@pytest.fixture
def db():
    from arango import ArangoClient

    client = ArangoClient(hosts=os.environ["ODIN_TEST_ARANGO_URL"])
    password = os.environ.get("ODIN_TEST_ARANGO_PASSWORD", "")
    system = client.db("_system", username="root", password=password)
    name = "odin_backend_test_" + uuid4().hex
    system.create_database(name)
    database = client.db(name, username="root", password=password)
    try:
        entities = database.create_collection("ExtractedEntities")
        edges = database.create_collection("ExtractedRelationships", edge=True)
        for key in ("A", "B", "C"):
            entities.insert({"_key": key, "type": "Person"})
        edges.insert({"_key": "ab", "_from": "ExtractedEntities/A",
                      "_to": "ExtractedEntities/B", "relationship": "related_to"})
        edges.insert({"_key": "bc", "_from": "ExtractedEntities/B",
                      "_to": "ExtractedEntities/C", "relationship": "related_to"})
        yield database
    finally:
        system.delete_database(name)
        client.close()


def test_snapshot_tracks_same_count_mutations_and_types(db):
    source = ArangoBackend(db).triple_source()
    first = source.snapshot()
    expected = [("ExtractedEntities/" + key, "has_type", "Person") for key in ("A", "B", "C")]
    expected += [("ExtractedEntities/A", "related_to", "ExtractedEntities/B"),
                 ("ExtractedEntities/B", "related_to", "ExtractedEntities/C")]
    assert first == TrainingSnapshot(expected)
    db.collection("ExtractedRelationships").update({"_key": "ab", "_to": "ExtractedEntities/C"})
    second = source.snapshot()
    assert len(first.triples) == len(second.triples)
    assert first.data_hash != second.data_hash
    db.collection("ExtractedEntities").update({"_key": "A", "type": "Organization"})
    assert second.data_hash != source.snapshot().data_hash


def test_lossless_roundtrip_and_concurrent_replacement(db):
    backend = ArangoBackend(db)
    store = backend.model_store()
    artifact = model_artifact(101, 120)
    revision = store.save(MODEL_KEY, artifact, expected_revision=None)
    assert store.load(MODEL_KEY).document == artifact

    def compete(weight):
        candidate = model_artifact(101, 120)
        candidate["rule_weights"] = [weight]
        try:
            backend.model_store().save(MODEL_KEY, candidate, expected_revision=revision)
            return "saved"
        except ModelConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(compete, [0.25, 0.75])) == ["conflict", "saved"]
    latest = store.load(MODEL_KEY)
    del latest.document["evidence"]
    store.save(MODEL_KEY, latest.document, expected_revision=latest.revision)
    assert store.load(MODEL_KEY).document == latest.document
    assert ArangoBackend(db, community_id="other").model_store().load(MODEL_KEY) is None


def test_real_train_save_reload_and_serve(db, monkeypatch):
    import npll.bootstrap as bootstrap_module
    from npll.bootstrap import KnowledgeBootstrapper
    from npll.training.npll_trainer import TrainingConfig
    from npll.utils.config import NPLLConfig
    from retrieval.confidence import NPLLConfidence

    # Exercise the real training loop with small dimensions and one iteration.
    config = NPLLConfig(entity_embedding_dim=4, relation_embedding_dim=4,
                        rule_embedding_dim=8, scoring_hidden_dim=8,
                        max_ground_rules=16, batch_size=4, device="cpu")
    monkeypatch.setattr(bootstrap_module, "get_config", lambda name: config)
    monkeypatch.setattr(bootstrap_module, "TrainingConfig", lambda **kwargs: TrainingConfig(
        num_epochs=1, max_em_iterations_per_epoch=1, save_checkpoints=False,
    ))
    backend = ArangoBackend(db)
    first = KnowledgeBootstrapper(backend.triple_source(), backend.model_store()).ensure_model_ready()
    assert first.source == "trained"
    second = KnowledgeBootstrapper(backend.triple_source(), backend.model_store()).ensure_model_ready()
    assert second.source == "cached_weights"
    assert first.report == second.report
    assert first.data_hash == second.data_hash
    assert first.model.mln.rule_weights.tolist() == second.model.mln.rule_weights.tolist()
    confidence = NPLLConfidence(second.model).confidence(
        "ExtractedEntities/A", "related_to", "ExtractedEntities/B",
    )
    assert math.isfinite(confidence) and 0 <= confidence <= 1
