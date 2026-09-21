"""Tests for the backend-neutral OdinEngine constructor and capabilities."""

from unittest.mock import patch

import pytest

from odin.engine import OdinEngine
from retrieval.backends.base import BackendCapabilityError, BackendConfigurationError


class CompleteAccessor:
    def iter_out(self, node):
        return iter(())

    def iter_in(self, node):
        return iter(())

    def nodes(self, community_id):
        return iter(())

    def degree(self, node):
        return 0

    def get_node(self, node_id, fields=None):
        return {}

    def community_seed_norm(self, community_id, seeds):
        return seeds


class RetrievalOnlyBackend:
    def __init__(self):
        self.accessor_calls = []

    def accessor(self, community_id, community_mode):
        self.accessor_calls.append((community_id, community_mode))
        return CompleteAccessor()


class NullTrainingBackend(RetrievalOnlyBackend):
    def triple_source(self):
        return None

    def model_store(self, community_id, community_mode):
        return None


class IncompleteAccessorBackend(RetrievalOnlyBackend):
    def accessor(self, community_id, community_mode):
        return object()


def test_retrieval_only_backend_initializes_when_training_is_disabled():
    backend = RetrievalOnlyBackend()

    engine = OdinEngine(
        backend,
        community_id="fraud",
        community_mode="mapping",
        auto_train=False,
    )

    assert backend.accessor_calls == [("fraud", "mapping")]
    assert engine.backend is backend
    assert engine.global_accessor is None
    assert engine.has_npll is False


def test_training_requires_declared_backend_capabilities():
    with pytest.raises(BackendCapabilityError, match="triple_source, model_store"):
        OdinEngine(RetrievalOnlyBackend())


def test_training_rejects_unavailable_backend_capabilities():
    with pytest.raises(BackendCapabilityError, match="triple_source, model_store"):
        OdinEngine(NullTrainingBackend())


def test_retrain_requires_training_capabilities():
    engine = OdinEngine(RetrievalOnlyBackend(), auto_train=False)

    with pytest.raises(BackendCapabilityError, match="triple_source, model_store"):
        engine.retrain_model()


def test_raw_database_handles_raise_a_migration_error():
    with pytest.raises(BackendConfigurationError, match="GraphBackend, not a raw database handle"):
        OdinEngine(object(), auto_train=False)


def test_engine_rejects_an_incomplete_retrieval_accessor():
    with pytest.raises(BackendConfigurationError, match="iter_out"):
        OdinEngine(IncompleteAccessorBackend(), auto_train=False)


def test_unexpected_bootstrap_failure_is_not_converted_to_constant_confidence():
    backend = NullTrainingBackend()
    backend.triple_source = lambda: object()
    backend.model_store = lambda *_: object()

    with patch("odin.engine.KnowledgeBootstrapper") as bootstrap:
        bootstrap.return_value.ensure_model_ready.side_effect = RuntimeError("training exploded")
        with pytest.raises(RuntimeError, match="training exploded"):
            OdinEngine(backend)
