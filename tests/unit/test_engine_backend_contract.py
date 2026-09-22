"""Tests for the backend-neutral OdinEngine constructor and capabilities."""

from unittest.mock import patch

import pytest
import torch

from odin.engine import OdinEngine
from npll.bootstrap import TrainingError
from retrieval.adapters import GraphAccessor
from retrieval.backends.base import BackendCapabilityError, BackendConfigurationError
from tests.utils.backend_fakes import MemorySource, MemoryStore
from tests.unit.test_training_telemetry import make_training_result


class CompleteAccessor:
    def iter_out(self, node):
        return iter(())

    def iter_in(self, node):
        return iter(())

    def iter_out_edges(self, node):
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


@pytest.mark.parametrize("missing", [
    "iter_out", "iter_in", "iter_out_edges", "nodes", "degree", "get_node", "community_seed_norm",
])
def test_engine_rejects_inherited_protocol_placeholders(missing):
    implementations = {
        name: method for name, method in vars(CompleteAccessor).items()
        if callable(method) and name != missing
    }
    incomplete = type("Incomplete", (GraphAccessor,), implementations)
    backend = RetrievalOnlyBackend()
    backend.accessor = lambda *_: incomplete()
    with pytest.raises(BackendConfigurationError, match=missing):
        OdinEngine(backend, auto_train=False)


def test_engine_accepts_concrete_protocol_subclass():
    class Concrete(CompleteAccessor, GraphAccessor):
        pass

    backend = RetrievalOnlyBackend()
    backend.accessor = lambda *_: Concrete()
    assert OdinEngine(backend, auto_train=False).get_neighbors("missing")["neighbors"] == []


def test_unexpected_bootstrap_failure_is_not_converted_to_constant_confidence():
    backend = NullTrainingBackend()
    backend.triple_source = lambda: object()
    backend.model_store = lambda *_: object()

    with patch("odin.engine.KnowledgeBootstrapper") as bootstrap:
        bootstrap.return_value.ensure_model_ready.side_effect = RuntimeError("training exploded")
        with pytest.raises(RuntimeError, match="training exploded"):
            OdinEngine(backend)


class TrainingBackend(RetrievalOnlyBackend):
    def __init__(self, triples=(("A", "r", "B"),)):
        super().__init__()
        self.source = MemorySource(triples)
        self.store = MemoryStore()

    def triple_source(self):
        return self.source

    def model_store(self, community_id, community_mode):
        return self.store


def test_exception_inside_trainer_fails_initialization_with_original_cause():
    backend = TrainingBackend()
    failure = RuntimeError("optimizer failed")
    with patch("npll.bootstrap.create_initialized_npll_model"), \
         patch("npll.bootstrap.create_trainer") as trainer:
        trainer.return_value.train.side_effect = failure
        with pytest.raises(TrainingError) as raised:
            OdinEngine(backend)
    assert raised.value.__cause__ is failure
    assert backend.store.saves == 0


def test_empty_training_graph_requires_explicit_retrieval_only_operation():
    backend = TrainingBackend(triples=())
    with pytest.raises(TrainingError, match="did not produce a model"):
        OdinEngine(backend)
    engine = OdinEngine(backend, auto_train=False)
    assert engine.score_edge("A", "r", "B") == 0.8
    assert backend.store.saves == 0


@pytest.mark.parametrize("failure_stage", ["trainer", "empty_result", "serving_setup"])
def test_failed_retraining_preserves_complete_serving_state(failure_stage):
    backend = TrainingBackend()
    with patch("npll.bootstrap.create_initialized_npll_model") as create, \
         patch("npll.bootstrap.create_trainer") as trainer:
        create.return_value.mln.rule_weights = torch.nn.Parameter(torch.tensor([0.5]))
        trainer.return_value.train.return_value = make_training_result()
        engine = OdinEngine(backend)
        previous = (engine.npll_model, engine.training_report, engine.npll_source,
                    engine.confidence, engine.orchestrator)
        previous_status = engine.get_status()
        if failure_stage == "trainer":
            trainer.return_value.train.side_effect = RuntimeError("optimizer failed")
            with pytest.raises(TrainingError):
                engine.retrain_model()
        elif failure_stage == "empty_result":
            backend.source.triples.clear()
            with pytest.raises(TrainingError, match="did not produce a model"):
                engine.retrain_model()
        else:
            with patch("odin.engine.RetrievalOrchestrator", side_effect=RuntimeError("setup failed")):
                with pytest.raises(RuntimeError, match="setup failed"):
                    engine.retrain_model()
        current = (engine.npll_model, engine.training_report, engine.npll_source,
                   engine.confidence, engine.orchestrator)
        assert all(a is b for a, b in zip(previous, current))
        assert engine.get_status() == previous_status
        if failure_stage != "serving_setup":
            assert backend.store.saves == 1
