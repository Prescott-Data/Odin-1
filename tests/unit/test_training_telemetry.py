"""
Unit tests for NPLL training telemetry: TrainingReport construction and
round-trip, BootstrapResult threading through KnowledgeBootstrapper, and
non-convergence warnings. Uses fake DB objects; no database required.
"""

import logging
from unittest.mock import patch

import pytest
import torch

from npll.bootstrap import (
    NPLL_MODEL_KEY,
    ODIN_MODELS_COLLECTION,
    BootstrapResult,
    KnowledgeBootstrapper,
    TrainingReport,
)
from npll.training.npll_trainer import TrainingResult


def make_training_result(**overrides) -> TrainingResult:
    params = dict(
        total_epochs=3,
        total_em_iterations=7,
        final_elbo=-1.25,
        best_elbo=-1.10,
        converged=True,
        elbo_history=[-3.0, -2.0, -1.5, -1.3, -1.25, -1.24, -1.25],
        validation_metrics_history=[],
        total_training_time=42.5,
        average_epoch_time=14.17,
        convergence_epoch=2,
        early_stopping_triggered=False,
        rule_weight_delta_history=[None, 0.5, 0.2, 0.05, 0.01, 0.002, 0.0005],
    )
    params.update(overrides)
    return TrainingResult(**params)


class TestTrainingReport:
    def test_from_training_result_preserves_all_evidence(self):
        result = make_training_result()
        report = TrainingReport.from_training_result(result, "2026-09-08T12:00:00Z")

        assert report.converged is True
        assert report.convergence_epoch == 2
        assert report.final_elbo == pytest.approx(-1.25)
        assert report.best_elbo == pytest.approx(-1.10)
        assert report.total_epochs == 3
        assert report.total_em_iterations == 7
        # Complete histories, no truncation
        assert report.elbo_history == result.elbo_history
        assert report.rule_weight_delta_history == result.rule_weight_delta_history
        assert report.trained_at == "2026-09-08T12:00:00Z"
        assert set(report.convergence_criteria) == {
            "elbo_rel_tol",
            "weight_abs_tol",
            "convergence_patience",
        }

    def test_dict_round_trip_is_lossless(self):
        report = TrainingReport.from_training_result(
            make_training_result(converged=False, convergence_epoch=None),
            "2026-09-08T12:00:00Z",
        )
        rehydrated = TrainingReport.from_dict(report.to_dict())
        assert rehydrated == report

    def test_to_dict_is_json_serializable(self):
        import json

        report = TrainingReport.from_training_result(
            make_training_result(), "2026-09-08T12:00:00Z"
        )
        encoded = json.dumps(report.to_dict())
        assert TrainingReport.from_dict(json.loads(encoded)) == report


class _Collection:
    def __init__(self):
        self.docs = {}

    def get(self, key):
        return self.docs.get(key)

    def has(self, key):
        return key in self.docs

    def insert(self, doc):
        self.docs[doc["_key"]] = doc

    def update(self, doc):
        self.docs[doc["_key"]].update(doc)

    def count(self):
        return len(self.docs)


class _AQL:
    def __init__(self, responses):
        self._responses = list(responses)

    def execute(self, query, **kwargs):
        return iter(self._responses.pop(0)) if self._responses else iter([])


class _FakeDB:
    """Minimal StandardDatabase stand-in for bootstrapper tests."""

    def __init__(self, aql_responses=None):
        self._collections = {ODIN_MODELS_COLLECTION: _Collection()}
        self.aql = _AQL(aql_responses or [])

    def has_collection(self, name):
        return name in self._collections

    def create_collection(self, name):
        self._collections[name] = _Collection()

    def collection(self, name):
        return self._collections.setdefault(name, _Collection())


class TestBootstrapResultThreading:
    def test_failed_extraction_returns_failed_result(self):
        bootstrapper = KnowledgeBootstrapper(db=_FakeDB())

        with patch.object(bootstrapper, "_compute_data_hash", return_value="h" * 64), \
             patch.object(bootstrapper, "_extract_triples", return_value=[]):
            result = bootstrapper.ensure_model_ready()

        assert isinstance(result, BootstrapResult)
        assert result.model is None
        assert result.source == "failed"
        assert result.report is None
        assert result.data_hash == "h" * 64

    def test_cached_weights_rehydrate_report(self):
        db = _FakeDB()
        report = TrainingReport.from_training_result(
            make_training_result(converged=False, convergence_epoch=None),
            "2026-09-01T00:00:00Z",
        )
        db.collection(ODIN_MODELS_COLLECTION).insert(
            {
                "_key": NPLL_MODEL_KEY,
                "data_hash": "h" * 64,
                "rule_weights": [0.5],
                "training_report": report.to_dict(),
                "trained_at": report.trained_at,
            }
        )
        bootstrapper = KnowledgeBootstrapper(db=db)

        sentinel_model = object()
        with patch.object(bootstrapper, "_compute_data_hash", return_value="h" * 64), \
             patch.object(bootstrapper, "_extract_triples", return_value=[("A", "r1", "B")]), \
             patch.object(bootstrapper, "_generate_smart_rules", return_value=[object()]), \
             patch("npll.bootstrap.create_initialized_npll_model") as create_model:
            model = create_model.return_value
            model.mln.rule_weights = torch.nn.Parameter(torch.tensor([0.0]))
            result = bootstrapper.ensure_model_ready()

        assert result.source == "cached_weights"
        assert result.model is not None
        assert result.report == report
        assert result.report.converged is False

    def test_legacy_doc_without_report_yields_none_report(self):
        db = _FakeDB()
        db.collection(ODIN_MODELS_COLLECTION).insert(
            {
                "_key": NPLL_MODEL_KEY,
                "data_hash": "h" * 64,
                "rule_weights": [0.5],
                "trained_at": "2026-08-01T00:00:00Z",
            }
        )
        bootstrapper = KnowledgeBootstrapper(db=db)

        with patch.object(bootstrapper, "_compute_data_hash", return_value="h" * 64), \
             patch.object(bootstrapper, "_extract_triples", return_value=[("A", "r1", "B")]), \
             patch.object(bootstrapper, "_generate_smart_rules", return_value=[object()]), \
             patch("npll.bootstrap.create_initialized_npll_model") as create_model:
            model = create_model.return_value
            model.mln.rule_weights = torch.nn.Parameter(torch.tensor([0.0]))
            result = bootstrapper.ensure_model_ready()

        assert result.source == "cached_weights"
        assert result.report is None

    def test_persisted_doc_contains_full_training_report(self):
        db = _FakeDB()
        bootstrapper = KnowledgeBootstrapper(db=db)
        training_result = make_training_result(converged=False, convergence_epoch=None)

        with patch.object(bootstrapper, "_extract_triples", return_value=[("A", "r1", "B"), ("B", "r2", "C")]), \
             patch("npll.bootstrap.create_initialized_npll_model") as create_model, \
             patch("npll.bootstrap.create_trainer") as create_trainer_mock:
            model = create_model.return_value
            model.mln.rule_weights = torch.nn.Parameter(torch.tensor([0.5]))
            create_trainer_mock.return_value.train.return_value = training_result

            result = bootstrapper.ensure_model_ready(force_retrain=True)

        assert result.source == "trained"
        assert result.report is not None
        assert result.report.converged is False

        doc = db.collection(ODIN_MODELS_COLLECTION).get(NPLL_MODEL_KEY)
        assert doc["version"] == "2.1"
        stored = TrainingReport.from_dict(doc["training_report"])
        assert stored == result.report
        assert stored.elbo_history == training_result.elbo_history

    def test_non_convergence_logs_warning(self, caplog):
        db = _FakeDB()
        bootstrapper = KnowledgeBootstrapper(db=db)
        training_result = make_training_result(converged=False, convergence_epoch=None)

        with patch.object(bootstrapper, "_extract_triples", return_value=[("A", "r1", "B"), ("B", "r2", "C")]), \
             patch("npll.bootstrap.create_initialized_npll_model") as create_model, \
             patch("npll.bootstrap.create_trainer") as create_trainer_mock, \
             caplog.at_level(logging.WARNING, logger="npll.bootstrap"):
            model = create_model.return_value
            model.mln.rule_weights = torch.nn.Parameter(torch.tensor([0.5]))
            create_trainer_mock.return_value.train.return_value = training_result

            bootstrapper.ensure_model_ready(force_retrain=True)

        assert any("WITHOUT convergence" in message for message in caplog.messages)


class TestEngineWarning:
    def _engine_stub(self, report, model=object(), source="trained"):
        """Build a bare OdinEngine carrying just the telemetry attributes."""
        from odin.engine import OdinEngine

        engine = OdinEngine.__new__(OdinEngine)
        engine.npll_model = model
        engine.training_report = report
        engine.npll_source = source
        return engine

    def test_warns_when_model_did_not_converge(self, caplog):
        report = TrainingReport.from_training_result(
            make_training_result(converged=False, convergence_epoch=None),
            "2026-09-08T12:00:00Z",
        )
        engine = self._engine_stub(report, source="cached_weights")

        with caplog.at_level(logging.WARNING, logger="odin"):
            engine._warn_if_not_converged()

        assert any("did NOT converge" in message for message in caplog.messages)
        assert any("cached_weights" in message for message in caplog.messages)

    def test_silent_when_converged(self, caplog):
        report = TrainingReport.from_training_result(
            make_training_result(converged=True), "2026-09-08T12:00:00Z"
        )
        engine = self._engine_stub(report)

        with caplog.at_level(logging.WARNING, logger="odin"):
            engine._warn_if_not_converged()

        assert caplog.messages == []

    def test_silent_without_report(self, caplog):
        engine = self._engine_stub(report=None)

        with caplog.at_level(logging.WARNING, logger="odin"):
            engine._warn_if_not_converged()

        assert caplog.messages == []

    def test_get_status_exposes_convergence(self):
        report = TrainingReport.from_training_result(
            make_training_result(converged=False, convergence_epoch=None),
            "2026-09-08T12:00:00Z",
        )
        engine = self._engine_stub(report, source="cached_weights")
        engine.community_id = "global"
        engine.accessor = object()

        status = engine.get_status()

        assert status["npll_converged"] is False
        assert status["npll_source"] == "cached_weights"
        assert status["npll_trained_at"] == "2026-09-08T12:00:00Z"
