"""Independent in-memory capability and Arango transport fakes."""

from copy import deepcopy
from types import SimpleNamespace

from retrieval.backends.base import (
    ARTIFACT_VERSION, ModelConflictError, StoredModel, TrainingSnapshot,
)


class MemorySource:
    def __init__(self, triples=()):
        self.triples = list(triples)
        self.calls = 0

    def snapshot(self):
        self.calls += 1
        return TrainingSnapshot(tuple(self.triples))


class MemoryStore:
    def __init__(self):
        self.docs = {}
        self.saves = 0

    def load(self, key):
        return deepcopy(self.docs.get(key))

    def save(self, key, document, *, expected_revision):
        current = self.docs.get(key)
        if (current.revision if current else None) != expected_revision:
            raise ModelConflictError("Stale revision")
        self.saves += 1
        revision = str(self.saves)
        self.docs[key] = StoredModel(deepcopy(document), revision)
        return revision


class ServerError(Exception):
    def __init__(self, code):
        self.error_code = code


class FakeCollection:
    def __init__(self):
        self.docs = {}
        self.revision = 0

    def get(self, key):
        return deepcopy(self.docs.get(key))

    def insert(self, doc):
        if doc["_key"] in self.docs:
            raise ServerError(1210)
        return self._write(doc)

    def replace(self, doc, *, check_rev):
        assert check_rev is True
        current = self.docs.get(doc["_key"])
        if current is None:
            raise ServerError(1202)
        if current["_rev"] != doc["_rev"]:
            raise ServerError(1200)
        return self._write(doc)

    def _write(self, doc):
        self.revision += 1
        stored = deepcopy(doc)
        stored["_rev"] = str(self.revision)
        self.docs[doc["_key"]] = stored
        return {"_key": doc["_key"], "_rev": stored["_rev"]}


class FakeArango:
    name = "test_graph"

    def __init__(self, triples=()):
        self.collections = {}
        self.triples = list(triples)
        self.queries = []
        self.aql = SimpleNamespace(execute=self.execute)

    def has_collection(self, name):
        return name in self.collections

    def create_collection(self, name):
        if name in self.collections:
            raise ServerError(1207)
        self.collections[name] = FakeCollection()

    def collection(self, name):
        return self.collections[name]

    def execute(self, query):
        self.queries.append(query)
        return iter(deepcopy(self.triples))


def model_artifact(relation_count=1, history_length=2):
    """A complete finite JSON artifact; extra nested evidence must also survive."""
    return {
        "version": ARTIFACT_VERSION,
        "model_type": "npll",
        "storage_type": "weights_only",
        "trained_at": "2026-09-21T00:00:00Z",
        "data_hash": "a" * 64,
        "rule_weights": [0.5],
        "rules": [{"rule_id": "r", "rule_text": "r(x,y) => r(x,y)", "confidence": 0.5}],
        "schema_snapshot": {
            "entity_count": 2, "fact_count": relation_count,
            "relation_count": relation_count,
            "relation_names": ["relation_%03d" % i for i in range(relation_count)],
        },
        "training_report": {
            "converged": True, "convergence_epoch": 1,
            "final_elbo": -1.0, "best_elbo": -1.0,
            "total_epochs": 1, "total_em_iterations": history_length,
            "elbo_history": [-float(i) for i in range(history_length)],
            "rule_weight_delta_history": [None] + [0.01] * (history_length - 1),
            "training_time_seconds": 1.0, "trained_at": "2026-09-21T00:00:00Z",
            "early_stopping_triggered": False,
            "convergence_criteria": {
                "elbo_rel_tol": 0.001, "weight_abs_tol": 0.001, "convergence_patience": 2,
            },
        },
        "evidence": {"nested": [None, True, "完整", {"tail": "preserve me"}]},
    }
