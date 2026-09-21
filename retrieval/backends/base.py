"""Backend contracts: snapshot identity and revision-checked model persistence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple

from retrieval.adapters import GraphAccessor

Triple = Tuple[str, str, str]
MODEL_KEY = "npll_current"
ARTIFACT_VERSION = "3.0"


class BackendError(RuntimeError):
    """Base for failures that must never be interpreted as missing models."""


class BackendConfigurationError(BackendError):
    """A backend or retrieval accessor does not satisfy Odin's contract."""


class BackendCapabilityError(BackendError):
    """A requested operation is not supported by the configured backend."""


class BackendIOError(BackendError):
    """The backend could not complete an operation."""


class CorruptModelError(BackendError):
    """An artifact does not satisfy the current schema."""


class ModelConflictError(BackendError):
    """Another writer changed the artifact after it was read."""


@dataclass(frozen=True)
class TrainingSnapshot:
    """Canonical, immutable triples and the fingerprint of those exact triples.

    Input order is immaterial; duplicates and type triples are retained. IDs
    and relations are exact, case-sensitive strings supplied by the backend.
    A source must use the same identities as its retrieval accessor.
    """

    triples: Tuple[Triple, ...]
    data_hash: str = field(init=False)

    def __post_init__(self):
        triples = tuple(tuple(triple) for triple in self.triples)
        if any(len(t) != 3 or any(not isinstance(v, str) or not v for v in t)
               for t in triples):
            raise ValueError("Training triples require three non-empty string identities")
        triples = tuple(sorted(triples))
        payload = json.dumps(triples, ensure_ascii=False, separators=(",", ":"))
        object.__setattr__(self, "triples", triples)
        object.__setattr__(self, "data_hash", hashlib.sha256(payload.encode("utf-8")).hexdigest())


@dataclass(frozen=True)
class StoredModel:
    """Complete artifact plus an opaque revision for compare-and-swap."""

    document: Dict[str, Any]
    revision: str


class TripleSource(Protocol):
    def snapshot(self) -> TrainingSnapshot:
        """Read once; derive the fingerprint from the materialized training data."""
        ...


class ModelStore(Protocol):
    """A graph/community-scoped store of complete JSON model artifacts.

    None means absent only. Invalid artifacts raise CorruptModelError; backend
    failures raise BackendIOError. Save atomically replaces the whole artifact.
    expected_revision=None means create only; a revision means replace only if
    unchanged. Conflicts raise ModelConflictError, without retry or overwrite.
    """

    def load(self, key: str) -> Optional[StoredModel]: ...

    def save(self, key: str, document: Dict[str, Any], *,
             expected_revision: Optional[str]) -> str: ...


class SchemaIntrospector(Protocol):
    def get_schema_map(self, refresh: bool = False) -> Dict[str, Any]: ...


class GraphBackend(Protocol):
    def accessor(self, community_id: str, community_mode: str) -> GraphAccessor: ...


class TrainingBackend(Protocol):
    def triple_source(self) -> TripleSource: ...
    def model_store(self, community_id: str, community_mode: str) -> ModelStore: ...


class GlobalAccessBackend(Protocol):
    def global_accessor(self) -> Optional[GraphAccessor]: ...


class SchemaInspectionBackend(Protocol):
    def schema_inspector(self) -> Optional[SchemaIntrospector]: ...


def validate_model_artifact(document: Dict[str, Any]) -> None:
    """Validate the weights-only artifact without dropping any extra evidence.

    Version 3 requires a complete report, rules, and unbounded relation names.
    Transport metadata (keys, revisions, namespace) belongs to the store's
    envelope, not this document. Unknown JSON fields survive round trips.
    """
    def require(condition, message):
        if not condition:
            raise CorruptModelError(message)

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def count(value):
        return type(value) is int and value >= 0

    def json_value(value):
        if value is None or type(value) in (str, bool, int):
            return True
        if type(value) is float:
            return math.isfinite(value)
        if type(value) is list:
            return all(json_value(item) for item in value)
        if type(value) is dict:
            return all(type(key) is str and json_value(item) for key, item in value.items())
        return False

    require(type(document) is dict and json_value(document), "Artifact must be finite JSON")
    require(document.get("version") == ARTIFACT_VERSION, "Unsupported model artifact version")
    require(document.get("model_type") == "npll" and
            document.get("storage_type") == "weights_only", "Invalid model artifact type")
    digest = document.get("data_hash")
    require(isinstance(digest, str) and len(digest) == 64 and
            all(c in "0123456789abcdef" for c in digest), "Invalid snapshot fingerprint")
    require(isinstance(document.get("trained_at"), str) and bool(document["trained_at"]),
            "Missing training timestamp")
    weights = document.get("rule_weights")
    require(isinstance(weights, list) and bool(weights) and all(number(w) for w in weights),
            "Invalid rule weights")
    rules = document.get("rules")
    require(isinstance(rules, list) and len(rules) == len(weights), "Rules/weights mismatch")
    for rule in rules:
        require(isinstance(rule, dict) and
                all(isinstance(rule.get(k), str) and rule[k] for k in ("rule_id", "rule_text")) and
                number(rule.get("confidence")), "Invalid rule record")
    require(len({r["rule_id"] for r in rules}) == len(rules), "Duplicate rule identity")
    schema = document.get("schema_snapshot")
    require(isinstance(schema, dict), "Missing schema snapshot")
    require(all(count(schema.get(k)) for k in ("entity_count", "relation_count", "fact_count")),
            "Invalid schema counts")
    names = schema.get("relation_names")
    require(isinstance(names, list) and all(isinstance(n, str) and n for n in names),
            "Invalid relation names")
    require(len(names) == schema["relation_count"] and len(set(names)) == len(names),
            "Incomplete relation names")
    report = document.get("training_report")
    require(isinstance(report, dict), "Missing complete training report")
    require(all(type(report.get(k)) is bool for k in ("converged", "early_stopping_triggered")),
            "Invalid report flags")
    require(report.get("convergence_epoch") is None or count(report["convergence_epoch"]),
            "Invalid convergence epoch")
    require("convergence_epoch" in report, "Missing convergence epoch")
    require(all(count(report.get(k)) for k in ("total_epochs", "total_em_iterations")),
            "Invalid report iteration counts")
    require(all(number(report.get(k)) for k in ("final_elbo", "best_elbo", "training_time_seconds")),
            "Invalid report measurements")
    for key in ("elbo_history", "rule_weight_delta_history"):
        history = report.get(key)
        require(isinstance(history, list) and len(history) == report["total_em_iterations"],
                "Incomplete training history: " + key)
        require(all(number(v) or (key == "rule_weight_delta_history" and v is None)
                    for v in history), "Invalid training history: " + key)
    require(report.get("trained_at") == document["trained_at"], "Training timestamps differ")
    criteria = report.get("convergence_criteria")
    require(isinstance(criteria, dict) and all(number(criteria.get(k)) for k in
            ("elbo_rel_tol", "weight_abs_tol", "convergence_patience")),
            "Missing convergence criteria")
