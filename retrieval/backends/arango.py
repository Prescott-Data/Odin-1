"""Arango implementation of snapshot extraction and model persistence."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from retrieval.adapters_arango import ArangoCommunityAccessor, GlobalGraphAccessor
from .base import (
    BackendError, BackendIOError, CorruptModelError, ModelConflictError,
    StoredModel, TrainingSnapshot, validate_model_artifact,
)

ODIN_MODELS_COLLECTION = "OdinModels"


class ArangoTripleSource:
    """Read the existing global training graph in one AQL snapshot.

    Entity identities are full document IDs, matching ArangoCommunityAccessor.
    Relationship identities are stored verbatim (including case and spaces).
    Type triples use `has_type` and the exact string value of entity.type.
    Community retrieval scopes do not change the existing global training scope.
    """

    def __init__(self, db):
        self.db = db

    def snapshot(self) -> TrainingSnapshot:
        query = """
        LET relationships = (
          FOR rel IN ExtractedRelationships
            LET source = DOCUMENT(rel._from)
            LET target = DOCUMENT(rel._to)
            FILTER source != null AND target != null
            RETURN [source._id, rel.relationship, target._id]
        )
        LET types = (
          FOR entity IN ExtractedEntities
            FILTER entity.type != null
            RETURN [entity._id, "has_type", entity.type]
        )
        FOR triple IN UNION(relationships, types)
          RETURN triple
        """
        try:
            triples = tuple(self.db.aql.execute(query))
        except Exception as exc:
            raise BackendIOError("Could not extract the complete Arango training snapshot") from exc
        try:
            return TrainingSnapshot(triples)
        except ValueError as exc:
            raise BackendError("Arango training data contains invalid triple identities") from exc


class ArangoModelStore:
    """Namespace-isolated artifacts with Arango revision-checked replacement."""

    def __init__(self, db, *, namespace: str):
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("Model store namespace must be a non-empty string")
        self.db = db
        self.namespace = namespace

    def _key(self, key: str) -> str:
        if not isinstance(key, str) or not key:
            raise ValueError("Model key must be a non-empty string")
        payload = json.dumps([self.namespace, key], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(self, key: str) -> Optional[StoredModel]:
        storage_key = self._key(key)
        try:
            if not self.db.has_collection(ODIN_MODELS_COLLECTION):
                return None
            doc = self.db.collection(ODIN_MODELS_COLLECTION).get(storage_key)
        except Exception as exc:
            raise BackendIOError("Could not load Arango model artifact") from exc
        if doc is None:
            return None
        if (not isinstance(doc, dict) or doc.get("namespace") != self.namespace or
                doc.get("model_key") != key or not isinstance(doc.get("_rev"), str) or
                not doc["_rev"]):
            raise CorruptModelError("Invalid Arango model envelope")
        validate_model_artifact(doc.get("artifact"))
        return StoredModel(doc["artifact"], doc["_rev"])

    def save(self, key: str, document: Dict[str, Any], *,
             expected_revision: Optional[str]) -> str:
        validate_model_artifact(document)
        if expected_revision is not None and (not isinstance(expected_revision, str) or
                                              not expected_revision):
            raise ValueError("Expected revision must be a non-empty string or None")
        envelope = {
            "_key": self._key(key), "namespace": self.namespace,
            "model_key": key, "artifact": document,
        }
        try:
            if not self.db.has_collection(ODIN_MODELS_COLLECTION):
                try:
                    self.db.create_collection(ODIN_MODELS_COLLECTION)
                except Exception as exc:
                    # Another writer may have created the collection meanwhile.
                    if getattr(exc, "error_code", None) != 1207:
                        raise
            collection = self.db.collection(ODIN_MODELS_COLLECTION)
            if expected_revision is None:
                result = collection.insert(envelope)
            else:
                envelope["_rev"] = expected_revision
                result = collection.replace(envelope, check_rev=True)
        except Exception as exc:
            if getattr(exc, "error_code", None) in (1200, 1202, 1210):
                raise ModelConflictError("Arango model changed during training; save rejected") from exc
            raise BackendIOError("Could not save Arango model artifact") from exc
        if not isinstance(result, dict) or not isinstance(result.get("_rev"), str):
            raise BackendIOError("Arango save did not return a revision")
        return result["_rev"]


class ArangoBackend:
    """Capabilities for the existing ExtractedEntities/Relationships graph.

    The database and fixed collection pair identify the graph. The engine's
    community ID and mode additionally namespace model artifacts, even though
    training is global in this extraction phase.
    """

    def __init__(self, db):
        self.db = db

    def accessor(self, community_id: str, community_mode: str):
        return ArangoCommunityAccessor(self.db, community_id=community_id,
                                       community_mode=community_mode)

    def triple_source(self) -> ArangoTripleSource:
        return ArangoTripleSource(self.db)

    def model_store(self, community_id: str, community_mode: str) -> ArangoModelStore:
        namespace = json.dumps([
            self.db.name, "ExtractedEntities", "ExtractedRelationships",
            community_id, community_mode,
        ], ensure_ascii=False, separators=(",", ":"))
        return ArangoModelStore(self.db, namespace=namespace)

    def global_accessor(self):
        return GlobalGraphAccessor(db=self.db, algorithm="gnn")

    def schema_inspector(self):
        # Avoid importing odin's public entry point while bootstrap is loading.
        from odin.schema import SchemaInspector

        return SchemaInspector(self.db)
