"""Arango implementation of snapshot extraction and model persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from retrieval.adapters_arango import ArangoCommunityAccessor, GlobalGraphAccessor
from .base import (
    BackendConfigurationError,
    BackendError,
    BackendIOError,
    CorruptModelError,
    ModelConflictError,
    StoredModel,
    TrainingSnapshot,
    validate_model_artifact,
)

ODIN_MODELS_COLLECTION = "OdinModels"


@dataclass(frozen=True)
class ArangoGraphConfig:
    """Map an application's Arango collections and fields to Odin's graph contract.

    Arango document IDs (``_id``) and edge endpoints (``_from`` and ``_to``)
    are canonical. The relation field must contain a non-empty string for every
    trainable edge. Entity types are optional metadata that Odin emits as
    ``has_type`` triples when configured.
    """

    node_collection: str
    edge_collection: str
    relation_field: str
    entity_type_field: Optional[str] = None
    provenance_edge_collection: Optional[str] = None
    membership_collection: Optional[str] = None
    membership_entity_field: Optional[str] = None
    membership_community_field: Optional[str] = None
    bridge_collection: Optional[str] = None
    affinity_collection: Optional[str] = None
    community_algorithm: Optional[str] = None

    def __post_init__(self):
        required = (self.node_collection, self.edge_collection, self.relation_field)
        if any(not isinstance(value, str) or not value for value in required):
            raise ValueError(
                "node_collection, edge_collection, and relation_field must be "
                "non-empty strings"
            )
        optional = (
            self.entity_type_field,
            self.provenance_edge_collection,
            self.bridge_collection,
            self.affinity_collection,
            self.community_algorithm,
        )
        if any(value is not None and (not isinstance(value, str) or not value)
               for value in optional):
            raise ValueError("optional Arango graph fields must be non-empty strings")
        membership = (
            self.membership_collection,
            self.membership_entity_field,
            self.membership_community_field,
        )
        if any(membership) and not all(
            isinstance(value, str) and value for value in membership
        ):
            raise ValueError(
                "community membership requires collection, entity field, and community field"
            )
        bridge_access = (
            self.bridge_collection,
            self.affinity_collection,
            self.community_algorithm,
        )
        if any(bridge_access) and not all(
            isinstance(value, str) and value for value in bridge_access
        ):
            raise ValueError(
                "bridge access requires bridge collection, affinity collection, "
                "and community algorithm"
            )

    def namespace(self) -> str:
        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class ArangoTripleSource:
    """Read the existing global training graph in one AQL snapshot.

    Entity identities are full document IDs, matching ArangoCommunityAccessor.
    Relationship identities are stored verbatim (including case and spaces).
    Type triples use ``has_type`` and the exact configured entity-type value.
    Community retrieval scopes do not change the existing global training scope.
    """

    def __init__(self, db, graph: ArangoGraphConfig):
        self.db = db
        self.graph = graph

    def snapshot(self) -> TrainingSnapshot:
        types = ""
        bind_vars = {
            "@edges": self.graph.edge_collection,
            "relation_field": self.graph.relation_field,
        }
        if self.graph.entity_type_field is not None:
            types = """
        LET types = (
          FOR entity IN @@nodes
            FILTER entity[@type_field] != null
            RETURN [entity._id, "has_type", entity[@type_field]]
        )
            """
            bind_vars.update({
                "@nodes": self.graph.node_collection,
                "type_field": self.graph.entity_type_field,
            })
        else:
            types = "LET types = []"
        query = """
        LET relationships = (
          FOR rel IN @@edges
            LET source = DOCUMENT(rel._from)
            LET target = DOCUMENT(rel._to)
            FILTER source != null AND target != null
            RETURN [source._id, rel[@relation_field], target._id]
        )
        """ + types + """
        FOR triple IN UNION(relationships, types)
          RETURN triple
        """
        try:
            triples = tuple(self.db.aql.execute(query, bind_vars=bind_vars))
        except Exception as exc:
            raise BackendIOError(
                "Could not extract the complete Arango training snapshot"
            ) from exc
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
    """Arango capabilities mapped from an application's explicit graph config."""

    def __init__(self, db, graph: ArangoGraphConfig):
        self.db = db
        self.graph = graph

    def accessor(self, community_id: str, community_mode: str):
        if community_mode == "mapping" and self.graph.membership_collection is None:
            raise BackendConfigurationError(
                "community_mode='mapping' requires membership fields in "
                "ArangoGraphConfig"
            )
        provenance_collection = self.graph.provenance_edge_collection
        if provenance_collection and not self.db.has_collection(provenance_collection):
            provenance_collection = None
        return ArangoCommunityAccessor(
            self.db,
            community_id=community_id,
            nodes_collection=self.graph.node_collection,
            edges_collection=self.graph.edge_collection,
            relation_property=self.graph.relation_field,
            node_type_property=self.graph.entity_type_field or "",
            community_mode=community_mode,
            membership_collection=self.graph.membership_collection or "",
            membership_entity_field=self.graph.membership_entity_field or "",
            membership_community_field=self.graph.membership_community_field or "",
            provenance_edge_collection=provenance_collection,
            bridge_collection=self.graph.bridge_collection,
            affinity_collection=self.graph.affinity_collection,
            algorithm=self.graph.community_algorithm,
        )

    def triple_source(self) -> ArangoTripleSource:
        return ArangoTripleSource(self.db, self.graph)

    def model_store(self, community_id: str, community_mode: str) -> ArangoModelStore:
        namespace = json.dumps([
            self.db.name, self.graph.namespace(),
            community_id, community_mode,
        ], ensure_ascii=False, separators=(",", ":"))
        return ArangoModelStore(self.db, namespace=namespace)

    def global_accessor(self):
        global_collections = (
            self.graph.membership_collection,
            self.graph.bridge_collection,
            self.graph.affinity_collection,
            self.graph.community_algorithm,
        )
        if any(value is None for value in global_collections):
            return None
        return GlobalGraphAccessor(
            db=self.db,
            nodes_collection=self.graph.node_collection,
            edges_collection=self.graph.edge_collection,
            relation_property=self.graph.relation_field,
            membership_collection=self.graph.membership_collection,
            membership_entity_field=self.graph.membership_entity_field,
            membership_community_field=self.graph.membership_community_field,
            bridge_collection=self.graph.bridge_collection,
            affinity_collection=self.graph.affinity_collection,
            algorithm=self.graph.community_algorithm,
        )

    def schema_inspector(self):
        from retrieval.backends.arango_schema import ArangoSchemaInspector

        return ArangoSchemaInspector(self.db)
