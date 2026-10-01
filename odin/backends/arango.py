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
    edge_weight_field: Optional[str] = None
    edge_timestamp_field: Optional[str] = None
    edge_valid_from_field: Optional[str] = None
    edge_valid_to_field: Optional[str] = None
    edge_status_field: Optional[str] = None
    allowed_edge_statuses: Optional[tuple[str, ...]] = None
    edge_provenance_fields: tuple[str, ...] = ()
    community_property_field: Optional[str] = None
    provenance_edge_collection: Optional[str] = None
    provenance_target_collections: tuple[str, ...] = ()
    membership_collection: Optional[str] = None
    membership_entity_field: Optional[str] = None
    membership_community_field: Optional[str] = None
    bridge_collection: Optional[str] = None
    affinity_collection: Optional[str] = None
    community_algorithm: Optional[str] = None
    membership_algorithm_field: Optional[str] = None
    bridge_entity_field: Optional[str] = None
    bridge_strength_field: Optional[str] = None
    bridge_community_field: Optional[str] = None
    bridge_algorithm_field: Optional[str] = None
    affinity_from_field: Optional[str] = None
    affinity_to_field: Optional[str] = None
    affinity_score_field: Optional[str] = None
    affinity_algorithm_field: Optional[str] = None

    def __post_init__(self):
        required = (self.node_collection, self.edge_collection, self.relation_field)
        if any(not isinstance(value, str) or not value for value in required):
            raise ValueError(
                "node_collection, edge_collection, and relation_field must be "
                "non-empty strings"
            )
        optional = (
            self.entity_type_field,
            self.edge_weight_field,
            self.edge_timestamp_field,
            self.edge_valid_from_field,
            self.edge_valid_to_field,
            self.edge_status_field,
            self.community_property_field,
            self.provenance_edge_collection,
            self.bridge_collection,
            self.affinity_collection,
            self.community_algorithm,
        )
        if any(value is not None and (not isinstance(value, str) or not value)
               for value in optional):
            raise ValueError("optional Arango graph fields must be non-empty strings")
        if (not isinstance(self.edge_provenance_fields, tuple) or
                any(not isinstance(value, str) or not value
                    for value in self.edge_provenance_fields)):
            raise ValueError("edge_provenance_fields must be a tuple of non-empty strings")
        if (not isinstance(self.provenance_target_collections, tuple) or
                any(not isinstance(value, str) or not value
                    for value in self.provenance_target_collections)):
            raise ValueError("provenance_target_collections must be a tuple of non-empty strings")
        if self.allowed_edge_statuses is not None:
            if (self.edge_status_field is None or
                    not isinstance(self.allowed_edge_statuses, tuple) or
                    any(not isinstance(value, str) or not value
                        for value in self.allowed_edge_statuses)):
                raise ValueError("allowed_edge_statuses requires a status field and tuple of strings")
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
        for label, values in (
            ("bridge", (self.bridge_collection, self.bridge_entity_field,
                        self.bridge_strength_field, self.bridge_community_field)),
            ("affinity", (self.affinity_collection, self.affinity_from_field,
                          self.affinity_to_field, self.affinity_score_field)),
        ):
            if any(v is not None for v in values) and not all(
                isinstance(v, str) and v for v in values
            ):
                raise ValueError(f"{label} access requires a complete collection and field mapping")
        algorithm_fields = (self.membership_algorithm_field, self.bridge_algorithm_field,
                            self.affinity_algorithm_field)
        if any(v is not None for v in algorithm_fields):
            if not self.community_algorithm or any(
                v is not None and (not isinstance(v, str) or not v) for v in algorithm_fields
            ):
                raise ValueError("algorithm field mappings require community_algorithm")
        elif self.community_algorithm is not None:
            raise ValueError("community_algorithm requires an explicit algorithm field mapping")

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
        self._schema_inspector = None

    def accessor(self, community_id: str, community_mode: str):
        if community_mode == "mapping" and self.graph.membership_collection is None:
            raise BackendConfigurationError(
                "community_mode='mapping' requires membership fields in "
                "ArangoGraphConfig"
            )
        if community_mode == "property" and self.graph.community_property_field is None:
            raise BackendConfigurationError(
                "community_mode='property' requires community_property_field in "
                "ArangoGraphConfig"
            )
        provenance_collection = self.graph.provenance_edge_collection
        if provenance_collection and not self.db.has_collection(provenance_collection):
            raise BackendConfigurationError(
                f"Configured provenance collection does not exist: {provenance_collection}"
            )
        return ArangoCommunityAccessor(
            self.db,
            community_id=community_id,
            nodes_collection=self.graph.node_collection,
            edges_collection=self.graph.edge_collection,
            relation_property=self.graph.relation_field,
            node_type_property=self.graph.entity_type_field or "",
            weight_property=self.graph.edge_weight_field,
            edge_timestamp_property=self.graph.edge_timestamp_field,
            edge_valid_from_property=self.graph.edge_valid_from_field,
            edge_valid_to_property=self.graph.edge_valid_to_field,
            edge_status_property=self.graph.edge_status_field,
            allowed_edge_statuses=self.graph.allowed_edge_statuses,
            community_mode=community_mode,
            community_property=self.graph.community_property_field or "",
            membership_collection=self.graph.membership_collection or "",
            membership_entity_field=self.graph.membership_entity_field or "",
            membership_community_field=self.graph.membership_community_field or "",
            provenance_edge_collection=provenance_collection,
            provenance_target_collections=list(self.graph.provenance_target_collections),
            edge_provenance_fields=list(self.graph.edge_provenance_fields),
            bridge_collection=self.graph.bridge_collection,
            affinity_collection=self.graph.affinity_collection,
            algorithm=self.graph.community_algorithm,
            membership_algorithm_field=self.graph.membership_algorithm_field,
            bridge_entity_field=self.graph.bridge_entity_field,
            bridge_strength_field=self.graph.bridge_strength_field,
            bridge_community_field=self.graph.bridge_community_field,
            bridge_algorithm_field=self.graph.bridge_algorithm_field,
            affinity_from_field=self.graph.affinity_from_field,
            affinity_to_field=self.graph.affinity_to_field,
            affinity_score_field=self.graph.affinity_score_field,
            affinity_algorithm_field=self.graph.affinity_algorithm_field,
        )

    def triple_source(self) -> ArangoTripleSource:
        return ArangoTripleSource(self.db, self.graph)

    def model_store(self, community_id: str, community_mode: str) -> ArangoModelStore:
        namespace = json.dumps([
            self.db.name, "global-training",
            self.graph.node_collection, self.graph.edge_collection,
            self.graph.relation_field, self.graph.entity_type_field,
        ], ensure_ascii=False, separators=(",", ":"))
        return ArangoModelStore(self.db, namespace=namespace)

    def global_accessor(self):
        if self.graph.bridge_collection is None and self.graph.affinity_collection is None:
            return None
        # Global exploration uses the same evidence mappings as scoped retrieval.
        return self.accessor("global", "none")

    def schema_inspector(self):
        from odin.backends.arango_schema import ArangoSchemaInspector

        if self._schema_inspector is None:
            self._schema_inspector = ArangoSchemaInspector(self.db)
        return self._schema_inspector
