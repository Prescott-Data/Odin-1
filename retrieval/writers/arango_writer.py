from __future__ import annotations
from typing import Dict, Any, Optional

from .base import PersistenceWriter
from odin.backends.arango import ArangoGraphConfig
from odin.backends.base import BackendConfigurationError, BackendIOError
from copy import deepcopy


class ArangoWriter(PersistenceWriter):
    """Persist links through a connected database and explicit field mappings."""

    def __init__(self, db, graph: ArangoGraphConfig, *, confidence_field: str,
                 metadata_field: str, persist_threshold: float = 0.8):
        fields = (confidence_field, metadata_field)
        if any(not isinstance(field, str) or not field for field in fields):
            raise BackendConfigurationError("Writer confidence and metadata fields are required")
        reserved = {"_from", "_to", "_id", "_key", "_rev", graph.relation_field}
        if len(set(fields)) != len(fields) or any(field in reserved for field in fields):
            raise BackendConfigurationError("Writer fields must be distinct from graph identities")
        self.db = db
        self.graph = graph
        self.confidence_field = confidence_field
        self.metadata_field = metadata_field
        self.persist_threshold = persist_threshold

    def maybe_write_link(self, src_entity: str, rel: str, dst_entity: str,
                         confidence: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if confidence < self.persist_threshold:
            return False
        if any(not isinstance(value, str) or "/" not in value or
               not all(value.split("/", 1)) for value in (src_entity, dst_entity)):
            raise BackendConfigurationError("Writer endpoints must be full Arango document IDs")
        if not isinstance(rel, str) or not rel:
            raise BackendConfigurationError("Writer relation must be a non-empty string")
        document = {
            "_from": src_entity, "_to": dst_entity,
            self.graph.relation_field: rel,
            self.confidence_field: float(confidence),
            self.metadata_field: deepcopy(metadata) if metadata is not None else {},
        }
        try:
            self.db.collection(self.graph.edge_collection).insert(document)
        except Exception as exc:
            raise BackendIOError("Could not persist configured Arango link") from exc
        return True
