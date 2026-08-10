from __future__ import annotations
from typing import Dict, Any, Optional

from .base import PersistenceWriter

class ArangoWriter(PersistenceWriter):
    """
    Implementation of the PersistenceWriter protocol for ArangoDB.
    """
    def __init__(self, arango_client, database: str, persist_threshold: float = 0.8):
        self.client = arango_client
        self.db = self.client.db(database)
        self.persist_threshold = persist_threshold

    def maybe_write_link(self, src_entity: str, rel: str, dst_entity: str, confidence: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if confidence < self.persist_threshold:
            return False
        
        edges = self.db.collection('kg_edges')
        doc = {
            '_from': f'entities/{src_entity}',
            '_to': f'entities/{dst_entity}',
            'relation': rel,
            'confidence': float(confidence),
            'metadata': metadata or {},
        }
        edges.insert(doc)
        return True
