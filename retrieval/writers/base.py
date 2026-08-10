from __future__ import annotations
from typing import Protocol, Dict, Any, Optional

class PersistenceWriter(Protocol):
    """
    Protocol for a generic persistence writer.
    Defines the interface for writing links and relations to a database.
    """
    def maybe_write_link(
        self, 
        src_entity: str, 
        rel: str, 
        dst_entity: str, 
        confidence: float, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Writes a link to the database if the confidence meets the threshold.
        Returns True if the link was written, False otherwise.
        """
        ...
