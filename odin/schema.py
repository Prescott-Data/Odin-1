"""Backend-neutral schema inspection entry points."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from retrieval.backends.base import BackendCapabilityError, GraphBackend


def inspect_schema(
    backend: GraphBackend,
    *,
    refresh: bool = False,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspect a backend schema and optionally write its map to JSON."""
    factory = getattr(backend, "schema_inspector", None)
    if not callable(factory):
        raise BackendCapabilityError(
            "Backend does not support schema inspection"
        )
    inspector = factory()
    if inspector is None:
        raise BackendCapabilityError(
            "Backend does not support schema inspection"
        )
    schema = inspector.get_schema_map(refresh=refresh)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as stream:
            json.dump(schema, stream, indent=2, default=str)
    return schema
