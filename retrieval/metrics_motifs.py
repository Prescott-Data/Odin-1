from __future__ import annotations
from typing import Iterable, Tuple, Dict, Set

from .adapters import GraphAccessor, NodeId, RelId


def wedge_and_triad_closures(
    accessor: GraphAccessor,
    community_id: str,
    nodes: Iterable[NodeId],
    relation_filter: Set[RelId] | None = None,
    hop_cap: int = 3,
) -> Dict[str, float]:
    """
    Estimate tiny-link yield via wedge/triad closures within a hop cap.
    Returns fraction of wedges that close (triangles) and count estimates.
    """
    nodes = list(nodes)
    if not nodes:
        return {"wedges": 0, "triads": 0, "closure_rate": 0.0}
    wedges = 0
    triads = 0
    for u in nodes:
        nbrs1 = [v for v, r, _ in accessor.iter_out(u) if (not relation_filter or r in relation_filter)]
        for v in nbrs1:
            nbrs2 = [w for w, r, _ in accessor.iter_out(v) if w != u and (not relation_filter or r in relation_filter)]
            for w in nbrs2:
                wedges += 1
                # Closure if an edge from u to w exists (any relation in filter)
                closed = any((x == w and (not relation_filter or r in relation_filter)) for x, r, _ in accessor.iter_out(u))
                if closed:
                    triads += 1
    rate = (triads / wedges) if wedges else 0.0
    return {"wedges": wedges, "triads": triads, "closure_rate": rate}


