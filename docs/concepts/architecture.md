---
icon: material/hexagon-multiple
---

# Architecture

Odin is a single library assembled from a small number of well-separated components. The public entry point is [`OdinEngine`](../reference/engine.md), which wires everything together and exposes a handful of methods.

---

## The pipeline

A call to `retrieve()` runs four stages in sequence:

```
┌─────────────┐    ┌──────────────────────────────────────┐    ┌─────────────────┐
│ Seed        │ -> │ PPR → Beam Search → NPLL → Aggregate │ -> │ Ranked Paths +  │
│ Entities    │    │            (Odin Engine)             │    │ Motifs + Scores │
└─────────────┘    └──────────────────────────────────────┘    └─────────────────┘
```

1. **PPR** — Personalized PageRank scores nodes by structural importance relative to the seeds, producing anchor nodes to explore from.
2. **Beam Search** — a bounded best-first walk expands the top-K paths at each hop, keeping the search tractable.
3. **NPLL scoring** — each candidate edge is scored for plausibility; implausible paths are down-weighted or dropped.
4. **Aggregation** — the surviving paths are summarized into motifs, relation shares, and a triage score.

---

## Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              OdinEngine                                   │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    RetrievalOrchestrator                             │ │
│  │  Coordinates PPR → Beam → NPLL → Aggregation and builds the result  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Cached Graph │  │ APPRAnchors  │  │  NPLLModel   │  │ Aggregators  │  │
│  │  Accessor    │  │   (PPR)      │  │ (confidence) │  │  (motifs)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

| Component | Responsibility | Source |
|-----------|----------------|--------|
| `OdinEngine` | Public API; wiring and lifecycle | `odin/engine.py` |
| `RetrievalOrchestrator` | Runs the pipeline, assembles the result | `retrieval/orchestrator.py` |
| `ArangoCommunityAccessor` | Reads nodes/edges from ArangoDB | `retrieval/adapters_arango.py` |
| `CachedGraphAccessor` | LRU cache over graph access | `retrieval/cache.py` |
| `APPRAnchors` | Personalized PageRank anchors | `retrieval/ppr/` |
| `NPLLConfidence` / `ConstantConfidence` | Edge plausibility scoring | `retrieval/confidence.py` |
| `NPLLModel` / `KnowledgeBootstrapper` | Model and its lifecycle | `npll/` |
| aggregators | Motifs, relation shares, triage | `retrieval/aggregators.py` |

---

## How the engine is wired

When you construct an `OdinEngine`, it:

1. Builds an `ArangoCommunityAccessor` for the given `community_id` and wraps it in a `CachedGraphAccessor`.
2. Initializes **intelligence** — loads or trains the NPLL model via `KnowledgeBootstrapper` (unless `auto_train=False`), producing either `NPLLConfidence` or a `ConstantConfidence` fallback.
3. Creates the `RetrievalOrchestrator` with the accessor and the confidence function.
4. Creates the `APPRAnchors` engine for PPR queries.

```python
engine = OdinEngine(
    db=db,
    community_id="global",   # scope of exploration
    cache_size=5000,          # graph-accessor LRU size
    auto_train=True,          # train NPLL if no model exists
    community_mode="none",    # "none" = global, "mapping" = scoped
)
```

See the [OdinEngine API](../reference/engine.md) for every parameter.

---

## Separation of concerns

Odin deliberately keeps **graph intelligence** separate from **language reasoning**:

- Odin returns *structured, scored evidence* — nodes, edges, motifs, and numbers.
- The agent (an LLM) interprets that evidence and decides what to do next.

This boundary keeps Odin explainable and testable, and keeps hallucination out of the graph layer: every path Odin returns actually exists in your data.

---

## Where to go next

- [Data Model](data-model.md) — how Odin reads your graph
- [Personalized PageRank](ppr.md) — the structural signal
- [Result Schema](../reference/result-schema.md) — the exact shape returned by `retrieve()`
