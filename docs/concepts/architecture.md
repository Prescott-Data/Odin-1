---
icon: material/hexagon-multiple
---

# Architecture

Odin is a single library assembled from a few well-separated parts. You only ever touch one of them directly: [`OdinEngine`](../reference/engine.md), the public entry point. Understanding how it fits together makes the parameters, the result shape, and the performance characteristics all make sense.

## The pipeline

A call to `retrieve()` runs four stages in order, each feeding the next:

```
┌─────────────┐    ┌──────────────────────────────────────┐    ┌─────────────────┐
│ Seed        │ -> │ PPR → Beam Search → NPLL → Aggregate │ -> │ Ranked Paths +  │
│ Entities    │    │            (Odin Engine)             │    │ Motifs + Scores │
└─────────────┘    └──────────────────────────────────────┘    └─────────────────┘
```

[**PPR**](ppr.md) goes first, scoring nodes by structural importance relative to your seeds and handing back the anchors worth exploring. [**Beam search**](beam-search.md) grows multi-hop paths outward from those anchors, keeping only the top-K at each hop so the search stays bounded. As it goes, [**NPLL**](npll.md) scores each candidate edge for plausibility and prunes the nonsense. Finally, [**aggregation**](aggregation.md) folds the survivors into motifs, relation shares, and a triage score. Each stage exists to make the next one tractable: PPR shrinks where beam search looks, NPLL keeps the beam clean, and aggregation turns raw paths into something an agent can act on.

## The pieces that do the work

Behind `OdinEngine`, the `RetrievalOrchestrator` runs that pipeline and assembles the result. Everything else is a component it coordinates:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              OdinEngine                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    RetrievalOrchestrator                             │ │
│  │  Coordinates PPR → Beam → NPLL → Aggregation and builds the result  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
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

## What happens when you build an engine

Constructing an `OdinEngine` wires those parts together in one shot. It opens an `ArangoCommunityAccessor` for your `community_id` and wraps it in a `CachedGraphAccessor`; it initializes intelligence by loading or training the NPLL model (unless `auto_train=False`), yielding either `NPLLConfidence` or the `ConstantConfidence` fallback; and it hands both to a fresh `RetrievalOrchestrator` and `APPRAnchors` engine. From then on the engine is ready to serve retrievals.

```python
engine = OdinEngine(
    db=db,
    community_id="global",   # scope of exploration
    cache_size=5000,          # graph-accessor LRU size
    auto_train=True,          # train NPLL if no model exists
    community_mode="none",    # "none" = global, "mapping" = scoped
)
```

Every one of these parameters is documented in the [OdinEngine API](../reference/engine.md).

## One deliberate boundary

The design keeps **graph intelligence** and **language reasoning** strictly apart. Odin returns structured, scored evidence (nodes, edges, motifs, numbers) and stops there; the agent, an LLM, interprets that evidence and decides what to do. Drawing the line here buys two things: the graph layer stays explainable and testable, and hallucination is kept out of it entirely. Every path Odin returns genuinely exists in your data, because Odin never invents relationships; it only ranks the ones that are already there.

---

From here, follow the pipeline in order, [Data Model](data-model.md) then [PPR](ppr.md), or skip to the [Result Schema](../reference/result-schema.md) for the exact shape `retrieve()` hands back.
