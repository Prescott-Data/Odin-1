---
icon: material/brain
---

# Concepts

The ideas behind Odin: what it does, how the retrieval pipeline is assembled, and how each scoring signal contributes to the final ranking.

Odin's job is narrow and deliberate: **given seed entities, return the most relevant scored paths through a knowledge graph.** It does not answer natural-language questions or generate prose; that is the agent's job. Odin is the *compass*; the agent is the *explorer*.

<div class="grid cards" markdown>

-   :material-hexagon-multiple: **Architecture**

    How PPR, beam search, NPLL, and aggregation compose into one pipeline.

    [Read more →](architecture.md)

-   :material-graph: **Data Model**

    Entities, relationships, and communities: how Odin sees your graph.

    [Read more →](data-model.md)

-   :material-star-four-points: **Personalized PageRank**

    Finding the structurally important nodes relative to your seeds.

    [Read more →](ppr.md)

-   :material-magnify-scan: **Beam Search**

    Bounded, best-first multi-hop path exploration.

    [Read more →](beam-search.md)

-   :material-shield-check: **NPLL Edge Scoring**

    Learned edge plausibility that filters invalid paths.

    [Read more →](npll.md)

-   :material-chart-donut: **Motifs & Aggregation**

    Turning paths into recurring patterns and summaries.

    [Read more →](aggregation.md)

-   :material-numeric: **Triage & Insight Scoring**

    Collapsing many signals into one 0-100 prioritization number.

    [Read more →](scoring.md)

-   :material-cached: **Caching**

    Why graph access is LRU-cached and what it costs.

    [Read more →](caching.md)

</div>

---

## The three signals

Every path Odin returns is scored by combining three complementary signals:

| Signal | Question it answers | Component |
|--------|---------------------|-----------|
| **Structural** | Is this node important in the graph? | [Personalized PageRank](ppr.md) |
| **Reachable** | Can we get there efficiently? | [Beam Search](beam-search.md) |
| **Semantic** | Is this edge plausible? | [NPLL](npll.md) |

No single signal is sufficient. PPR alone finds important nodes but follows nonsensical edges. Beam search alone explodes without a scoring signal to prune. NPLL alone validates edges but has no notion of importance. Together they let an agent focus on paths that are *important, reachable, and plausible*.
