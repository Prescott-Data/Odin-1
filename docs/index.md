---
icon: material/home
---

<div class="jc-hero" markdown>

<img src="assets/combo-brand.svg" class="jc-hero-logo" alt="Odin Logo" />

# Graph intelligence for autonomous AI agents

Odin is a Python library that guides AI agents through large knowledge graphs. It combines Personalized PageRank, learned edge-plausibility scoring (NPLL), and pattern detection to surface high-signal paths, so agents spend their compute on what matters.

<div class="jc-cta-text" markdown>
[Get started](getting-started.md) [Concepts](concepts/index.md) [API Reference](reference/engine.md)
</div>

</div>

---

## What Odin provides

<div class="jc-grid" markdown>
<div class="jc-card" markdown>
<span class="jc-card-label">Navigation</span>

### Guided exploration

Odin is the **compass**, not the explorer. Given seed entities, it returns ranked, scored paths through the graph, leaving the agent to interpret what it finds instead of drowning in raw traversal.
</div>

<div class="jc-card" markdown>
<span class="jc-card-label">Structure</span>

### Personalized PageRank

PPR identifies the structurally important nodes relative to your seeds, so exploration starts from the parts of the graph that actually matter.

[PPR concept →](concepts/ppr.md)
</div>

<div class="jc-card" markdown>
<span class="jc-card-label">Semantics</span>

### NPLL edge scoring

Neural Probabilistic Logic Learning scores how plausible each edge is, filtering semantically invalid paths that naive traversal would follow.

[NPLL concept →](concepts/npll.md)
</div>

<div class="jc-card" markdown>
<span class="jc-card-label">Search</span>

### Beam search

A bounded, best-first walk keeps multi-hop exploration tractable: top-K paths at each hop instead of exponential blow-up.

[Beam search →](concepts/beam-search.md)
</div>

<div class="jc-card" markdown>
<span class="jc-card-label">Patterns</span>

### Motifs & triage

Aggregation surfaces recurring motifs and produces a 0-100 triage score, giving agents a single prioritization signal per retrieval.

[Aggregation →](concepts/aggregation.md)
</div>

<div class="jc-card" markdown>
<span class="jc-card-label">Zero-ops ML</span>

### Self-managing model

Odin trains its NPLL model from your graph on first run and persists the weights in ArangoDB. No separate ML pipeline, no `.pt` files to manage.

[Model lifecycle →](guides/npll-lifecycle.md)
</div>
</div>

---

## Quickstart

```bash title="Install"
pip install odin-engine
```

```python title="explore.py"
from arango import ArangoClient
from odin import OdinEngine

# 1. Connect to your knowledge graph
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("my_graph", username="root", password="")

# 2. Initialize Odin (auto-trains NPLL from your graph on first run)
engine = OdinEngine(db=db, community_id="global")

# 3. Explore from seed entities
result = engine.retrieve(
    seeds=["entity/claim_123", "entity/provider_456"],
    max_paths=50,
    hop_limit=3,
)

# 4. Read the ranked paths
print(f"Triage score: {result['triage']['score']}/100")
for path in result["paths"][:5]:
    nodes = " → ".join(str(n) for n in path["nodes"])
    print(f"  [{path['score']:.2f}] {nodes}")
```

---

## Where to start

If you are new to Odin, read in this order:

1. [Getting Started](getting-started.md): install, connect ArangoDB, and run your first retrieval
2. [Architecture](concepts/architecture.md): the mental model for how the pipeline fits together
3. [Personalized PageRank](concepts/ppr.md) → [Beam Search](concepts/beam-search.md) → [NPLL](concepts/npll.md): the three scoring signals
4. [Triage & Insight Scoring](concepts/scoring.md): how paths become a single prioritization number

If you are evaluating for a specific use case:

- [AI Agent Integration](guides/agent-integration.md): wire Odin into an agent loop
- [Healthcare Fraud Detection](examples/healthcare-fraud.md): a worked end-to-end example
- [OdinEngine API](reference/engine.md): the full method surface
- [Configuration](reference/configuration.md): every parameter and its default

---

## Explore the ecosystem

| | |
|---|---|
| **Reference** | Full API surface, parameters, and result schema. [View reference](reference/engine.md) |
| **Source** | Browse the code, open issues, and submit PRs. [GitHub](https://github.com/Prescott-Data/Odin-1){ target="_blank" rel="noopener" } |
| **PyPI** | Install the released package: [odin-engine](https://pypi.org/project/odin-engine/){ target="_blank" rel="noopener" } |
| **Prescott Data** | The team behind Odin: [prescottdata.io](https://prescottdata.io){ target="_blank" rel="noopener" } |

---

<div class="jc-ecosystem" markdown>
<div class="jc-ecosystem-card" markdown>

### Star us on GitHub

Help more developers discover Odin. Every star makes the project easier to find and keeps it growing.

<div class="jc-cta" markdown>
[Star on GitHub](https://github.com/Prescott-Data/Odin-1){ .jc-btn .jc-btn-github target="_blank" rel="noopener" }
</div>

</div>
<div class="jc-ecosystem-card" markdown>

### Build something

Odin is MIT-licensed and made to be extended: new adapters, aggregators, and PPR variants welcome.

<div class="jc-cta" markdown>
[Read the guides](guides/index.md){ .jc-btn }
</div>

</div>
</div>
