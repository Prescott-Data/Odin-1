---
icon: material/rocket-launch
---

# Getting Started

This guide takes you from a fresh Python environment to your first scored retrieval with Odin. It covers installation, connecting a graph database, and running the engine end to end.

---

## Requirements

Odin requires **Python 3.9 or later** and a **graph database**. The reference backend is [ArangoDB](https://www.arangodb.com/) 3.10+.

| Dependency | Why |
|------------|-----|
| Python ≥ 3.9 | Runtime |
| ArangoDB ≥ 3.10 | Stores your knowledge graph and Odin's learned NPLL weights |
| PyTorch ≥ 2.0 | Powers the NPLL edge-plausibility model |

PyTorch, `python-arango`, NumPy, scikit-learn, and `gremlinpython` are installed automatically with the package.

---

## Installation

```bash
pip install odin-engine
```

=== "From PyPI"
    ```bash
    pip install odin-engine
    ```

=== "From source"
    ```bash
    git clone https://github.com/Prescott-Data/Odin-1.git
    cd Odin-1
    pip install -e .
    ```

=== "With dev tools"
    ```bash
    pip install -e ".[dev]"   # pytest, flake8, bandit
    ```

!!! note "Import name"
    The package is published as `odin-engine`, but you import it as `odin`:

    ```python
    from odin import OdinEngine, SchemaInspector
    ```

---

## Start ArangoDB

If you do not already have a graph database, the fastest way to get one locally is Docker:

```bash
docker run -d --name arango -p 8529:8529 \
  -e ARANGO_NO_AUTH=1 arangodb:3.12
```

This exposes ArangoDB on `http://localhost:8529` with authentication disabled — fine for local development, never for production. See [Connecting ArangoDB](guides/arangodb.md) for authenticated setups.

---

## Connect and initialize

```python
from arango import ArangoClient
from odin import OdinEngine

client = ArangoClient(hosts="http://localhost:8529")
db = client.db("my_graph", username="root", password="")

engine = OdinEngine(db=db, community_id="global")
```

On the **first** initialization against a graph, Odin extracts edge patterns and trains its NPLL model (typically 2–5 minutes). It then stores the learned weights in an ArangoDB collection, so subsequent runs load in about 30 seconds. See [Model Lifecycle](guides/npll-lifecycle.md).

!!! tip "No model? No problem"
    If training cannot run (for example, an empty graph), Odin falls back to a constant edge-confidence and keeps working. Check `engine.has_npll` to see which mode you are in.

---

## Run your first retrieval

```python
result = engine.retrieve(
    seeds=["entity/claim_123"],
    max_paths=50,
    hop_limit=3,
)

print(f"Triage score: {result['triage']['score']}/100")
print(f"Paths found:  {len(result['paths'])}")

for path in result["paths"][:5]:
    nodes = " → ".join(str(n) for n in path["nodes"])
    print(f"  [{path['score']:.2f}] {nodes}")
```

`retrieve()` runs the full pipeline — **PPR → Beam Search → NPLL scoring → aggregation** — and returns a dictionary of ranked paths, motifs, and scores. The full shape is documented in the [Result Schema](reference/result-schema.md).

---

## Score a single edge

Beyond whole-path retrieval, you can ask Odin how plausible one relationship is:

```python
score = engine.score_edge("entity/patient_001", "treated_by", "entity/doctor_smith")
print(score)   # 0.0 (impossible) … 1.0 (highly plausible)
```

This is the same NPLL signal Odin uses internally, exposed for use inside agent decision loops. See [Scoring Edges](guides/edge-scoring.md).

---

## Find the important nodes

```python
anchors = engine.find_anchors(seeds=["community/insurance_claims"], topn=20)
for node_id, ppr_score in anchors[:10]:
    print(f"{ppr_score:.4f}  {node_id}")
```

`find_anchors()` returns the top-N nodes by Personalized PageRank relative to your seeds — a quick way to locate where to explore. See [Finding Anchors](guides/anchors.md).

---

## Next steps

<div class="grid cards" markdown>

-   :material-database: **Connecting ArangoDB**

    Authenticated connections, communities, and data layout.

    [Read more →](guides/arangodb.md)

-   :material-hexagon-multiple: **Architecture**

    How the retrieval pipeline is assembled.

    [Read more →](concepts/architecture.md)

-   :material-robot: **AI Agent Integration**

    Put Odin inside an agent's reasoning loop.

    [Read more →](guides/agent-integration.md)

-   :material-tune: **Tuning Retrieval**

    Beam width, hop limits, and path budgets.

    [Read more →](guides/tuning.md)

</div>
