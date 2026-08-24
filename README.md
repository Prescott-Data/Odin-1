<p align="center">
    <img src="https://raw.githubusercontent.com/Prescott-Data/Odin-1/main/brand-assets/svg/combo-brand.svg" alt="Odin" width="260" />
</p>

<h1 align="center">Graph intelligence for AI agents</h1>

Odin is an open-source Python library for navigating connected evidence in
knowledge graphs. Given seed entities, it uses Personalized PageRank, bounded
beam search, learned edge-plausibility scoring, and pattern aggregation to
return ranked, inspectable paths.

Odin navigates and ranks graph evidence. The consuming agent interprets that
evidence, decides what is missing, and chooses the next seed or action.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Prescott-Data/Odin-1/actions/workflows/ci.yml/badge.svg)](https://github.com/Prescott-Data/Odin-1/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-odin--engine-blue)](https://pypi.org/project/odin-engine/)
[![Docs](https://img.shields.io/badge/docs-odin.developers.prescottdata.io-1758F5)](https://odin.developers.prescottdata.io)

> **Odin-1** is the open-source edition of the Odin graph-intelligence engine, published by Prescott Data under the MIT license so the community can build on it.

---

## See Odin Navigate Connected Evidence

This observed run uses Odin `0.3.0` with a synthetic insurance graph containing
66 entities and 190 recorded relationships. Claim 1042 is the selected seed.

<p align="center">
    <img src="https://raw.githubusercontent.com/Prescott-Data/Odin-1/main/docs/assets/demo/odin-demo-full.gif" alt="Animated Odin 0.3.0 retrieval from a raw synthetic insurance graph to ranked, inspectable evidence paths" width="760" />
</p>

The request asked for 12 paths, a 10-hop limit, and a beam width of 32. Odin's
adaptive pass returned 24 ranked paths with an effective 4-hop limit and beam
width of 64. One retained route connects:

```text
Claim 1042
-> submitted_by -> Ana Torres
-> owns_vehicle -> Vehicle V-204
-> serviced_at -> Central Repairs
```

<p align="center">
    <img src="https://raw.githubusercontent.com/Prescott-Data/Odin-1/main/docs/assets/demo/odin-ranked-evidence.png" alt="Odin 0.3.0 ranked path from Claim 1042 to Central Repairs with PPR, NPLL, inactive signals, and three source records" width="760" />
</p>

The interface labels requested and effective bounds, shows inactive signals as
inactive, and traces every edge in the selected path to a source record. The
complete raw result is preserved in the
[canonical run artifact](https://github.com/Prescott-Data/Odin-1/blob/main/docs/assets/demo/odin-retrieve-52e74681-8344-438e-b9ee-4fde3173a24f.json).
This is a deterministic demonstration dataset, not a scale or accuracy
benchmark. Odin ranks the connected evidence; the consuming agent or
investigator interprets it and chooses the next action.

---

## The Problem

AI agents exploring knowledge graphs face three critical challenges:

1. **Exponential Path Growth** - A 3-hop exploration from a single node in a densely connected graph can generate 100K+ paths, most of which are noise
2. **Semantic Invalidity** - Naive traversal follows edges that violate domain logic (e.g., `Patient → diagnosed_by → Medication`)
3. **No Prioritization** - Without ranking, agents waste turns analyzing low-value paths while missing critical patterns

**Traditional approaches fail:**
- **BFS/DFS**: Exponential explosion, no signal filtering
- **Fixed Cypher Queries**: Only finds patterns you already know exist
- **Random Walk**: No convergence guarantees, wasted compute
- **LLM Prompting Alone**: Hallucinates relationships, can't verify graph structure

## The Solution

Odin provides a **guided exploration framework** that acts as a navigation compass for agents:

```
┌─────────────┐    ┌──────────────────────────────────────┐    ┌─────────────────┐
│ Seed        │ -> │ PPR → Beam Search → NPLL → Aggregate │ -> │ Ranked Paths +  │
│ Entities    │    │        (Odin Engine)                 │    │ Patterns + Score│
└─────────────┘    └──────────────────────────────────────┘    └─────────────────┘
```

**Key Components:**

| Component | Purpose | Impact |
|-----------|---------|--------|
| **Personalized PageRank (PPR)** | Identifies structurally important nodes relative to the selected seeds | Directs attention within the current graph context |
| **Beam Search** | Explores a bounded number of paths at each hop | Keeps multi-hop navigation within an explicit budget |
| **NPLL (Neural Probabilistic Logic)** | Scores edge plausibility using patterns learned from the graph | Adds a semantic signal to path ranking |
| **Motif Detection** | Surfaces recurring relationship sequences | Makes repeated graph structure inspectable |
| **Triage Scoring** | Produces a 0-100 prioritization signal | Helps an agent decide what to inspect next |

---

## Quick Start

### Installation

```bash
# From PyPI (recommended)
pip install odin-engine

# From source
git clone https://github.com/Prescott-Data/Odin-1.git
cd Odin-1
pip install -e .
```

**Requirements:**
- Python 3.9+
- ArangoDB 3.10+ with a populated knowledge graph (reference backend)
- PyTorch 2.0+ (for NPLL)

The example below assumes that the referenced entity IDs already exist in your
graph. Follow the [Getting Started guide](https://odin.developers.prescottdata.io/getting-started/)
for local ArangoDB setup and data-model requirements.

### Minimal Integration

```python
from arango import ArangoClient
from odin import OdinEngine

# 1. Connect to your knowledge graph
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("my_database", username="user", password="pass")

# 2. Initialize Odin (auto-trains NPLL from your graph on first run)
engine = OdinEngine(db=db, community_id="my_community")

# 3. Explore from seed entities
result = engine.retrieve(
    seeds=["entity/claim_123", "entity/provider_456"],
    max_paths=50,
    hop_limit=3,
)

# 4. Access scored paths
print(f"Found {len(result['paths'])} paths")
print(f"Triage Score: {result['triage']['score']}/100")

for p in result['paths'][:5]:
    edges = p['edges']
    nodes = [edges[0]['u'], *(e['v'] for e in edges)] if edges else []
    print(f"  [{p['score']:.2f}]", " -> ".join(str(n) for n in nodes))

# 5. NEW in v0.2.0: Inspect your database schema
from odin import SchemaInspector
inspector = SchemaInspector(db)  # Uses same db connection from step 1
schema = inspector.get_schema_map()
print(f"Database: {schema['database_name']}")
print(f"Collections: {len(schema['collections'])}")
```

**Output Example:**
```
Found 47 paths
Triage Score: 87/100
  [0.94] claim_123 → billed_by → provider_456 → flagged_in → audit_07
  [0.89] claim_123 → has_diagnosis → sepsis_dx → rare_in → nursing_home_cluster
  [0.82] provider_456 → prescribed → medication_999 → contraindicated_with → patient_history
```

### Self-Managing Intelligence

Odin automatically manages its NPLL model lifecycle:

1. **First Run**: Extracts edge patterns from your graph and trains the NPLL model
2. **Stores Weights**: Saves learned parameters in ArangoDB collection (`NPLLWeights`)
3. **Subsequent Runs**: Loads weights from the database and rebuilds the model

**No separate ML pipeline, no .pt files, no DevOps overhead.** Just initialize `OdinEngine` and it handles everything.

---

## Core Features

### 1. Intelligent Path Finding

```python
# Start from suspicious entities, let Odin find connections
result = engine.retrieve(
    seeds=["claim/CLM_99285"],
    max_paths=100,
    hop_limit=4,
)

# Returns scored paths with:
# - Structural importance (PPR)
# - Semantic plausibility (NPLL) 
# - Pattern frequency (motif detection)
```

### 2. Edge Plausibility Scoring

```python
# Validate specific relationships
score = engine.score_edge(
    head="entity/patient_001",
    relation="treated_by",
    tail="entity/doctor_smith"
)
# Returns: 0.0 (impossible) to 1.0 (highly plausible)

# Use in agent decision loops
if score > 0.7:
    agent.investigate_further(path)
```

### 3. Anchor Node Discovery

```python
# Find most important nodes in your graph
anchors = engine.find_anchors(
    seeds=["community/insurance_claims"],
    topn=20
)
# Returns top-N nodes by PageRank for targeted exploration
```

### 4. Pattern Detection

```python
# Automatic motif extraction
motifs = result['aggregates']['motifs']
for motif in motifs:
    print(f"{motif['pattern']} spans {motif['edge_count']} edges across {motif['path_count']} paths")

# Example output:
# has_policyholder->rare_in spans 47 edges across 23 paths
# billed_by->flagged_in spans 12 edges across 6 paths
```

### 5. Schema Introspection (v0.2.0+)

```python
from arango import ArangoClient
from odin import SchemaInspector

# Connect to ArangoDB (same as step 1 above)
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("my_database", username="user", password="pass")

# Inspect your ArangoDB schema at runtime
inspector = SchemaInspector(db)
schema = inspector.get_schema_map()

# Get all collections and their fields
for collection in schema['collections']:
    print(f"{collection['name']}: {collection['count']} docs")
    print(f"  Fields: {', '.join(collection['fields'][:5])}")

# Get specific collection info
entities_info = inspector.get_collection_info('ExtractedEntities')
print(f"Entity fields: {entities_info['fields']}")

# Get edge collection relationships
edge_info = inspector.get_edge_info('ExtractedRelationships')
print(f"From: {edge_info['from_collections']}")
print(f"To: {edge_info['to_collections']}")

# Save schema to file for documentation or agent context
from odin import inspect_arango_schema
inspect_arango_schema(db, output_file='schema.json')
```

**Use Cases:**
- **AI Agents**: Provide schema context to agents for writing valid AQL queries
- **Documentation**: Auto-generate database schema documentation
- **Validation**: Verify collection structures across environments
- **Schema Evolution**: Track changes to your graph structure over time

---

## Architecture

Odin is composed of four layers working in concert:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ODIN ENGINE                                    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    RETRIEVAL ORCHESTRATOR                             │  │
│  │  Coordinates PPR → Beam Search → NPLL → Aggregation pipeline         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Graph        │  │ PPR Engine   │  │ NPLL Model   │  │ Aggregators  │   │
│  │ Accessor     │  │ (Anchors)    │  │ (Confidence) │  │ (Motifs)     │   │
│  │ + Cache      │  │              │  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Component Details:**

| Layer | Responsibility |
|-------|----------------|
| **Graph Accessor** | Read and cache graph neighborhoods |
| **PPR Engine** | Compute seed-relative structural importance |
| **Beam Search** | Explore bounded multi-hop paths |
| **NPLL Confidence** | Add learned edge-plausibility signals |
| **Aggregators** | Summarize motifs, relation shares, and triage signals |

---

## Use Cases

### 1. Healthcare Fraud Detection
**Scenario:** Find providers billing unusual procedure combinations

```python
engine = OdinEngine(db, community_id="medicare_claims")
result = engine.retrieve(
    seeds=["provider/high_volume_clinic"],
    max_paths=100,
)
# Odin surfaces: "CPT_99285 + CPT_office_visit" pattern in 34 claims
```

### 2. Supply Chain Risk Analysis
**Scenario:** Identify cascading supplier dependencies

```python
result = engine.retrieve(
    seeds=["supplier/critical_vendor"],
    hop_limit=5,  # Deep supply chain exploration
)
# Discovers: Tier-3 supplier affects 47 downstream products
```

### 3. Regulatory Compliance Checks
**Scenario:** Validate entity relationships against compliance rules

```python
score = engine.score_edge(
    head="entity/investment_fund",
    relation="managed_by",
    tail="entity/sanctioned_entity"
)
if score > 0.5:
    compliance_agent.flag_for_review()
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [**Documentation Site**](https://odin.developers.prescottdata.io) | Full guides, concepts, and API reference |
| [**Architecture**](whitepaper/ARCHITECTURE.md) | Complete technical design |
| [**Agent Integration Guide**](whitepaper/AGENT_INTEGRATION_GUIDE.md) | How to integrate with AI agents |
| [**Whitepaper**](whitepaper/ODIN_WHITEPAPER.md) | Research background and evaluation |

---

## API Reference

### OdinEngine

```python
from odin import OdinEngine

engine = OdinEngine(
    db: StandardDatabase,              # ArangoDB connection
    community_id: str = "global",      # Scope for exploration
    cache_size: int = 5000,            # LRU cache size
    auto_train: bool = True,           # Auto-train NPLL if needed
    community_mode: str = "none"       # "none" | "mapping"
)
```

**Methods:**

#### `retrieve(seeds, max_paths, hop_limit, **kwargs)`
Find and score paths from seed entities.

**Parameters:**
- `seeds: List[str]` - Starting entity IDs (e.g., `["entity/123"]`)
- `max_paths: int = 50` - Maximum paths to return
- `hop_limit: int = 3` - Maximum hops from seeds
- `beam_width: int = 64` - Top-K paths to explore at each hop

**Returns:** a dictionary (see the [Result Schema](https://odin.developers.prescottdata.io/reference/result-schema/) for the full shape):
```python
{
    "topk_ppr": [...],
    "paths": [
        {
            "id": "path_0",
            "score": 0.94,
            "edges": [
                {"u": "entity/A", "v": "entity/B", "relation": "billed_by",
                 "confidence": 0.89, "created_at": "...", "provenance": {...}}
            ]
        }
    ],
    "insight_score": 0.82,
    "aggregates": {
        "motifs": [{"pattern": "billed_by->flagged_in", "edge_count": 12, "path_count": 6}],
        "relation_share": {"billed_by": {"count": 42, "share": 0.42}},
        "summary": {...}
    },
    "triage": {"score": 87, "components": {...}, "dominant_relation": {...}}
}
```

#### `score_edge(src, rel, dst)`
Score plausibility of a single edge.

**Parameters:**
- `src: str` - Source entity ID
- `rel: str` - Relationship type
- `dst: str` - Target entity ID

**Returns:** `float` (0.0-1.0)

#### `find_anchors(seeds, topn)`
Get top-N most important nodes by PageRank.

**Parameters:**
- `seeds: List[str]` - Seed entities for personalized PageRank
- `topn: int = 20` - Number of top nodes to return

**Returns:** `List[Tuple[str, float]]` - (entity_id, ppr_score)

#### `retrain_model(force_retrain=True)`
Force NPLL model retraining (use after major graph updates).

---

## Performance

Runtime depends on graph density, database latency, retrieval bounds, cache
state, and whether NPLL weights are already available. The repository includes
performance and memory regression tests in [`tests/performance`](tests/performance/)
and benchmark utilities in [`benchmarks`](benchmarks/). Publish workload-specific
measurements with the dataset, parameters, hardware, and Odin version used.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage report
pytest tests/ --cov=odin --cov=npll --cov=retrieval --cov-report=html
```

---

## Contributing

We welcome contributions! Areas of interest:

- **Scalability**: Optimizations for graphs >10M entities
- **Algorithms**: Alternative PPR implementations, new aggregators
- **Database Support**: Neo4j, Neptune adapters
- **Benchmarks**: Academic dataset comparisons

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Authors

**Prescott Data**
- Muyukani Kizito - Lead Engineer
- Elizabeth Nyambere - NPLL & GNN Research

---

## Citation

If you use Odin in academic work, please cite:

```bibtex
@software{odin_engine,
  title={Odin: Graph Intelligence for Autonomous AI Agents},
  author={Prescott Data},
  year={2026},
  url={https://github.com/Prescott-Data/Odin-1}
}
```

---

## Links

- [PyPI Package](https://pypi.org/project/odin-engine/)
- [Documentation](https://odin.developers.prescottdata.io)
- [GitHub Repository](https://github.com/Prescott-Data/Odin-1)
- [Prescott Data](https://prescottdata.io)
