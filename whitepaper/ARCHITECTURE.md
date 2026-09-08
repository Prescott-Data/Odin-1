# Odin KG Engine - Architecture & Technical Documentation

**Version:** 1.0  
**Last Updated:** 2026-01-12  
**Status:** Production-Ready for Medium-Scale Deployments (<5M entities)

---

## Table of Contents

1. [What is Odin?](#what-is-odin)
2. [The Problem It Solves](#the-problem-it-solves)
3. [Architecture Overview](#architecture-overview)
4. [Core Components](#core-components)
5. [How It Works](#how-it-works)
6. [The Retrieval Pipeline](#the-retrieval-pipeline)
7. [NPLL: The Confidence Layer](#npll-the-confidence-layer)
8. [Value Proposition](#value-proposition)
9. [Comparison with Alternatives](#comparison-with-alternatives)
10. [Current Limitations](#current-limitations)
11. [API Reference](#api-reference)
12. [Integration Guide](#integration-guide)

---

## What is Odin?

Odin is a **graph intelligence engine** that helps AI agents navigate knowledge graphs to find meaningful patterns. It combines:

- **Graph algorithms** (PageRank, Beam Search) for structural importance
- **Probabilistic logic** (NPLL) for relationship confidence
- **Aggregation** (Motifs, Triage Scoring) for pattern detection

**Odin is NOT:**
- A knowledge graph database (use ArangoDB, Neo4j for storage)
- An LLM (use GPT-4, Claude for reasoning)
- A complete RAG system (it's the retrieval layer)

**Odin IS:**
- The "intelligence layer" between your KG and your agents
- A compass that tells agents WHERE to look in the graph
- A scoring system that ranks paths by importance and plausibility

---

## The Problem It Solves

### The Challenge: Needle in a Haystack

You have a knowledge graph with:
- 100K+ entities (patients, claims, diagnoses, providers...)
- 1M+ relationships (diagnosed_with, prescribed, billed_by...)
- Rich text, tables, and images as source documents

An AI agent needs to find **insights** like:
- "34 patients from nursing homes have sepsis" (pattern)
- "Provider X handles 40% of claims" (anomaly)
- "Drug interaction risk for 7 patients" (safety issue)

### Why Naive Approaches Fail

**Approach 1: Random Walk**
```
Start at random entity → follow random edges → hope to find something
Result: Random paths, no signal, wasted compute
```

**Approach 2: BFS/DFS**
```
Start at entity → explore ALL neighbors → explore their neighbors...
Result: Exponential explosion, most paths are noise
```

**Approach 3: Cypher/AQL Query**
```
"MATCH pattern WHERE conditions RETURN results"
Result: Only finds patterns you already know to look for
```

### Odin's Approach: Intelligent Guided Exploration

```
Start at seeds → Score neighbors by (importance + plausibility) → 
Follow BEST paths → Aggregate patterns → Rank by significance
```

Odin doesn't randomly explore. It uses **structural importance** (PageRank) and **semantic plausibility** (NPLL) to focus on paths that matter.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ODIN KG ENGINE                                     │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         RETRIEVAL ORCHESTRATOR                          │    │
│  │  The main entry point. Coordinates all components.                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│         │                    │                    │                    │        │
│         ▼                    ▼                    ▼                    ▼        │
│  ┌───────────┐        ┌───────────┐        ┌───────────┐        ┌───────────┐   │
│  │    PPR    │        │   BEAM    │        │  SCORING  │        │AGGREGATORS│   │
│  │  Engines  │        │  SEARCH   │        │           │        │           │   │
│  │           │        │           │        │           │        │           │   │
│  │ -Push PPR │        │ Multi-hop │        │ -Path     │        │ -Motifs   │   │
│  │ -MC PPR   │        │ path      │        │ -Insight  │        │ -Triage   │   │
│  │ -BiPPR    │        │ finding   │        │ -Evidence │        │ -Baseline │   │
│  └───────────┘        └───────────┘        └───────────┘        └───────────┘   │
│         │                    │                                                  │
│         │                    ▼                                                  │
│         │             ┌───────────┐                                             │
│         │             │   NPLL    │                                             │
│         │             │ Confidence│                                             │
│         │             │           │                                             │
│         │             │ Scores    │                                             │
│         │             │ edge      │                                             │
│         │             │ plausibility│                                           │
│         │             └───────────┘                                             │
│         │                    │                                                  │
│         ▼                    ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        GRAPH ACCESSOR                                   │    │
│  │  Abstraction layer for graph database access                            │    │
│  │  (ArangoDB, Neo4j, or in-memory)                                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     KNOWLEDGE GRAPH (ArangoDB)                          │    │
│  │  ExtractedEntities | ExtractedRelationships | Documents | Communities   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Retrieval Orchestrator

**File:** `retrieval/orchestrator.py`

The main entry point. Coordinates the entire retrieval pipeline.

```python
from odin_kg_engine.retrieval import RetrievalOrchestrator, OrchestratorParams

orchestrator = RetrievalOrchestrator(
    accessor=graph_accessor,      # How to access the graph
    edge_confidence=npll_confidence,  # How to score edges
)

result = orchestrator.retrieve(
    seeds=["entity/claim_123"],   # Starting points
    params=OrchestratorParams(
        community_id="insurance_claims",
        hop_limit=3,              # How deep to explore
        beam_width=64,            # How many candidates per hop
        max_paths=200,            # Maximum paths to return
    )
)
```

**What it returns:**
```python
{
    "topk_ppr": [(node, score), ...],  # Important nodes
    "paths": [...],                     # Multi-hop paths
    "aggregates": {
        "motifs": [...],               # Recurring patterns
        "relation_share": {...},       # Relationship distribution
        "snippet_anchors": [...],      # Evidence locations
    },
    "triage": {"score": 73},           # 0-100 importance
    "insight_score": 0.68,             # 0-1 quality
}
```

---

### 2. PPR Engines (PageRank)

**Files:** `retrieval/ppr/engines.py`, `retrieval/ppr/bippr.py`

Personalized PageRank identifies **structurally important** nodes starting from seed nodes.

**Three implementations:**

| Engine | Method | Best For |
|--------|--------|----------|
| `PushPPREngine` | Deterministic local approximation | Fast, local exploration |
| `MonteCarloPPREngine` | Random walks with sampling | Global coverage |
| `BiPPREngine` | Bidirectional (source ↔ targets) | Target-aware scoring |

**How it works:**
```
Seeds: [A, B]
        │
        ▼
PPR computes "importance" of all reachable nodes:
- Nodes close to seeds get higher scores
- Nodes with many incoming edges get higher scores
- Nodes that are "hubs" get higher scores

Output: [(node_X, 0.12), (node_Y, 0.09), (node_Z, 0.07), ...]
```

**Why it matters:**
Instead of exploring randomly, the agent knows which nodes are likely to be important.

---

### 3. Beam Search

**File:** `retrieval/beam.py`

Finds multi-hop paths through the graph using a scoring function.

**The scoring formula (per edge):**
```
score = λ₁ * log(PPR_score)      # Structural importance
      + λ₂ * log(edge_prior)     # How common is this relation?
      + λ₃ * log(NPLL_conf)      # How plausible is this edge?
      + λ₄ * log(recency)        # How recent is this edge?

Default lambdas: (0.6, 0.2, 0.15, 0.05)
```

**How it works:**
```
Hop 1: Start at seeds, score all neighbor edges
       Keep top beam_width (e.g., 64) candidates

Hop 2: From each candidate, score their neighbors
       Keep top beam_width candidates

Hop 3: Repeat...

Output: Top max_paths (e.g., 200) highest-scoring paths
```

**Why it matters:**
Beam search prevents exponential explosion while finding the BEST paths, not just ANY paths.

---

### 4. NPLL Confidence

**Files:** `retrieval/confidence.py`, `npll/`

Neural Probabilistic Logic Learning scores edge **plausibility**.

**What NPLL does:**
```
Input:  (Patient_4521, prescribed, Warfarin)
        (Patient_4521, prescribed, Ibuprofen)

NPLL asks: "Given the patterns in the graph, how plausible is each edge?"

Output: 
  (Patient_4521, prescribed, Warfarin) → 0.92 (plausible)
  (Patient_4521, prescribed, Ibuprofen) + Warfarin → 0.34 (implausible - contraindication!)
```

**How it's trained:**
1. Learn embeddings for entities and relations
2. Learn logical patterns (e.g., "friendOf(X,Y) ∧ likes(X,Z) → likes(Y,Z)")
3. Score new edges based on learned patterns

**Why it matters:**
NPLL acts as a **bullshit detector**. It filters out implausible edges during traversal, keeping only semantically meaningful paths.

---

### 5. Aggregators

**File:** `retrieval/aggregators.py`

Aggregators analyze retrieved paths to extract patterns.

**What they compute:**

| Aggregator | What it finds | Example |
|------------|---------------|---------|
| `extract_motifs()` | Recurring relationship sequences | "has_claim→billed_by" appears 47 times |
| `calculate_relation_shares()` | Distribution of relations | "billed_by" is 34% of edges |
| `compute_triage_score()` | 0-100 importance score | 73/100 based on provenance, recency, surprise |
| `extract_snippet_anchors()` | Pointers to source documents | doc_123, page 2, chars 120-180 |

**Why it matters:**
Raw paths are hard to interpret. Aggregators distill them into actionable patterns.

---

### 6. Graph Accessor

**Files:** `retrieval/adapters.py`, `retrieval/adapters_arango.py`

Abstraction layer for graph database access.

**Interface:**
```python
class GraphAccessor(Protocol):
    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]: ...
    def iter_in(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]: ...
    def nodes(self, community_id: str) -> Iterable[NodeId]: ...
```

**Implementations:**
- `ArangoCommunityAccessor` - For ArangoDB
- `KGCommunityAccessor` - For in-memory KGs
- `CachedGraphAccessor` - LRU caching wrapper (production optimization)

---

## How It Works

### The Complete Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ODIN RETRIEVAL PIPELINE                                 │
│                                                                                 │
│  INPUT: seeds = ["entity/claim_123"]                                            │
│         community_id = "insurance_claims"                                       │
│                                                                                 │
│  STEP 1: PPR (PageRank)                                                         │
│  ════════════════════════                                                       │
│  • Push PPR: Local deterministic approximation                                  │
│  • Monte Carlo PPR: Random walks for global coverage                            │
│  • Combine: Weighted sum of both                                                │
│  • Output: Top-k important nodes                                                │
│                                                                                 │
│  STEP 2: BEAM SEARCH                                                            │
│  ════════════════════════                                                       │
│  For each hop (1 to hop_limit):                                                 │
│    • For each current path:                                                     │
│      • Get neighbors (accessor.iter_out)                                        │
│      • Score each edge: PPR + NPLL + Recency + Prior                            │
│      • Keep top beam_width extensions                                           │
│    • Collect paths into best_paths                                              │
│  • Output: Top max_paths multi-hop paths                                        │
│                                                                                 │
│  STEP 3: SCORING                                                                │
│  ════════════════════════                                                       │
│  • Path scores: Combined edge scores                                            │
│  • Evidence strength: How well-sourced are the paths?                           │
│  • Community relevance: How central to the community?                           │
│  • Insight score: Overall quality (0-1)                                         │
│                                                                                 │
│  STEP 4: AGGREGATION                                                            │
│  ════════════════════════                                                       │
│  • Motifs: Recurring relationship patterns                                      │
│  • Relation shares: Distribution of edge types                                  │
│  • Snippet anchors: Document evidence locations                                 │
│  • Triage score: 0-100 importance                                               │
│                                                                                 │
│  OUTPUT:                                                                        │
│  {                                                                              │
│    "topk_ppr": [...],         // Important nodes                                │
│    "paths": [...],            // Multi-hop paths with scores                    │
│    "aggregates": {...},       // Motifs, relations, anchors                     │
│    "triage": {"score": 73},   // Importance 0-100                               │
│    "insight_score": 0.68      // Quality 0-1                                    │
│  }                                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## The Retrieval Pipeline

### Detailed Execution Trace

```python
# 1. INITIALIZATION
orchestrator = RetrievalOrchestrator(
    accessor=ArangoCommunityAccessor(db, community_id="insurance"),
    edge_confidence=NPLLConfidence(npll_model),
)

# 2. CALL RETRIEVE
result = orchestrator.retrieve(
    seeds=["entity/claim_123"],
    params=OrchestratorParams(
        community_id="insurance",
        hop_limit=3,
        beam_width=64,
        max_paths=200,
    )
)

# 3. INTERNAL EXECUTION

## 3a. Personalization Vector
personalization = {"entity/claim_123": 1.0}  # Uniform over seeds

## 3b. PPR Engines
push_result = PushPPREngine.run(seeds, params)    # Deterministic
mc_result = MonteCarloPPREngine.run(seeds, params) # Random walks
combined = merge(push_result, mc_result)           # Weighted combo

## 3c. Beam Search
for hop in range(1, 4):  # 3 hops
    for path in current_paths:
        for neighbor in accessor.iter_out(path.last_node):
            score = score_extension(neighbor, ppr, npll, recency, prior)
            candidates.append((score, extended_path))
    current_paths = top_k(candidates, beam_width)
    best_paths.extend(current_paths)

## 3d. Scoring
for path in best_paths:
    path.score = compute_path_score(path, ppr, npll)
insight_score = compute_insight_score(best_paths, evidence)

## 3e. Aggregation
motifs = extract_motifs(best_paths)
relations = calculate_relation_shares(best_paths)
anchors = extract_snippet_anchors(best_paths)
triage = compute_triage_score(provenance, recency, surprise, motif_density)

# 4. RETURN RESULT
return {
    "topk_ppr": combined.scores,
    "paths": best_paths,
    "aggregates": {"motifs": motifs, "relation_share": relations, ...},
    "triage": {"score": triage},
    "insight_score": insight_score,
}
```

---

## NPLL: The Confidence Layer

### What is NPLL?

NPLL (Neural Probabilistic Logic Learning) combines:
- **Neural embeddings** (like TransE, RotatE) for entity/relation representations
- **Logical rules** (like Markov Logic Networks) for structured reasoning
- **Probabilistic inference** for uncertainty quantification

### How NPLL Scores Edges

```
INPUT: Triple (head, relation, tail)
       e.g., (Patient_4521, prescribed, Warfarin)

STEP 1: Get Embeddings
  head_emb = embedding_manager.get_entity_embedding("Patient_4521")
  rel_emb = embedding_manager.get_relation_embedding("prescribed")
  tail_emb = embedding_manager.get_entity_embedding("Warfarin")

STEP 2: Compute Score
  score = bilinear_scoring(head_emb, rel_emb, tail_emb)
  # score = head_emb ᵀ · diag(rel_emb) · tail_emb

STEP 3: Convert to Probability
  probability = sigmoid(score * temperature)
  # Returns value in [0, 1]

OUTPUT: 0.92 (high confidence - this prescription makes sense)
```

### Why NPLL Matters

Without NPLL, beam search would follow ANY edge. With NPLL:

| Edge | NPLL Score | Action |
|------|------------|--------|
| (Patient, prescribed, Warfarin) | 0.92 | ✅ Follow |
| (Patient, prescribed, Ibuprofen) + Warfarin | 0.34 | ⚠️ Flag as risky |
| (Patient, diagnosed, Pregnancy) [78M] | 0.12 | ❌ Skip (implausible) |
| (Claim, processed_by, VetClinic) | 0.08 | ❌ Skip (wrong domain) |

NPLL acts as a **semantic filter**, ensuring paths are not just structurally connected but also **make sense**.

---

## Value Proposition

### What Odin Provides

| Capability | What It Means | Business Value |
|------------|---------------|----------------|
| **Guided Exploration** | Agents don't wander randomly | 10x faster insight discovery |
| **Semantic Filtering** | NPLL removes implausible edges | Higher quality insights |
| **Pattern Detection** | Motifs surface recurring structures | Automated anomaly detection |
| **Triage Scoring** | 0-100 importance ranking | Prioritized alerts |
| **Evidence Anchoring** | Points to source documents | Explainable, auditable insights |

### Realistic Expectations

**Odin WILL:**
- ✅ Find structurally important nodes (PageRank works)
- ✅ Filter implausible paths (NPLL works for trained domains)
- ✅ Surface recurring patterns (Motifs work)
- ✅ Provide ranked results (Scoring works)
- ✅ Scale to medium graphs (<5M entities)

**Odin WON'T:**
- ❌ Understand natural language (use LLM for that)
- ❌ Generate insights automatically (use LLM for that)
- ❌ Explain WHY a pattern matters (use LLM for that)
- ❌ Scale to 100M+ entities without optimization
- ❌ Work well on sparse/disconnected graphs

### The Right Mental Model

```
ODIN is the COMPASS, not the EXPLORER.

COMPASS: "The gold is 200 meters northwest, confidence 73%"
EXPLORER: "I found the gold and here's what it means for our business"

Odin tells you WHERE. The LLM agent tells you WHAT IT MEANS.
```

---

## Comparison with Alternatives

### vs. Traditional Graph RAG

| Aspect | Traditional Graph RAG | Odin |
|--------|----------------------|------|
| Query type | Fixed patterns (Cypher/AQL) | Dynamic exploration |
| Discovery | Manual query design | Autonomous |
| Ranking | Basic (by result count) | Multi-signal (PPR + NPLL + Recency) |
| Patterns | Must know what to look for | Surfaces unknown patterns |
| Confidence | Binary (edge exists or not) | Probabilistic (0-1) |

**Verdict:** Traditional RAG is for known-question answering. Odin is for unknown-pattern discovery.

### vs. Vector RAG

| Aspect | Vector RAG | Odin |
|--------|------------|------|
| Data structure | Flat document chunks | Graph with relationships |
| Retrieval | Semantic similarity | Structural importance + Logic |
| Multi-hop | Limited (reranking) | Native (beam search) |
| Relationships | Implicit in embeddings | Explicit and typed |
| Explainability | "Similar to query" | "Path A→B→C with confidence X" |

**Verdict:** Vector RAG is for document Q&A. Odin is for relationship-aware discovery.

### vs. LLM-Only Agents

| Aspect | LLM-Only | LLM + Odin |
|--------|----------|------------|
| Context | Limited to context window | Full graph via guided retrieval |
| Hallucination | High risk | Reduced (evidence-grounded) |
| Efficiency | Expensive (large prompts) | Efficient (targeted retrieval) |
| Consistency | Variable | Reproducible (same seeds = same paths) |

**Verdict:** LLM-only is expensive and hallucinates. Odin grounds the LLM in graph evidence.

### vs. Pure GNN Approaches

| Aspect | Pure GNN | Odin (PPR + NPLL) |
|--------|----------|-------------------|
| Training | Requires labeled data | Can use unsupervised |
| Inference | End-to-end neural | Interpretable pipeline |
| Explainability | Black box | Clear path traces |
| Flexibility | Fixed architecture | Modular components |

**Verdict:** GNNs are powerful but opaque. Odin is interpretable and modular.

---

## Current Limitations

### 1. Scale Limits

| Component | Current Limit | Reason |
|-----------|---------------|--------|
| Total entities | <5M | Embedding RAM |
| Community size | <500K | PPR fallback loads all nodes |
| Node degree | <10K | First-fetch latency |

**Mitigation:** Lazy embedding loading, streaming access (future work).

### 2. Cold Start

- NPLL requires **training data** to learn patterns
- New domains need **model fine-tuning**
- Empty graphs return empty results

**Mitigation:** Pre-trained domain models, graceful fallback to ConstantConfidence.

### 3. Sparse Graphs

- PPR works poorly on disconnected components
- Beam search can't find paths that don't exist
- Isolated nodes are never discovered

**Mitigation:** Multiple seed selection strategies, community detection preprocessing.

### 4. Real-Time Latency

- Full retrieval takes 1-5 seconds (medium graph)
- Not suitable for <100ms response requirements
- First request is slower (cache cold)

**Mitigation:** Warm caching, pre-computed results for common queries.

---

## API Reference

### RetrievalOrchestrator

```python
class RetrievalOrchestrator:
    def __init__(
        self,
        accessor: GraphAccessor,
        ppr_cache: Optional[PPRCache] = None,
        edge_confidence: Optional[EdgeConfidenceProvider] = None,
        walk_index: Optional[RandomWalkIndex] = None,
        redact_pii_in_trace: bool = True,
    ): ...
    
    def retrieve(
        self,
        seeds: List[NodeId],
        params: OrchestratorParams,
        budget: Optional[SearchBudget] = None,
        now_ts: Optional[float] = None,
        include_baseline: bool = False,
        anchor_prior: Optional[Dict[NodeId, float]] = None,
        beam_override: Optional[BeamParams] = None,
    ) -> Dict[str, object]: ...
    
    def link_and_retrieve(
        self,
        mentions: List[Mention],
        base_accessor: GraphAccessor,
        params: OrchestratorParams,
        ...
    ) -> Dict[str, object]: ...
```

### OrchestratorParams

```python
@dataclass
class OrchestratorParams:
    community_id: str
    alpha: float = 0.15           # PPR damping factor
    eps: float = 1e-4             # PPR convergence threshold
    num_walks: int = 5000         # Monte Carlo walks
    walk_len: int = 40            # Walk length
    topn: int = 200               # Top-k PPR nodes
    hop_limit: int = 3            # Beam search depth
    beam_width: int = 64          # Candidates per hop
    max_paths: int = 200          # Maximum output paths
```

### ArangoCommunityAccessor

```python
class ArangoCommunityAccessor:
    # Core methods
    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]: ...
    def iter_in(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]: ...
    def nodes(self, community_id: str) -> Iterable[NodeId]: ...
    
    # Discovery methods (NEW)
    @staticmethod
    def get_community_summaries(db, ...) -> List[dict]: ...
    @staticmethod
    def get_top_entities_in_community(db, community_id, limit) -> List[dict]: ...
    @staticmethod
    def get_recent_entities(db, since, community_id) -> List[dict]: ...
    @staticmethod
    def search_entities(db, query, community_id) -> List[dict]: ...
    
    # Content hydration methods (NEW)
    @staticmethod
    def get_document_content(db, doc_id) -> Optional[dict]: ...
    @staticmethod
    def get_entity_sources(db, entity_id) -> List[dict]: ...
    @staticmethod
    def search_content(db, query, ...) -> List[dict]: ...
```

---

## Integration Guide

### Basic Usage (Recommended)

The simplest way to use Odin is through the `OdinEngine` class, which handles all setup automatically:

```python
from arango import ArangoClient
from odin import OdinEngine

# 1. Connect to ArangoDB
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("mydb", username="root", password="...")

# 2. Initialize Odin (auto-trains NPLL if needed)
engine = OdinEngine(db, community_id="my_community")

# 3. Retrieve
result = engine.retrieve(seeds=["entity/my_entity"], max_paths=100)

# 4. Use results
print(f"Found {len(result['paths'])} paths")
print(f"Triage score: {result['triage']['score']}/100")

# 5. Score individual edges
score = engine.score_edge("Patient_123", "treated_by", "Dr_Smith")
print(f"Edge plausibility: {score}")
```

**What OdinEngine handles automatically:**
- Graph accessor setup with caching
- NPLL model training (if no weights exist in database)
- NPLL weight loading from `OdinModels` collection
- Retrieval orchestration

### NPLL Model Lifecycle

NPLL weights are stored **in the database** (`OdinModels` collection), not in external files:

```
First Run:
1. OdinEngine checks OdinModels collection
2. No weights found → extracts triples from KG
3. Generates domain-aware rules (healthcare/insurance/generic)
4. Trains NPLL (~55 seconds)
5. Saves rule weights to OdinModels (~2 KB)

Subsequent Runs:
1. OdinEngine checks OdinModels collection
2. Weights found with matching data hash → loads weights
3. Rebuilds model from KG (~30 seconds)
4. Ready for inference
```

Force retrain if data changes significantly:
```python
engine.retrain_model()
```

### Advanced Usage (Direct Components)

For more control, you can use components directly:

```python
from arango import ArangoClient
from retrieval.adapters_arango import ArangoCommunityAccessor
from retrieval.cache import CachedGraphAccessor
from retrieval.orchestrator import RetrievalOrchestrator, OrchestratorParams
from retrieval.confidence import NPLLConfidence
from npll.bootstrap import KnowledgeBootstrapper

# 1. Connect
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("mydb", username="root", password="...")

# 2. Setup NPLL via bootstrapper
bootstrapper = KnowledgeBootstrapper(db)
bootstrap_result = bootstrapper.ensure_model_ready()
npll_model = bootstrap_result.model
if bootstrap_result.report and not bootstrap_result.report.converged:
    print("Warning: NPLL training did not converge; confidences may be miscalibrated")
confidence = NPLLConfidence(npll_model, cache_size=10000)

# 3. Setup accessor with caching
base_accessor = ArangoCommunityAccessor(db, community_id="my_community")
cached_accessor = CachedGraphAccessor(base_accessor, cache_size=5000)

# 4. Create orchestrator
orchestrator = RetrievalOrchestrator(
    accessor=cached_accessor,
    edge_confidence=confidence,
)

# 5. Retrieve with full control
result = orchestrator.retrieve(
    seeds=["entity/my_entity"],
    params=OrchestratorParams(
        community_id="my_community",
        hop_limit=3,
        beam_width=64,
        max_paths=200,
    )
)
```

### Integration with AI Agents

```python
from odin import OdinEngine

class InvestigatorAgent:
    def __init__(self, db, llm: LLMClient):
        self.engine = OdinEngine(db)  # One line setup
        self.llm = llm
        self.db = db
    
    def investigate(self, mission: str, seeds: List[str]) -> Brief:
        # 1. Get graph patterns from Odin
        result = self.engine.retrieve(seeds=seeds, max_paths=100)
        
        # 2. Hydrate with actual content
        evidence = []
        for anchor in result["aggregates"]["snippet_anchors"]:
            content = ArangoCommunityAccessor.get_document_content(
                self.db, anchor["document_id"]
            )
            evidence.append(content)
        
        # 3. Reason with LLM
        prompt = f"""
        Mission: {mission}
        
        Graph Patterns Found:
        - Top motif: {result['aggregates']['motifs'][0]['pattern']}
        - Triage score: {result['triage']['score']}/100
        
        Evidence:
        {format_evidence(evidence)}
        
        What insight can you derive? Structure as What/Why/Impact/Action.
        """
        
        analysis = self.llm.generate(prompt)
        
        # 4. Return structured brief
        return Brief.from_analysis(analysis, result)
```

---

## Summary

Odin is a **graph intelligence engine** that guides AI agents through knowledge graphs. It combines:

- **PageRank** for structural importance
- **NPLL** for semantic plausibility
- **Beam Search** for efficient exploration
- **Aggregators** for pattern extraction

**It's not magic.** It's solid graph algorithms + probabilistic logic + careful engineering.

**It works for:** Medium-scale graphs (<5M entities), structured discovery, agent-assisted insight generation.

**It doesn't work for:** Natural language understanding, insight generation (use LLM), massive scale (needs optimization), real-time (<100ms).

For deployment guidance, including caching and memory management, see the [Production Deployment guide](../docs/guides/production.md).

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [Agent Integration Guide](AGENT_INTEGRATION_GUIDE.md) | How agents should use Odin's tools |
| [Production Deployment](../docs/guides/production.md) | Operational guidance for caching, memory, and security |
| [Testing Guide](../tests/README.md) | Test suite documentation |
| [NPLL Paper](2407.03704v1.pdf) | Academic paper on NPLL |

---

**Questions?** Check the test suite in `tests/` or the [Production Deployment guide](../docs/guides/production.md).
