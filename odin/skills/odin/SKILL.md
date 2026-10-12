---
name: odin
description: Use when writing or reviewing Python that calls the Odin graph-intelligence engine (the `odin-engine` package, `from odin import OdinEngine`) on ArangoDB or an existing backend. Covers the explicit-backend API (0.4.0 and later), the exact retrieve() result shape, how to handle evidence and failures, and common agent mistakes. Not for writing a new database backend (use odin-custom-backend) and not for generic graph or PageRank code.
---

# Odin

Odin returns ranked, scored paths through a knowledge graph (Personalized PageRank, beam search, learned edge plausibility). It is the compass, not the explorer: it returns structured evidence and the calling code interprets it.

Package `odin-engine`, import name `odin`.

## Check the installed library first

These examples target the explicit-backend API (odin-engine 0.4.0 and later). `__version__` is not a reliable signal in a source checkout, so probe the capability:

```bash
python -c "from odin.backends import GraphBackend"
```

If that import fails, the installed Odin is older. Do not use these examples; tell the user to install a matching version, or read `references/migration.md` if it exists.

## Where to look next

Load only what the task needs. If a referenced file is missing in this install, use the docs at https://odin.developers.prescottdata.io (index for agents: https://odin.developers.prescottdata.io/llms.txt).

| Task | Read |
| --- | --- |
| Map an ArangoDB graph, scopes, timestamps, provenance | `references/arango-config.md` |
| Interpret any `retrieve()` field | `references/result-schema.md` |
| Handle an exception | `references/errors.md` |
| Update code written for 0.3.0 (`OdinEngine(db)`) | `references/migration.md` |
| Implement a backend for another database | the `odin-custom-backend` skill |

## Setup (ArangoDB)

```python
import os

from arango import ArangoClient
from odin import OdinEngine
from odin.backends.arango import ArangoBackend, ArangoGraphConfig

client = ArangoClient(hosts=os.environ["ARANGO_HOSTS"])
db = client.db(os.environ["ARANGO_DB"],
               username=os.environ["ARANGO_USER"], password=os.environ["ARANGO_PASSWORD"])

graph = ArangoGraphConfig(
    node_collection="CaseRecords",
    edge_collection="EvidenceLinks",
    relation_field="predicate",
)
engine = OdinEngine(ArangoBackend(db, graph))
```

Credentials stay in the caller's connection code. Never hard-code a password. The three fields above are the only required mapping; Odin never assumes collection or field names. The first construction trains the model and can take minutes; build the engine once and reuse it.

## API

```text
OdinEngine(backend, community_id="global", cache_size=5000, auto_train=True, community_mode="none")
engine.retrieve(seeds, max_paths=50, hop_limit=3, beam_width=64)
engine.score_edge(src, rel, dst)
engine.find_anchors(seeds, topn=20)
engine.get_neighbors(node_id)
engine.retrain_model()
engine.get_status()
```

`has_npll` is a property. `seeds` are full document IDs such as `"CaseRecords/claim_123"` and must exist in the graph. `find_anchors` returns `(node_id, ppr_score)` pairs. `inspect_schema(backend)` (from `odin`) returns a map of the live schema.

## Reading a result

```python
for p in result["paths"]:
    edges = p["edges"]
    nodes = [edges[0]["u"], *(e["v"] for e in edges)] if edges else []
```

Derive nodes from `edges`. A path dict may also carry internal keys such as `nodes` and `decomp`; they are not part of the documented schema, so do not depend on them. Each edge has `u`, `v`, `relation`, `confidence`, `created_at` and `provenance`. Full shape: `references/result-schema.md`.

## Evidence rules

- **Keep what the edges say.** When you format paths for a user or an LLM, keep each relation label, the edge direction (`u` to `v`) and the provenance. Do not flatten a path to a list of node names.
- **Scores rank, they do not prove.** `confidence`, `score`, `insight_score` and `triage["score"]` are ranking signals from learned plausibility. Never present them as the probability that something is true. Point to provenance for support.
- **Thresholds belong to the application.** Read the cutoff for "worth investigating" from configuration or the user. Do not invent or hard-code one.
- **Empty is not failure.** No reachable evidence returns a well-formed result with empty `paths` and `summary["low_support"]` true. Failures raise exceptions (see `references/errors.md`). Do not catch an exception and treat it as "no results", and do not treat empty results as an error.
- **Make truncation visible.** If you show, log or pass on fewer paths, edges or motifs than Odin returned, say how many were returned and how many are shown. Do not slice silently.

## Common mistakes

- `path["nodes"]` is not a documented field. Use `path["edges"]`.
- Motif fields are `edge_count` and `path_count`, not `count`. `relation_share[rel]` is `{"count", "share"}`.
- `score_edge` takes `(src, rel, dst)` in that order.
- `OdinEngine` takes a backend, never a raw database handle, URL or connection string.
- Do not pass `auto_train=False` to hide a training error. It means structural retrieval with constant confidence, chosen on purpose. Training failures raise.
- Check `engine.has_npll` before relying on fine-grained plausibility.
- IDs are full document IDs including the collection, exactly as they exist in the graph. A bare key finds nothing in ArangoDB, but never invent a collection prefix: use the IDs the user or the data gives you.
