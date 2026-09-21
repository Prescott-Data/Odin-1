---
icon: material/lifebuoy
---

# Troubleshooting

Common issues and how to resolve them.

---

## `intelligence_mode` is `Constant`, not `NPLL`

`engine.get_status()["intelligence_mode"] == "Constant"` means you explicitly
disabled NPLL with `auto_train=False`. Set `auto_train=True` only when the backend
supplies training capabilities and the graph is ready to train. Training
failures raise instead of selecting constant confidence.

You still get PPR-driven structural exploration in constant mode; you lose semantic pruning until a model trains. See [Model Lifecycle](../guides/npll-lifecycle.md).

```python
import logging
logging.getLogger("odin").setLevel(logging.INFO)
```

---

## First retrieval is very slow

Engine construction trains or loads the NPLL model when `auto_train=True`;
`retrieve()` does not start training. Startup cost depends on graph size and
training settings. Construct the engine before serving traffic and reuse it.
See [Production Deployment](../guides/production.md).

## Backend migration and training errors

| Error | Action |
| --- | --- |
| `BackendConfigurationError` | Pass a backend, such as `ArangoBackend(db)`, and implement every accessor method; inherited protocol placeholders do not count. |
| `BackendCapabilityError` | Supply the requested capability, or explicitly disable training for a retrieval-only backend. |
| `npll.TrainingError` | Inspect the exception cause and verify the graph contains valid training triples. Failed retraining leaves the active serving state intact. |
| `BackendIOError` | Check database connectivity, access permissions, and the original exception cause. |
| `CorruptModelError` | Inspect the stored artifact against the current schema; do not treat corruption as an absent model. |
| `ModelConflictError` | Another writer changed the artifact. Coordinate training and load the current artifact rather than blindly overwriting it. |

Invalid triple identities raise `BackendError`; Arango entity IDs and relation
labels must be non-empty strings, and entity types must be strings when present.
See [Backend migration](../guides/backend-migration.md) for import changes and
the first-startup retraining requirement.

---

## `retrieve()` returns no paths

An empty `paths` list with `summary["low_support"] == True` usually means:

- The **seed IDs do not exist** or are misformatted (use the full `collection/key` form).
- The seeds have **no reachable neighbors** within `hop_limit`.
- The scope (`community_id` / `community_mode`) **excludes** the relevant subgraph.

Verify a seed exists and inspect its neighborhood:

```python
engine.get_neighbors("entity/claim_123")["degree"]   # should be > 0
```

---

## Triage score is surprisingly low

Inspect the components, since a guard may have fired:

```python
result["triage"]["components"]
```

- `penalty == 15.0` → **label coverage below 0.8**; improve node/edge labeling.
- `low_support == True` → **too little evidence**; broaden seeds or raise `max_paths`.
- Low `recency`/`provenance` → edges lack `created_at`/`provenance` metadata.

See [Triage & Insight Scoring](../concepts/scoring.md).

---

## Connection errors to ArangoDB

- Confirm the database is reachable: `curl http://localhost:8529/_api/version`.
- Check host, database name, username, and password.
- In Docker, ensure the port is published (`-p 8529:8529`).

See [Connecting ArangoDB](../guides/arangodb.md).

---

## `ImportError` / wrong package

For the unreleased backend API, install from the matching source checkout and
import the **short** name:

```bash
pip install -e ".[arango]"
```

```python
from odin import OdinEngine       # correct
# not: import odin_engine
```

---

## Out-of-memory on large graphs

Memory scales with `cache_size` and graph density (~500 MB-2 GB typical). Lower `cache_size` to reduce the resident working set at the cost of more database round-trips. See [Caching](../concepts/caching.md).

---

## High latency

Inspect where time goes before changing parameters:

```python
result["trace"]["timings_ms"]
```

Then, per [Tuning Retrieval](../guides/tuning.md): narrow `beam_width`, reduce `hop_limit`, or tighten seeds.

---

## Still stuck?

Open an issue with a minimal reproduction: [github.com/Prescott-Data/Odin-1/issues](https://github.com/Prescott-Data/Odin-1/issues).
