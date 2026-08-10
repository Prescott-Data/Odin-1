---
icon: material/server
---

# Production Deployment

Guidance for running Odin against real graphs at scale — sizing, caching, model management, and operability.

---

## Tested scale

Odin has been run on graphs from **10K to 5M entities**. Representative deployments:

| Graph | Entities | Edges | Typical retrieval |
|-------|----------|-------|-------------------|
| Healthcare KG | 2.3M | 8.7M | ~450 ms |
| Insurance claims | 850K | 4.1M | ~320 ms |
| Supply chain | 450K | 2.3M | ~280 ms |

Typical end-to-end latency is **300–800 ms** for a 50-path retrieval at a 3-hop limit.

---

## Resource sizing

| Resource | Guidance |
|----------|----------|
| **Memory** | ~500 MB–2 GB depending on `cache_size` and graph density |
| **CPU** | PPR and beam search are CPU-bound; scale horizontally for concurrency |
| **PyTorch** | Required for NPLL; CPU inference is sufficient for scoring |
| **Cache** | Aim for a warm cache (>80% hit rate) before measuring latency |

Tune `cache_size` to hold the working set of your typical queries — see [Caching](../concepts/caching.md).

---

## Model management

- **Warm-up**: the first retrieval after startup trains or loads the [NPLL model](npll-lifecycle.md). Loading a persisted model takes ~30 s; training from scratch takes 2–5 min. Warm the engine before serving traffic.
- **Weights live in the database**: there are no model files to ship or mount. Ensure the ArangoDB user can read/write the weights collection.
- **Retrain on structural change**, not on every write — see [Model Lifecycle](npll-lifecycle.md).

---

## Concurrency

An `OdinEngine` holds a cache and a loaded model. For a service:

- Construct one engine per community and reuse it across requests.
- Do not construct a new engine per request — you would pay warm-up costs every time.
- For multi-tenant serving, keep a small pool of engines keyed by `community_id`.

---

## Security

- **Least privilege**: give Odin a dedicated ArangoDB user scoped to the database it needs.
- **Secrets from the environment**: never hard-code credentials; Odin takes an already-connected `db` handle and never logs credentials.
- **Network**: keep the database on a private network; terminate TLS in front of ArangoDB in production.

See [Connecting ArangoDB](arangodb.md#authenticated-production-connections).

---

## Observability

Every result carries a trace you can log or export:

```python
result["trace"]["timings_ms"]   # per-stage timings incl. 'total'
result["used_budget"]           # exploration budget consumed
result["insight_score"]         # quality signal for dashboards
```

Emit `triage["score"]` and `timings_ms["total"]` as metrics to watch signal quality and latency over time.

---

## Containerizing

A minimal image is provided in the repository (`Dockerfile`). Install the package and your app on top of a slim Python base:

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir odin-engine
# ... copy your service code, set env, run
```

Provide the ArangoDB connection via environment variables at runtime.

---

## Pre-flight checklist

- [ ] Dedicated, least-privilege ArangoDB user
- [ ] Credentials injected via environment / secrets manager
- [ ] Engine constructed once per community and reused
- [ ] Engine warmed (model loaded) before serving
- [ ] `cache_size` tuned to the working set
- [ ] `timings_ms` and `triage.score` exported as metrics
- [ ] Retrain strategy defined for structural graph changes

---

## Next

- [Tuning Retrieval](tuning.md)
- [Troubleshooting](../reference/troubleshooting.md)
