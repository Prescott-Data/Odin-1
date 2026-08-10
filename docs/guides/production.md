---
icon: material/server
---

# Production Deployment

Running Odin against a real graph at scale comes down to a handful of concerns: sizing the host, keeping the cache warm, managing the model, and making the system observable. This guide works through each in turn.

## What it has been run on

Odin has been deployed on graphs ranging from 10K to 5M entities, with typical end-to-end latency of 300 to 800 ms for a 50-path retrieval at a 3-hop limit. Some representative deployments:

| Graph | Entities | Edges | Typical retrieval |
|-------|----------|-------|-------------------|
| Healthcare KG | 2.3M | 8.7M | ~450 ms |
| Insurance claims | 850K | 4.1M | ~320 ms |
| Supply chain | 450K | 2.3M | ~280 ms |

## Sizing the host

Memory is the resource to plan around: expect roughly 500 MB to 2 GB depending on `cache_size` and graph density. PPR and beam search are CPU-bound, so scale horizontally for concurrency rather than reaching for a bigger box. PyTorch is required for NPLL, but CPU inference is enough for scoring, so no GPU is needed. Above all, aim for a warm cache (a hit rate above 80%) before you judge latency, and tune `cache_size` to hold the working set of your typical queries as described in [Caching](../concepts/caching.md).

## Managing the model

The first retrieval after startup trains or loads the [NPLL model](npll-lifecycle.md): loading persisted weights takes about thirty seconds, training from scratch two to five minutes. Warm the engine before it serves traffic so no request pays that cost. Because the weights live in the database rather than in files, there is nothing to ship or mount, but the ArangoDB user does need read and write access to the weights collection. Retrain on structural change, not on every write, following the rules in [Model Lifecycle](npll-lifecycle.md).

## Concurrency

An `OdinEngine` holds a cache and a loaded model, which makes it something to build once and reuse. Construct one engine per community and share it across requests; constructing a fresh engine per request would pay the warm-up cost every time. For multi-tenant serving, keep a small pool of engines keyed by `community_id`.

## Security

Odin never manages credentials itself, which keeps the security story simple. Give it a dedicated, least-privilege ArangoDB user scoped to the database it needs, and inject secrets from the environment or a secrets manager rather than hard-coding them; Odin takes an already-connected `db` handle and never logs credentials. In production, keep the database on a private network and terminate TLS in front of ArangoDB. See [Connecting ArangoDB](arangodb.md#production-connections) for the connection details.

## Observability

Every result carries a trace you can log or export straight to your metrics stack:

```python
result["trace"]["timings_ms"]   # per-stage timings incl. 'total'
result["used_budget"]           # exploration budget consumed
result["insight_score"]         # quality signal for dashboards
```

Emitting `triage["score"]` and `timings_ms["total"]` as metrics lets you watch signal quality and latency drift over time, which is usually the earliest sign that a model needs retraining.

## Containerizing

The repository ships a minimal `Dockerfile`. For a service, install the package on a slim Python base and add your own code on top:

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir odin-engine
# ... copy your service code, set env, run
```

Provide the ArangoDB connection through environment variables at runtime.

## Pre-flight checklist

- [ ] Dedicated, least-privilege ArangoDB user
- [ ] Credentials injected via environment or secrets manager
- [ ] Engine constructed once per community and reused
- [ ] Engine warmed (model loaded) before serving
- [ ] `cache_size` tuned to the working set
- [ ] `timings_ms` and `triage.score` exported as metrics
- [ ] Retrain strategy defined for structural graph changes

## Next

[Tuning Retrieval](tuning.md) covers the parameters behind latency, and [Troubleshooting](../reference/troubleshooting.md) collects the issues that most often come up in a live deployment.
