---
icon: material/server
---

# Production Deployment

Running Odin against a real graph at scale comes down to a handful of concerns: sizing the host, keeping the cache warm, managing the model, and making the system observable. This guide works through each in turn.

## Performance evidence

Odin has no published benchmark across graph sizes, so treat latency and memory as things to measure on your own data and hardware. The only recorded measurements are from one small local run (`compliance_docs`, 171 nodes and 173 edges, local ArangoDB): about 98 seconds for fresh initialization, about 0.16 seconds for a cached model reload, and 292 returned paths from 32 seeds. These describe that run only and do not predict behavior on large graphs. Large-graph performance is untested.

## Sizing the host

Memory is the resource to plan around, and it grows with `cache_size` and graph density; measure it on your data. PPR and beam search are CPU-bound, so scale horizontally for concurrency rather than reaching for a bigger box. PyTorch is required for NPLL; the scorer is a small model, but check inference cost on your hardware before assuming a GPU is unnecessary. Watch the cache hit rate before you judge latency, and tune `cache_size` to hold the working set of your typical queries as described in [Caching](../concepts/caching.md).

## Managing the model

Engine construction trains or loads the [NPLL model](npll-lifecycle.md): compatible artifacts load learned tensors without training. Startup still reads and fingerprints the graph; measure its cost on your data and hardware. Warm the engine before it serves traffic so no request pays that cost. Because the weights live in the database rather than in files, there is nothing to ship or mount, but the ArangoDB user does need read and write access to the weights collection. Retrain on structural change, not on every write, following the rules in [Model Lifecycle](npll-lifecycle.md).

## Concurrency

An `OdinEngine` holds a cache and a loaded model, which makes it something to build once and reuse. Construct one engine per configured graph and retrieval scope and share it across requests; constructing a fresh engine per request would pay the warm-up cost every time. Community scope is optional and requires an explicit membership mapping. If you use it, each scope can have its own engine while sharing the globally trained artifact. For multi-tenant serving, key engine pools by database, graph mapping, and retrieval scope; a community ID alone does not isolate tenants.

## Security

Your application owns credentials and connections. For ArangoDB, use a dedicated
user scoped to the database it needs, obtain credentials from the environment
or a secrets manager, and pass the connected handle plus an explicit
`ArangoGraphConfig` to `ArangoBackend(db, graph)`.
Pass that backend to `OdinEngine`. In production, keep the database on a private
network and terminate TLS in front of ArangoDB. See
[Connecting ArangoDB](arangodb.md#production-connections) for details.

## Observability

Every result carries a trace you can log or export straight to your metrics stack:

```python
result["trace"]["timings_ms"]   # per-stage timings incl. 'total'
result["used_budget"]           # exploration budget consumed
result["insight_score"]         # quality signal for dashboards
```

Emitting `triage["score"]` and `timings_ms["total"]` as metrics lets you watch signal quality and latency drift over time, which is usually the earliest sign that a model needs retraining.

## Containerizing

Odin is a library; create a Dockerfile for your service:

```dockerfile
FROM python:3.11-slim
WORKDIR /opt/odin
RUN pip install --no-cache-dir "odin-engine[arango]"
# ... copy your service code, set env, run
```

Provide the ArangoDB connection through environment variables at runtime.

## Pre-flight checklist

- [ ] Dedicated, least-privilege ArangoDB user
- [ ] Credentials injected via environment or secrets manager
- [ ] Engine constructed once per graph and retrieval scope and reused
- [ ] Engine warmed (model loaded) before serving
- [ ] `cache_size` tuned to the working set
- [ ] `timings_ms` and `triage.score` exported as metrics
- [ ] Retrain strategy defined for structural graph changes

## Next

[Tuning Retrieval](tuning.md) covers the parameters behind latency, and [Troubleshooting](../reference/troubleshooting.md) collects the issues that most often come up in a live deployment.
