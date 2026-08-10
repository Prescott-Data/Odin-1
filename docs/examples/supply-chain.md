---
icon: material/truck-delivery
---

# Supply Chain Risk

**Scenario:** identify cascading supplier dependencies — where a problem at a deep-tier supplier ripples up to many downstream products.

---

## The graph

A supply-chain graph contains entities like `Supplier`, `Component`, `Product`, `Facility`, and `Region`, connected by relations such as `supplies`, `depends_on`, `manufactured_at`, and `ships_to`. Risk often hides several tiers deep.

---

## Explore deep dependencies

Because supply chains are deep, this is a case for a **higher `hop_limit`**:

```python
from arango import ArangoClient
from odin import OdinEngine

db = ArangoClient(hosts="http://localhost:8529").db(
    "supply", username="root", password=""
)
engine = OdinEngine(db, community_id="supply", community_mode="mapping")

result = engine.retrieve(
    seeds=["supplier/critical_vendor"],
    hop_limit=5,          # deep exploration across tiers
    max_paths=100,
)

for p in result["paths"][:10]:
    print(f"[{p['score']:.2f}]", " -> ".join(str(n) for n in p["nodes"]))
# Discovers: a Tier-3 supplier feeds 47 downstream products
```

The deep walk reveals chains like `Tier-3 supplier → component → sub-assembly → product`, quantifying how far a single vendor's disruption propagates.

---

## Why deeper hops here

| Domain trait | Odin setting |
|--------------|--------------|
| Risk is many tiers deep | `hop_limit=5` (or more) |
| Chains fan out widely | Narrow the `beam_width` to keep depth affordable |
| One vendor, many products | Seed on the vendor; read node frequencies in the aggregates |

See [Tuning Retrieval](../guides/tuning.md) for the depth-vs-breadth trade-off.

---

## Rank the exposure

Use [anchors](../guides/anchors.md) to find the most structurally central suppliers before drilling in:

```python
anchors = engine.find_anchors(seeds=["region/southeast_asia"], topn=20)
for node_id, ppr in anchors[:10]:
    print(f"{ppr:.4f}  {node_id}")   # the load-bearing suppliers in the region
```

High-PPR suppliers are the ones whose failure would affect the most paths — the priorities for a resilience review.

---

## Hand off to an agent

```python
if result["triage"]["score"] >= 65:
    risk_agent.reason(
        prompt="Summarize the single-point-of-failure risk in this supply chain.",
        evidence=result["paths"][:10],
    )
```

---

## Next

- [Healthcare Fraud Detection](healthcare-fraud.md)
- [Regulatory Compliance](compliance.md)
