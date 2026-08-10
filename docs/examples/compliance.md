---
icon: material/scale-balance
---

# Regulatory Compliance

Compliance work often comes down to one question: is this entity connected — directly or indirectly — to something it should not be, like a sanctioned entity? Odin is well-suited to it because the prohibited relationships are frequently *indirect*, hidden a few hops from anything an analyst would look at directly.

The graph holds entities like `Fund`, `Manager`, `Entity`, `Person`, and `Jurisdiction`, connected by relations such as `managed_by`, `owns`, `affiliated_with`, and `controlled_by`.

## The direct case: check one edge

When you have a specific prohibited relationship in mind, score it directly — no traversal needed:

```python
from arango import ArangoClient
from odin import OdinEngine

db = ArangoClient(hosts="http://localhost:8529").db(
    "compliance", username="root", password=""
)
engine = OdinEngine(db, community_id="compliance", community_mode="mapping")

score = engine.score_edge(
    "entity/investment_fund",
    "managed_by",
    "entity/sanctioned_entity",
)
if score > 0.5:
    compliance_agent.flag_for_review(
        "Fund plausibly managed by a sanctioned entity"
    )
```

`score_edge()` gives you the NPLL plausibility of a single relationship — a fast gate for rule checks. See [Scoring Edges](../guides/edge-scoring.md).

---

## Find indirect exposure with retrieval

Direct edges are the easy case. To catch *indirect* ties, retrieve paths from the fund and look for any that reach a sanctioned entity:

```python
result = engine.retrieve(
    seeds=["entity/investment_fund"],
    hop_limit=4,          # indirect control can be several hops away
    max_paths=100,
)

for p in result["paths"]:
    if any("sanctioned" in str(n) for n in p["nodes"]):
        compliance_agent.flag_for_review(p)
```

This finds chains like `Fund → managed_by → Manager → affiliated_with → Sanctioned Entity` that a single-edge check would miss.

---

## Why NPLL helps compliance

| Concern | How Odin helps |
|---------|----------------|
| False positives from coincidental edges | NPLL down-weights implausible connections |
| Hidden, indirect relationships | Multi-hop [beam search](../concepts/beam-search.md) surfaces them |
| Explainability for auditors | Every flagged path is a real, inspectable chain in the graph |

Because Odin only returns paths that exist in your data, every flag is auditable — there is no hallucinated relationship to defend.

---

## Escalation gate

```python
score = result["triage"]["score"]
if score >= 75:
    compliance_agent.escalate(result)     # strong, well-sourced exposure
elif score >= 40:
    human_review.enqueue(result)          # uncertain — needs an analyst
```

See [AI Agent Integration](../guides/agent-integration.md) for the human-in-the-loop pattern.

---

## Next

- [Scoring Edges](../guides/edge-scoring.md)
- [Examples overview](index.md)
