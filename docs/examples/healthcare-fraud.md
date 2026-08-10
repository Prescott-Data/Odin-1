---
icon: material/hospital-box
---

# Healthcare Fraud Detection

**Scenario:** find providers billing unusual procedure combinations that may indicate fraud, waste, or abuse.

---

## The graph

A claims knowledge graph typically contains entities like `Provider`, `Claim`, `Patient`, `Diagnosis`, `Procedure` (CPT codes), and `Audit`, connected by relations such as `billed_by`, `has_diagnosis`, `has_procedure`, and `flagged_in`.

---

## Explore from a suspicious provider

```python
from arango import ArangoClient
from odin import OdinEngine

db = ArangoClient(hosts="http://localhost:8529").db(
    "claims", username="root", password=""
)
engine = OdinEngine(db, community_id="medicare_claims", community_mode="mapping")

result = engine.retrieve(
    seeds=["provider/high_volume_clinic"],
    max_paths=100,
    hop_limit=3,
)

print(f"Triage: {result['triage']['score']}/100")
for m in result["aggregates"]["motifs"][:5]:
    print(m)
# e.g. {"pattern": "CPT_99285 + CPT_office_visit", "count": 34}
```

Odin surfaces the recurring billing pattern — *"CPT_99285 paired with an office visit appears in 34 claims"* — as a motif, without anyone writing a rule for that specific combination in advance.

---

## Why the three signals matter here

| Signal | In this domain |
|--------|----------------|
| [PPR](../concepts/ppr.md) | Finds the high-volume providers and central claims worth examining |
| [NPLL](../concepts/npll.md) | Rejects nonsensical edges (a procedure is not `diagnosed_by` a patient), keeping paths clinically coherent |
| [Motifs](../concepts/aggregation.md) | Turns thousands of claims into a handful of repeated billing patterns |

---

## Validate a specific concern

If an analyst suspects a particular relationship, score it directly:

```python
score = engine.score_edge(
    "provider/high_volume_clinic", "billed", "procedure/CPT_99285"
)
if score > 0.7:
    analyst.flag("Unusually strong billing tie to high-acuity code")
```

---

## Hand off to an agent

```python
if result["triage"]["score"] >= 70:
    fraud_agent.reason(
        prompt="Explain why this provider's billing pattern is anomalous.",
        evidence=result["paths"][:10],
        motifs=result["aggregates"]["motifs"][:5],
    )
```

The agent writes the narrative; Odin guarantees the evidence is real and ranked. See [AI Agent Integration](../guides/agent-integration.md).

---

## Tuning notes

- Keep `hop_limit=3` — fraud patterns are usually provider → claim → code/audit, a few hops.
- Widen `beam_width` if you suspect patterns are being missed across many claims.
- Scope with `community_mode="mapping"` so the NPLL model learns *this* payer's billing norms.

---

## Next

- [Supply Chain Risk](supply-chain.md)
- [Regulatory Compliance](compliance.md)
