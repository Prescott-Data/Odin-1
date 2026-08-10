---
icon: material/hospital-box
---

# Healthcare Fraud Detection

Say you want to find providers billing unusual procedure combinations, the kind of pattern that hints at fraud, waste, or abuse. This is Odin's home turf, and it shows how the three signals combine on a real problem.

The graph you are working with is a claims knowledge graph: entities like `Provider`, `Claim`, `Patient`, `Diagnosis`, `Procedure` (CPT codes), and `Audit`, joined by relations such as `billed_by`, `has_diagnosis`, `has_procedure`, and `flagged_in`.

## Explore from a suspicious provider

Start from a provider you want to scrutinize and let Odin find the patterns around it:

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

The payoff is that recurring billing pattern, *"CPT_99285 paired with an office visit appears in 34 claims"*, surfaced as a motif, without anyone having written a rule for that specific combination beforehand. That is the three signals working together: [PPR](../concepts/ppr.md) zeroes in on the high-volume providers and central claims worth examining, [NPLL](../concepts/npll.md) throws out clinically nonsensical edges (a procedure is not `diagnosed_by` a patient) so the paths stay coherent, and [motif aggregation](../concepts/aggregation.md) compresses thousands of claims into a handful of repeated patterns you can actually read.

## Validate a specific concern

When an analyst already suspects a particular relationship, there is no need to retrieve a whole neighborhood; just score the single edge:

```python
score = engine.score_edge(
    "provider/high_volume_clinic", "billed", "procedure/CPT_99285"
)
if score > 0.7:
    analyst.flag("Unusually strong billing tie to high-acuity code")
```

## Hand off to an agent

Once a retrieval clears the triage bar, the evidence goes to an agent to write up:

```python
if result["triage"]["score"] >= 70:
    fraud_agent.reason(
        prompt="Explain why this provider's billing pattern is anomalous.",
        evidence=result["paths"][:10],
        motifs=result["aggregates"]["motifs"][:5],
    )
```

The agent writes the narrative; Odin guarantees the evidence underneath it is real and ranked. See [AI Agent Integration](../guides/agent-integration.md) for the full loop.

A few domain-specific defaults are worth keeping in mind: a `hop_limit` of 3 fits fraud patterns well because they usually run provider → claim → code/audit in just a few hops; widen `beam_width` if you suspect patterns are slipping through across many claims; and scope with `community_mode="mapping"` so NPLL learns *this* payer's billing norms rather than a global average.

---

The same shape applies with deeper hops in [Supply Chain Risk](supply-chain.md) and with edge-level rule checks in [Regulatory Compliance](compliance.md).
