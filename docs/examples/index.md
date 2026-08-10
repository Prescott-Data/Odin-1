---
icon: material/lightbulb-on
---

# Examples

Worked, end-to-end examples of Odin applied to real problem shapes. Each one uses the same three-signal pipeline — only the seeds and interpretation change.

<div class="grid cards" markdown>

-   :material-hospital-box: **Healthcare Fraud Detection**

    Surface providers billing unusual procedure combinations.

    [Read more →](healthcare-fraud.md)

-   :material-truck-delivery: **Supply Chain Risk**

    Trace cascading dependencies across supplier tiers.

    [Read more →](supply-chain.md)

-   :material-scale-balance: **Regulatory Compliance**

    Validate entity relationships against compliance rules.

    [Read more →](compliance.md)

</div>

---

## The common shape

Every example follows the same skeleton:

```python
from arango import ArangoClient
from odin import OdinEngine

db = ArangoClient(hosts="http://localhost:8529").db(
    "my_graph", username="root", password=""
)
engine = OdinEngine(db, community_id="...", community_mode="mapping")

result = engine.retrieve(seeds=[...], max_paths=..., hop_limit=...)
# → inspect result["triage"], result["paths"], result["aggregates"]["motifs"]
```

What differs between domains is the **seeds** (where you start), the **hop limit** (how far the interesting chains reach), and how the agent **interprets** the returned motifs.
