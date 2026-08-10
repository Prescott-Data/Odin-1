# Agent Integration Guide for the Odin KG Engine

## 0. Quick Start with OdinEngine

The simplest way to integrate Odin with your agents:

```python
from arango import ArangoClient
from odin import OdinEngine

# Connect to database
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("KG-test", username="lisa", password="...")

# Initialize Odin (auto-trains NPLL if needed, stores weights in DB)
engine = OdinEngine(db, community_id="healthcare")

# Use in your agent
result = engine.retrieve(seeds=["Patient_123"], max_paths=100)
score = engine.score_edge("Patient_A", "treated_by", "Dr_Smith")
```

**That's it.** No manual NPLL training, no .pt files, no complex wiring.

---

## 1. A New Paradigm: From Retriever to Reasoning Partner

The Odin KG Engine is more than a simple fact-retrieval tool; it is a **reasoning partner** for your AI agents. To unlock its full potential, agents should be designed to move beyond simple question-answering and adopt a multi-step reasoning strategy that involves exploration, hypothesis generation, and validation.

This guide outlines the tools the engine provides and the strategic flows your agents should follow to perform both **proactive insight discovery** and **reactive deep dives**.

## 2. The Agent's Toolkit

### Option A: OdinEngine (Recommended)

The simplest approach using the high-level `OdinEngine` class:

| Method | Description |
|--------|-------------|
| `engine.retrieve(seeds, max_paths)` | Find paths from seed entities |
| `engine.score_edge(src, rel, dst)` | Score edge plausibility (0-1) |
| `engine.find_anchors(seeds, topn)` | Get top nodes by PageRank |
| `engine.retrain_model()` | Force NPLL retrain |

```python
from odin import OdinEngine

engine = OdinEngine(db)
result = engine.retrieve(seeds=["Patient_123"])
score = engine.score_edge("Patient_A", "treated_by", "Dr_Smith")
```

### Option B: RetrievalOrchestrator (Advanced)

For more control, use the orchestrator directly with three primary methods:

### Tool 1: `find_insights`

-   **Description:** "Use this tool to find and score the most relevant insights and connection paths starting from one or more known entities (seeds) in the knowledge graph. Best for open-ended exploration when you have specific starting points."
-   **Maps to:** `RetrievalOrchestrator.retrieve()`
-   **Key Arguments:** `seeds: List[str]`, `community_id: str`, `question: str` (for context), `anchor_prior: Optional[Dict]`

### Tool 2: `find_insights_from_text`

-   **Description:** "Use this tool when you have a block of unstructured text (like a document, email, or long user query). It will automatically link entities from the text to the knowledge graph, then find and score the most relevant insights connected to them. This is the most powerful tool for contextual analysis."
-   **Maps to:** `RetrievalOrchestrator.link_and_retrieve()`
-   **Key Arguments:** `text: str`, `community_id: str`, `question: str` (for context)

### Tool 3: `validate_connection`

-   **Description:** "Use this tool to efficiently check for a specific connection between a source entity and a list of target entities. It returns a confidence score for each potential connection. Best for hypothesis testing or when you need to confirm a specific relationship."
-   **Maps to:** `RetrievalOrchestrator.score_candidates()`
-   **Key Arguments:** `source: str`, `targets: List[str]`, `community_id: str`

## 3. Core Reasoning Flows

### Flow A: Proactive Insight Generation (for Briefs & Channels)

This is the autonomous process where the agent acts as an analyst, discovering insights without a direct user query.

1.  **Thematic Framing:** The agent identifies the customer's industry (e.g., "Insurance") and loads a "playbook" of high-value concepts (e.g., "Fraud," "Operational Efficiency"). Each concept has associated **insight templates** or reasoning patterns.
    -   *Example Pattern (Mileage Fraud):* `(Claim -> has_policyholder -> Person) AND (Claim -> has_mileage -> Mileage_Value)`

2.  **Community-Scoped Hypothesis Generation:** The agent uses pre-computed `APPRAnchors` or `GlobalPR` scores to identify central nodes within relevant communities (e.g., "high-claim-volume" community). These central nodes become the **seeds** for exploration.

3.  **Pattern Validation & Insight Discovery:**
    -   For each seed (e.g., a specific person), the agent calls the **`find_insights`** tool.
    -   It uses the `beam_override` parameter to pass in `BeamParams` that constrain the search to the relations in its current insight template (e.g., `allowed_relations={"has_policyholder", "has_mileage"}`). This makes the search highly efficient.
    -   The engine returns a set of scored, multi-hop paths.

4.  **Calculation & Synthesis:** The agent processes the returned paths, performing any necessary calculations (e.g., counting claims with identical mileage in a 6-month window). The `insight_score` from the engine helps rank the credibility of the findings.

5.  **Surfacing:**
    -   **For Briefs:** A single, high-scoring, validated pattern is formatted into the "What happened, why it matters, what to do" template.
    -   **For Channels:** The agent runs this process for many seeds, aggregates the results (e.g., total cost of mileage fraud), and generates a data story for the relevant channel (e.g., "Fraud Channel").

### Flow B: Reactive Deep Dives (for User Prompts)

This is the online process where the agent responds to a direct user query, often about an already-surfaced insight.

1.  **User Asks a Question:** The user provides a prompt, often with rich context (e.g., "Show me the assessor reports for Thomas's claims and see if the same garage was used.").

2.  **Agent Selects the Right Tool:** For a contextual, unstructured query, the agent's first choice should be **`find_insights_from_text`**.

3.  **Engine Processes the Query:** The engine's `CoherenceLinker` links entities from the user's text ("assessor reports," "garage") to the KG. It then runs the full retrieval pipeline, using the linked entities as seeds.

4.  **Agent Synthesizes the Answer:** The engine returns the connecting paths. The agent uses this evidence to construct a precise, intelligent answer, for example, "Yes, all three claims were serviced by 'Garage X.' The associated reports are A, B, and C."

## 4. Putting It All Together: Mileage Fraud Example

1.  **Proactive Discovery:**
    -   The agent's "Fraud" playbook contains the mileage fraud pattern.
    -   It identifies "Thomas" as a high-claim individual from community anchors.
    -   It calls `find_insights(seeds=["Thomas"], beam_override=BeamParams(allowed_relations={...}))`.
    -   It receives paths detailing three claims with identical mileage. It calculates the timeframe and confirms the pattern.
    -   It generates the brief: "Thomas had 3 consecutive claims within 6 months with the same mileage, indicating potential fraud."
    -   It adds this finding to its aggregation for the "Fraud Channel."

2.  **Reactive Deep Dive:**
    -   The user sees the brief and asks, "Was the same assessor involved in all three of Thomas's claims?"
    -   The agent calls `find_insights_from_text(text="Was the same assessor involved in all three of Thomas's claims?")`.
    -   The engine links "assessor" and runs retrieval, finding paths that connect all three claims to the same assessor, "Assessor Y."
    -   The agent responds: "Yes, 'Assessor Y' was involved in all three of Thomas's claims."

By following these patterns, your agents can fully leverage the Odin KG Engine's capabilities to move from simple fact retrieval to sophisticated, autonomous insight generation.

## 5. Advanced Topics

This section provides deeper guidance for experienced developers on tuning the engine, handling complex scenarios, and designing robust agentic logic.

### 5.1 A Deeper Dive on Tuning

While the default parameters in `OrchestratorParams` and `BeamParams` provide a good balance, you can tune them to optimize for speed, exhaustiveness, or specific use cases.

-   **For Faster, High-Confidence Results (e.g., real-time user interaction):**
    -   `hop_limit`: Decrease to `2`. This is the most significant factor in reducing latency.
    -   `beam_width`: Decrease to `32` or `48`. A smaller beam explores fewer paths at each hop.
    -   `max_paths`: Decrease to `50` or `100`.
    -   `num_walks` (in `OrchestratorParams`): Decrease to `2000` or `3000` for the Monte Carlo part of PPR.

-   **For More Exhaustive, "Deep Dive" Analysis (e.g., offline insight generation for Channels):**
    -   `hop_limit`: Increase to `3` or `4` (be cautious, as this can lead to a combinatorial explosion).
    -   `beam_width`: Increase to `128` or `256`.
    -   `max_paths`: Increase to `200` or `300`.
    -   `num_walks`: Increase to `5000` or `10000` for a more accurate PPR approximation.

-   **For Constraining the Search Space:**
    -   `allowed_relations`: Use this to force the beam search to only follow edges with specific relation types. This is extremely powerful for pattern-based analysis (as seen in the "Mileage Fraud" example).
    -   `max_out_degree`: Set this to a reasonable number (e.g., `100`) to prevent the search from getting stuck on super-nodes (entities with thousands of connections).

### 5.2 Advanced Agent Strategy & Error Handling

An intelligent agent should be able to react gracefully to different kinds of results from the engine.

-   **If `find_insights` returns no paths:**
    -   **Strategy 1 (Broaden the Seed):** The initial seed might be too specific. The agent could query the KG for a higher-level entity related to the seed (e.g., if "Project Titan" fails, try the "Department" it belongs to) and re-run the query with that new seed.
    -   **Strategy 2 (Relax Beam Parameters):** The agent could retry the query with a slightly larger `hop_limit` or `beam_width`, respecting a higher-level budget.
    -   **Strategy 3 (Fall back to NPLL):** The agent could use `NPLLModel.predict_unknown_facts()` to see if there are any high-confidence, single-hop links it might be missing, and then use those as new seeds.

-   **If `insight_score` is low:**
    -   This is a valuable signal. It means that while paths may exist, the engine does not have high confidence in them (due to low PPR relevance, low confidence edges, etc.).
    -   The agent should interpret this as "connections are tenuous or unproven" and can report this back to the user, rather than presenting a weak insight as a strong one.

-   **If `link_and_retrieve` fails to link key entities:**
    -   The agent can report that "The entity 'XYZ' could not be found in the knowledge graph."
    -   It could then suggest a clarifying question to the user or attempt a search using a partial or alternative name.

### 5.3 Implementing "Playbooks" for Proactive Insight

A "playbook" is a structured way to define the logic for your proactive insight generation agents. A good playbook, which could be stored as a YAML or JSON file, should contain:

1.  **`insight_id`:** A unique identifier (e.g., `mileage_fraud_v1`).
2.  **`description`:** A human-readable explanation of what the insight is about.
3.  **`target_community`:** The community to run this analysis in (e.g., "high_claim_volume").
4.  **`seed_strategy`:** How to get the initial entities for exploration.
    -   `type`: e.g., `APPR_ANCHORS`, `GLOBAL_PR`, or `ENTITY_TYPE_QUERY`.
    -   `params`: e.g., `top_k: 100`.
5.  **`retrieval_params`:** The specific `OrchestratorParams` and `BeamParams` to use.
    -   Crucially, this should include the `allowed_relations` that define the core pattern of the insight.
6.  **`validation_logic`:** A reference to a function or a set of rules the agent should apply to the paths returned by the engine to confirm the insight.
    -   `function`: e.g., `calculate_mileage_consistency`.
    -   `params`: e.g., `time_window_months: 6`, `min_claims: 3`.
7.  **`output_templates`:** Templates for formatting the final output for both Briefs and Channels.
    -   `brief_template`: "*{person}* had *{claim_count}* claims within *{time_window}* months with the same mileage, indicating potential fraud."
    -   `channel_aggregation`: `SUM(claim_value)`.

By structuring your agent's knowledge in this way, you can easily add new insight discovery patterns, tune their performance, and manage them at scale, all while using the Odin KG Engine as the powerful, flexible reasoning backend.
