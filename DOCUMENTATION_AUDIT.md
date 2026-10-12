# Documentation audit

Audit date: October 12, 2026. Implementation reviewed: `f67f231d848fc3f94cda9b2d51136bdca6da2d41`, branch `feat/backend-abstraction`.

## Verdict

The repository documentation is not yet uniformly accurate enough to cite as a
description of this implementation. The recently corrected backend capability,
reload, and confidence contracts hold. Broader review found the issues below in
other pages, examples, the packaged skill, and research/architecture documents.
This is an audit, not an implementation or documentation repair batch.

## Scope and method

The corpus scan covered 65 tracked text sources and related documentation assets:
root documentation, the MkDocs pages and configuration, editor skill, Markdown,
HTML and TeX whitepapers, diagrams, and documentation scripts. API and behavioral
claims were compared with the implementation; suspicious examples were checked
with imports, signature binding, syntax parsing, and read-only in-memory runs.
The existing October 8 local lifecycle and confidence artifacts were inspected.

This is not a certification of every sentence. Published research experiments,
third-party references, company biographies, customer deployments, and enterprise
product claims cannot be validated from this repository. The bundled
`whitepaper/2407.03704v1.pdf` is a reference artifact, not a verified specification
for this branch; its paper results were not reproduced. No external websites or
published package contents were re-audited. Illustrative API declarations and
abbreviated schema literals were distinguished from runnable Python; parser
warnings alone are not findings.

The unrelated untracked `docs/jev-semantic-retrieval-plan.md` was excluded from
the tracked-source audit and left untouched. MkDocs nevertheless includes it in
the generated site, with an informational notice that it is absent from nav.
Do not commit that generated page as verified current functionality.

Complete checks, corpus snapshots, logs, and synthetic retrieval results are in
`../verification-artifacts/odin-documentation-audit-20261012/`.
The stable audit call ID is `odin-documentation-audit-20261012`.

## High priority findings

### D01. Agent examples discard reasoning evidence before model delivery

Locations: `docs/guides/agent-integration.md:26`, `:27`, `:75`;
`docs/examples/healthcare-fraud.md:66`; `docs/examples/supply-chain.md:89`.

Examples pass only the first ten paths and first five motifs to agents. The path
formatter keeps node arrows and a path score but drops relation labels, edge
confidence, provenance, other evidence, and all PPR rows. This conflicts with the
workspace fidelity contract. Console previews elsewhere also use slicing and
must not be copied as complete tool output.

Required correction: distinguish a console preview from model delivery, preserve
the complete canonical result, and use an explicit domain formatter retaining
all returned paths, ordered nodes and edges, relations, confidences, aggregates,
and PPR rows. Vector filtering belongs at the consuming tool boundary, with
explicit exclusion paths. A formatter must not silently discard evidence.

### D02. Examples use IDs outside their configured node collection

Locations: `docs/index.md:118`; `docs/examples/healthcare-fraud.md:37`;
`docs/examples/supply-chain.md:37`; `docs/examples/compliance.md:37`;
`whitepaper/ARCHITECTURE.md:733`. README supply-chain and compliance snippets
also change ID prefixes without constructing a corresponding graph mapping.

The landing page configures `CaseRecords` but retrieves `entity/...`.
Healthcare configures `claims_entities` but uses `provider/...` and
`procedure/...`; supply chain configures `supply_entities` but uses
`supplier/...`; compliance configures `compliance_entities` but uses `entity/...`.
Arango extraction trains on the configured collection's full IDs. These are not
interchangeable namespaces. The scorer does not establish truth for arbitrary
out-of-vocabulary identities.

Required correction: use full IDs from the configured collection throughout each
example, and state which graph/membership records must already exist.

### D03. README compliance call has invalid keyword arguments

Location: `README.md:332`. Code: `odin/engine.py:208`.

The example calls `score_edge(head=..., relation=..., tail=...)`, but the method
accepts `src`, `rel`, and `dst`. Signature binding fails. Use positional arguments
or the real keyword names. This is a runnable-example bug, not an illustrative
signature abbreviation.

### D04. Documented hard retrieval limits are not hard limits

Locations: `docs/reference/engine.md:65`, `docs/reference/configuration.md:35`,
`docs/guides/tuning.md:11`, `docs/concepts/beam-search.md:38`.
Code: `retrieval/orchestrator.py:122`, `odin/engine.py:200`.

`OdinEngine.retrieve()` invokes the orchestrator with default dynamic expansion.
On low support, it can double max_paths and beam width and change hop_limit;
the second pass caps hop_limit at four, which can even reduce a larger requested
value. The docs present these as maximum output/path-length bounds without
explaining effective versus requested parameters or the engine's lack of a
public dynamic_scale option.

Executed reproduction: on an in-memory chain, `max_paths=1, hop_limit=3`
returned two paths with effective `max_paths=2, hop_limit=4, beam_width=128`.
The complete result is saved as `adaptive-result.json` in the audit artifacts.

Required correction: document the actual expansion policy and effective bounds.
If the intended API promises strict limits, that requires a separate code fix.

### D05. Temporal and community guidance are overstated as beam-pruning signals

Locations: `README.md:80`, `:90`; `docs/concepts/architecture.md:21`;
`whitepaper/ODIN_WHITEPAPER_ARXIV.tex:407`.
Code: `odin/engine.py:84`, `retrieval/orchestrator.py:292`, `:310`,
`retrieval/beam.py:53`, `retrieval/scoring.py:185`.

Beam extension scoring uses log-weighted PPR, relation prior, confidence, and a
timestamp lookup if one is supplied. The orchestrator does not pass its timestamp
lookup to beam_search, and the engine constructs the orchestrator without a
timestamp lookup. Final path scoring also relies on that lookup, rather than
reading canonical edge timestamps. Mapped created_at does reach aggregation.
Bridge and affinity factors are applied after beam candidate enumeration, not
in its pruning score. Therefore a bridge cannot rescue a path already pruned
by the default beam. The claim that full COMPASS decides which paths survive
each hop does not describe this code.

Required correction: explicitly separate beam extension scoring, final path
scoring, and aggregation. If mapped temporal fields are intended to affect
default engine path scores, wire and test that behavior in a separate change.

### D06. Source provenance does not imply the documented triage coverage

Locations: `docs/concepts/aggregation.md:30`, `docs/concepts/scoring.md:19`,
`docs/reference/troubleshooting.md:135`.
Code: `retrieval/adapters_arango.py:24`, `retrieval/aggregators.py:214`.

Arango emits provenance as `{assertion, sources}`. Provenance coverage recognizes
`document_id` or `source_doc`, not the canonical `sources` list or source
references in the raw assertion. The audit passed a nonempty canonical sources
list and measured coverage of 0.0. Similarly, label coverage counts u_label and
v_label, while the current Arango accessor supplies no label/node_label method;
merely adding a node label property does not make the documented advice work.

Required correction: describe the accepted aggregate input schema and the
current Arango limitation. Aligning canonical provenance/label mappings with
aggregators needs separate implementation and regression coverage.

### D07. Whitepapers retain obsolete lifecycle and storage claims

Locations: `whitepaper/ODIN_WHITEPAPER_ARXIV.tex:394`;
`whitepaper/ODIN_WHITEPAPER.tex:287`;
`whitepaper/ODIN_WHITEPAPER_EXECUTIVE.tex:419`;
`whitepaper/ODIN_WHITEPAPER.md:363`;
`whitepaper/ARCHITECTURE.md:754`.

Claims include ModelWeights storage, AMIE+ mining, fixed ten-epoch training,
rule-weights-only artifacts smaller than 1 KB, on-demand embedding reconstruction,
10–30-second first training, and graceful fallback after training failure.
Current code uses OdinModels, its own motif/prior generator, convergence-driven
E-M, persisted learned tensors and complete reports, and explicit failures.
The executive paper fixes failure wording but still retains obsolete storage
and timing claims. Architecture also describes roughly 55-second training,
2 KB artifacts, and 30-second reloads.

Required correction: preserve published papers as historical research if needed,
but clearly label their implementation/version scope and link a current
implementation-differences document. Do not cite them as current API/storage
documentation. Update maintained engineering derivatives directly.

### D08. Research performance, attribution, and deployment claims lack verification

Locations: `whitepaper/ODIN_WHITEPAPER_ARXIV.tex:442`, `:450`, `:460`, `:482`;
`whitepaper/ODIN_WHITEPAPER.tex:317`, `:325`;
`whitepaper/ODIN_WHITEPAPER.md:516`;
`whitepaper/ARCHITECTURE.md:5`, `:510`, `:588`;
`whitepaper/ODIN_WHITEPAPER.html:260`.

These include million-entity datasets, 65x fewer explored paths, 90% coverage,
expert quality ratings, ablation improvements, >80% cache hit rates, production
readiness below five million entities, and regulatory/customer outcomes.
The available local lifecycle and synthetic tests do not reproduce those
experiments. Shapley decomposition is described as implemented, but current
scoring exposes heuristic components, not a Shapley algorithm.

Required correction: attach reproducible experiment provenance and the measured
software revision, or mark these as historical/unverified research or product
claims. Lack of repository evidence does not establish that an experiment never
happened; it prevents this audit from certifying it. Enterprise capabilities in
`docs/infrastructure/enterprise.md` require their own product evidence outside
this OSS repository.

## Medium priority findings

### D09. Configuration reference still documents the removed training budget

Location: `docs/reference/configuration.md:108`.
Code: `npll/utils/config.py:135`, `npll/bootstrap.py` training configuration.

It says bootstrap uses ten epochs and five E-M iterations. Executed get_config
returns max_epochs=200 and em_iterations=10; bootstrap uses those values as
safety limits and requires convergence. Update the reference and mirror the
failure/report behavior already explained in the lifecycle guide.

### D10. Configuration reference incorrectly denies environment/global settings

Location: `docs/reference/configuration.md:7`.
Code: `retrieval/orchestrator.py:79`, `npll/utils/config.py:135`.

ODIN_REQUIRE_NPLL is read from the environment. Executed reproduction confirms
that setting it to true makes explicit constant-confidence engine construction
raise. Bootstrap also selects the module-defined OdinTriples profile internally;
the public engine does not expose every training parameter as a constructor
argument. Document these boundaries instead of saying everything is passed as
arguments and no environment variable is read.

### D11. Result schema and packaged skill deny fields that are returned

Locations: `docs/reference/result-schema.md:64`;
`odin/skills/odin/SKILL.md:94`, `:112`; `AGENTS.md:15`;
`docs/guides/ai-editors.md:9`, `:63`.
Code: `retrieval/scoring.py:279`, `retrieval/orchestrator.py:459`.

Actual paths contain score, nodes, edges, decomp, and id. id is currently None,
not an assigned path_0 string. The docs say there is no nodes field and omit the
decomposition and much of the canonical edge shape. Deriving nodes from edges
is valid, but the categorical denial is false. Update the schema and every
agent-facing copy; decide explicitly if a future contract should remove nodes.

### D12. Results are not directly JSON serializable

Locations: `docs/guides/agent-integration.md:89`;
`docs/reference/result-schema.md:136`.
Code: `retrieval/orchestrator.py:379`.

trace.params is an OrchestratorParams dataclass, not a plain dictionary.
Executed json.dumps(result) raises TypeError. Document explicit dataclass
conversion, retaining the complete result, or change the API separately.
The same caution applies to traces described as safe to log directly.

### D13. README uses the wrong model collection and promises inactive markers

Locations: `README.md:97`, `:205`.
Code: `odin/backends/arango.py:17`, `retrieval/scoring.py:284`.

The lifecycle names NPLLWeights, but storage is OdinModels. The README says
inactive signals are reported as inactive in the result; raw library output
instead contains neutral factors or empty/zero diagnostic values, without an
explicit active/inactive status contract. The demo interface's presentation must
not be generalized to the library return shape.

### D14. Complexity and PPR-to-beam explanations omit real work

Locations: `README.md:266`; `docs/concepts/beam-search.md:26`;
`docs/guides/tuning.md:19`; `docs/concepts/ppr.md:11`, `:30`;
`docs/concepts/architecture.md:21`.
Code: `retrieval/beam.py:46`, `:70`; `retrieval/orchestrator.py:295`.

Beam width bounds retained states, not the number of outgoing edges scanned.
Cost depends on degree, edge scoring, heap work, path copying, and database
latency; O(b*h) is not a general runtime bound. The tuning page simultaneously
says each extra hop multiplies work by beam width and that cost is width times
depth. Also, beam starts from the original seeds, not the top PPR anchors;
PPR scores guide ranking without restricting enumeration to an anchor whitelist.
Document these mechanics and qualify heuristic latency advice.

### D15. Examples conflate learned plausibility with verification or anomaly evidence

Locations: `docs/guides/agent-integration.md:37`;
`docs/examples/healthcare-fraud.md:55`, `:73`;
`docs/examples/compliance.md:10`, `:43`.

A high score for a proposed edge does not establish graph membership or confirm
a regulatory relationship. A low score does not establish missing support, fraud,
or clinical invalidity. Healthcare also says community_mode=mapping trains payer
norms; training is global and the mode only scopes retrieval. Mark decision
thresholds illustrative, require source/membership checks for conclusions, and
remove the scoped-training claim.

### D16. Advanced integration guide describes tool wrappers as library APIs

Locations: `whitepaper/AGENT_INTEGRATION_GUIDE.md:70`, `:78`, `:84`;
`whitepaper/ARCHITECTURE.md:154`.
Code: `retrieval/orchestrator.py:97`, `:151`, `:173`.

The guide lists text and question as arguments for mapped orchestrator calls,
but link_and_retrieve accepts Mention objects plus a base accessor and params,
and retrieve does not accept question. score_candidates uses structural BiPPR,
not learned edge confidence. Wrapper tool contracts may be proposed, but must
be identified and implemented outside these methods. Architecture imports
odin_kg_engine.retrieval, a package not supplied by this project. Replace stale
imports and distinguish adapter/wrapper preparation from engine functionality.

## Lower priority consistency and publication issues

### D17. Current editor-install instructions can fetch the wrong skill revision

Location: `docs/guides/ai-editors.md:13`, `:32`.

The page recommends pip install odin-engine or downloading the skill from main,
then teaches the unreleased feature-branch backend contract. Use the matching
checkout/package revision or label the released versus unreleased paths. The
CLI installation mechanics themselves are covered by unit tests.

### D18. Literal-looking Python blocks include pseudocode without clear labels

Locations: `README.md:357`; `docs/concepts/aggregation.md:46`;
`docs/reference/result-schema.md:77`; `odin/skills/odin/SKILL.md:60`.

Some are legitimate signature declarations or abbreviated schemas, not runnable
examples. Others mix Python fences with lowercase false, bare ellipses in dict
positions, or annotation syntax inside a function call. Label schematic blocks
or make copyable examples valid Python. The audit's full syntax candidate list
is in checks.json; do not count every candidate as a bug.

### D19. Citeability needs stable evidence and version scope

Locations: `docs/guides/npll-lifecycle.md` local validation section;
`docs/guides/production.md:11`; `BACKEND_ABSTRACTION_HANDOVER.md`.

The small compliance_docs metrics match archived results and are valid as one
local experiment. The logs and full data are outside the Git repository, however,
so an external reviewer cannot reproduce or inspect them from a source citation
alone. Publish an appropriately reviewed experiment manifest, hardware/runtime,
parameters, revision, and accessible evidence if external empirical citations
are intended. Do not publish raw application data without reviewing it.
The handover's not-pushed statement is a historical preparation-time state;
the full review range excludes later documentation corrections. Label it as a
dated snapshot. Production's phrase 'only recorded measurements' should be scoped
to this validation, since the repository also contains an older demo artifact.

## Verified areas and checks

- Backend signatures, explicit Arango mapping, optional drivers, global training
  namespaces, revision-safe storage, numeric version protection, learned-state
  loading, and explicit training failures align with the reviewed current code.
- The latest seven-method accessor skeleton passes method-presence validation.
  Its placeholder bodies are not a functioning graph implementation.
- Confidence wording corrected in the recent commits accurately distinguishes
  learned plausibility from calibrated truth.
- The lifecycle guide's 200/10 safety budget, convergence failure behavior, and
  backend-neutral optimization description match code and regression tests.
- Schema inspection signature and capability dispatch match code. Field discovery
  is sampled, not a guarantee that every field in the database has been seen.
- The triage weighting and label/low-support guards match aggregators. Their inputs
  have the provenance/label limitations in D06; thresholds do not prove accuracy.
- October 8 local lifecycle and confidence numbers match the archived reports.
- Fresh unit suite: **258 passed, seven existing Gremlin deprecation warnings**.
- Strict MkDocs build: **passed**. This validates build/navigation, not claims or
  code examples. It does not build TeX whitepapers or verify external hyperlinks.
- No database was modified, training restarted, or website published in this audit.

## Recommended repair order

1. Correct evidence-delivery examples, mismatched IDs, and invalid public calls.
2. Resolve intended retrieval limits, temporal scoring, and provenance coverage
   against the implementation; document current limits until code changes land.
3. Align API/configuration/schema/skill copies and advanced integration contracts.
4. Label historical research and unsupported empirical/product claims by version
   and evidence, without retroactively rewriting published experimental history.
5. Recheck corrected examples, run focused contract tests and strict docs build,
   then issue a new audit verdict pinned to the repaired revision.
