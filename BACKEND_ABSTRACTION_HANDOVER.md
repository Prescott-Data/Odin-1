# Backend abstraction review handover

Prepared October 8, 2026.

- Repository: Odin-1
- Branch: `feat/backend-abstraction`
- Implementation and documentation review range: `ad2bfe8..ce8af25`
- Scope at that head: 97 changed files.
- Remote state at preparation: the latest 15 commits are local and not pushed.

Review the final code across the full range, rather than treating intermediate
commits as the final contract. Some intermediate approaches, including scorer
replay and library-level vector stripping, were replaced by subsequent fixes.

## Objective and architecture

The branch changes Odin from an Arango-specific engine into an engine that
consumes explicit backend capabilities. Arango is the implemented training
backend; Neo4j implementation and parity remain deferred.

- `OdinEngine` accepts a backend rather than an Arango connection.
- Public contracts live under `odin.backends`: graph access, training snapshots,
  model storage, and schema inspection.
- `ArangoBackend` owns Arango extraction, persistence, accessors, and schema
  inspection.
- Database drivers are optional installation extras.
- Schema inspection uses `inspect_schema(backend)`.
- The configuration name is `OdinTriples`. Unknown names fail clearly, and
  configurations are returned as independent copies.
- Retrieval-only backends must explicitly disable training. Unsupported requested
  capabilities fail at the contract boundary.

Start with `odin/backends/base.py`, `odin/backends/arango.py`, `odin/engine.py`,
and `docs/development/backend-contracts.md`.

## Arango modelling contract

Users explicitly map the node collection, edge collection, and relation field.
Types, weights, timestamps, validity, status, provenance, memberships, bridges,
and affinities are opt-in mappings.

- Unconfigured community signals do not probe assumed collections.
- Configured query failures raise instead of silently becoming no signal; failed
  results are not cached.
- Missing configured provenance collections raise configuration errors.
- Full document IDs and exact relation labels survive training and retrieval.
- Queries bind configured field and collection names.
- Accessors expose explicit edge metadata; raw assertions appear under
  `provenance.assertion`. Recency and validity do not guess property names.
- Helpers return complete results. Raw documents and vectors remain unchanged;
  model-input vector filtering belongs in the consuming tool layer.
- The writer uses a connected handle and explicit mapping, and preserves user
  metadata.
- PPR and beam search share outbound-edge caching. Signal caches are invalidated
  with the retrieval cache.

Review `retrieval/adapters_arango.py`, `retrieval/cache.py`,
`retrieval/writers/arango_writer.py`, and `odin/backends/arango_schema.py`.

## Training and model storage

Training reads the global configured graph. Communities are optional retrieval
scopes and do not create separate model artifacts. `model_store()` takes no
community arguments. Retrieval-only mapping changes do not force duplicate
global models.

- Snapshot fingerprints derive from the actual extracted triples, including
  explicitly configured type triples.
- Model namespaces isolate databases and training graph mappings.
- Saves use compare-and-swap revision checks.
- On a first-boot conflict, losing workers load the compatible winner; incompatible
  conflicts still raise.
- Training failures preserve the active serving model and stored artifact.
- Current artifacts persist learned scorer tensors, rule weights, configuration,
  and complete training reports. Reload does not replay scorer training.
- Older numeric artifact versions and stale configuration schemas retrain.
  Newer versions raise `NewerModelVersionError` and cannot be overwritten by
  older workers, including through forced retraining.
- Device and Torch version differences are runtime provenance, not automatic
  invalidation of otherwise compatible learned tensors.
- Damaged current artifacts raise `CorruptModelError`. The troubleshooting guide
  documents how to export the correct hashed `OdinModels` document, delete its
  inspected revision, and initialize Odin to train again.

Review `npll/bootstrap.py`, `odin/backends/base.py`,
`tests/unit/test_model_state_loading.py`, and
`tests/unit/test_artifact_version_guard.py`.

## Scorer, rules, and convergence

- Logical rules are mined from the graph vocabulary instead of fixed Prescott
  relation names.
- Grounding uses observed supporting instances and sampled contrasts.
- Mining has confidence thresholds and bounded hub-path sampling.
- Neural scorer negatives use seeded head and tail corruptions. E-M holdout facts
  are excluded from scorer supervision and negative examples.
- E-M training must converge before saving. Safety-budget exhaustion raises
  `TrainingConvergenceError` with a complete report.
- The M-step uses configured L2 regularization and retains Adam state across E-M
  iterations. Its stopping checks include objective, weight, and gradient changes.
- ELBO and partition samples use stable local random streams.
- Backend-neutral optimizations batch rule counts, reuse weight-independent
  sampled evidence, and use indexed M-step lookups. Each weighted calculation
  builds a fresh autograd graph.
- Cache invalidation covers known-fact changes, added rules, and replaced graphs.
  Caller-supplied worlds use their own counts.
- Batching processes all worlds and preserves sample counts and convergence
  tolerances. Scalar/batched counts, values, and gradients have regression tests;
  floating-point reduction order may produce small numerical differences.

These optimizations operate on the in-memory training representation and apply
to any backend supplying the training contracts.

Review `npll/core/logical_rules.py`, `npll/core/mln.py`,
`npll/inference/elbo.py`, `npll/inference/m_step.py`, and the scorer, rule,
E-M, and telemetry unit tests.

## Optional communities

Bridge and affinity remain optional retrieval signals for users with existing
clusters. Training and storage require neither. Configured membership lookup
requires one distinct community per entity after configured algorithm filtering.
Multiple distinct memberships raise `BackendConfigurationError` during scoring;
multi-membership scoring is deferred and the current requirement is documented.

## Validation

- 258 unit tests passed.
- Six live Arango contract tests passed.
- Fatal Python lint, strict MkDocs build, and whitespace checks passed.
- Live contract tests cover renamed collections, arbitrary relation vocabulary,
  explicit metadata and community signals, and the train/save/reload/retrieve
  lifecycle.

### Fresh local lifecycle

Database: `compliance_docs`, local ArangoDB 3.12.9 on port 8530.
No community mappings were configured.

| Check | Result |
| --- | --- |
| Graph | 171 nodes, 173 edges |
| Extracted training triples | 344 |
| Fresh initialization | 97.91 seconds |
| E-M convergence | 12 epochs, 113 iterations |
| Cached reload | 0.159 seconds |
| Scores identical after reload | All 344 extracted triples |
| Stored revision changed on reload | No |
| Retrieval | 32 seeds, 292 returned paths |
| Edge confidence matches public scoring API | Every returned edge |
| Raw assertions preserved | Yes |

The old model was backed up and its inspected revision deleted before this fresh
run. Graph data remained unchanged. The newly converged model is stored in the
database.

### Confidence checks and limits

| Check | Result |
| --- | --- |
| Observed edges | 173 |
| Mean observed-edge confidence | 0.945 |
| Seeded unobserved endpoint corruptions | 1,730 |
| Mean corruption confidence | 0.080 |
| Pairwise ranking statistic | 0.9669 |

Scores express learned plausibility, not calibrated probabilities of truth.
These checks include training facts and unverified negative contrasts. They do
not establish held-out accuracy or calibration. Some observed edges scored very
low and deserve source-level inspection. Convergence is for the sampled E-M
approximation; it does not prove a global optimum or neural scorer calibration.

Timings describe this small local graph, not a scaling projection. Large-graph
performance and Neo4j parity remain untested.

## Documentation and cleanup

Migration guidance, backend contracts, installation, Arango setup, troubleshooting,
public API documentation, examples, the Odin skill, and whitepaper references
were updated throughout the branch. The latest documentation changes clarify
confidence semantics, record the local lifecycle checks, and explain optional
retrieval scopes and engine-pool identity.

Temporary testing scripts, loose logs, Python caches, and generated build output
were removed. Complete October 8 testing evidence was archived without content
filtering or truncation:

- Archive: `../verification-artifacts/odin-training-review-20261008.tar.gz`
- Manifest: `../verification-artifacts/odin-training-review-20261008-manifest.json`

The unrelated untracked `docs/jev-semantic-retrieval-plan.md` was left untouched.

## Suggested review priorities

1. Capability boundaries and public API migration, including all in-workspace
   consumers.
2. Absence of implicit Arango collection, field, vocabulary, or community
   assumptions.
3. Artifact compatibility, namespace identity, concurrency, and revision-safe
   replacement.
4. Scorer supervision, negative sampling, holdout separation, and rule grounding.
5. Cache invalidation and scalar/batched value and gradient parity.
6. Convergence criteria, complete reports, and failed-training/save protection.
7. Complete retrieval evidence and preservation of raw documents.

Neo4j implementation, multi-membership scoring, broad performance testing,
held-out accuracy, and calibration are remaining scope, not claims of this
validation.
