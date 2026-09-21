# Backend contracts (PR 1 and PR 2)

PR 1 extracts Arango database operations from `KnowledgeBootstrapper`. PR 2
makes the public engine backend-neutral: `OdinEngine(backend)` accepts a
`GraphBackend` and never constructs an Arango backend internally. Raw database
handles raise a migration error. Neo4j backend support remains a subsequent
phase.

## Engine capabilities

Every engine backend supplies an accessor with `iter_out`, `iter_in`, `nodes`,
`degree`, `get_node`, and `community_seed_norm`. The engine validates this
surface at construction rather than relying on deferred attribute errors.

Backends may be retrieval-only. They initialize with `auto_train=False` and
use constant confidence. Automatic training and `retrain_model()` require both
a non-null `TripleSource` and `ModelStore`; otherwise the engine raises
`BackendCapabilityError`. Unexpected bootstrap failures propagate rather than
being converted into constant confidence.

## Snapshot identity

`TripleSource.snapshot()` returns a frozen `TrainingSnapshot`. Its triples are
immutable, lexicographically ordered `(source, relation, target)` string tuples.
Duplicates are retained. SHA-256 covers the UTF-8 JSON serialization of these
exact tuples (`ensure_ascii=False`, compact separators). There is no separate
fingerprint query, count-based identity, or second extraction during bootstrap.
Two sources returning the same triples get the same fingerprint, irrespective
of database or iteration order. Database/community identity belongs to the
store namespace, not the training-data digest.

Arango extracts relationship and type triples in one AQL query. Entity IDs are
full `_id` strings, matching retrieval. Relationship labels preserve case and
spaces; missing or non-string labels fail extraction. Type triples are
`(entity._id, "has_type", entity.type)`, with a string type value. Dangling
relationships are excluded, as in the previous extractor. Training continues
to use the global `ExtractedEntities` / `ExtractedRelationships` graph;
community-scoped training is not introduced in this PR.

This intentionally replaces the old bare-key/lowercased training identities.
Existing weights must be retrained; there is no compatibility lookup.
The pre-existing generic self-rule is selected by the minimum relation name
so that reloading cannot attach its weight to an arbitrary set iteration result.

## Model artifacts and concurrency

`ModelStore.load(key)` returns `StoredModel(document, revision)` or `None` only
when absent. `save(key, document, expected_revision=...)` returns the new opaque
revision. A `None` expected revision means create only. An existing revision
means replace only if the artifact has not changed. Bootstrap reads this token
before training, including forced retraining. It does not retry a rejected save.

The Arango namespace is the canonical JSON array of database name, entity
collection, relationship collection, community ID, and community mode. The
storage key hashes `[namespace, logical_model_key]`; both original values are
also stored and checked on load. Two communities cannot overwrite each other's
artifacts even though training currently reads the global graph.

The envelope contains `namespace`, `model_key`, and the complete `artifact`,
alongside Arango `_key` / `_rev` metadata. Create uses non-overwriting insert.
Update uses document replacement with `_rev` and `check_rev=True`, so removed
nested fields are actually removed and competing writers cannot silently win.

Artifact version `3.0` requires:

- `model_type`, `storage_type`, `trained_at`, `data_hash`, and `version`;
- all `rule_weights` and ordered `rules` (ID, text, confidence);
- `schema_snapshot` with entity, relation, and fact counts and every relation name;
- a complete `training_report`, including both iteration histories and convergence criteria.

The format is finite JSON with string object keys. Complete nested values and
additional JSON evidence fields are preserved. No relation or history cap is
applied. Unscoped `OdinModels/npll_current` documents remain untouched and are
not reused. Incompatible or incomplete documents encountered in the new
namespace fail validation rather than triggering a compatibility path.

Errors have distinct meanings:

| Outcome | Meaning |
| --- | --- |
| `None` | Artifact or model collection is absent |
| `CorruptModelError` | Unsupported/incomplete artifact or invalid envelope |
| `BackendIOError` | Extraction, transport, permission, or persistence failure |
| `ModelConflictError` | Another writer created, replaced, or deleted the artifact since the read |

A valid artifact whose saved rules no longer match the current rule-generation
code is *stale*, not corrupt: bootstrap treats it as a cache miss and retrains
against the store revision it read. Corruption is reserved for artifacts that
fail schema validation.

These errors propagate through bootstrap and the engine's initialization and
retraining methods. Trainer exceptions raise `npll.TrainingError`, preserving
the original cause. Bootstrap can return a failed result for an empty snapshot
or no generated rules; the public engine treats this as `TrainingError` too.
Only explicit `auto_train=False` selects constant confidence. Retraining builds
replacement serving components before changing the active engine state.

## Scope and verification

Updated direct bootstrap consumers: engine initialization, engine retraining,
the factory, telemetry tests, and the whitepaper example. The other workspace
implementation in `odin-kg-engine` has its own bootstrap and serving lifecycle;
it is not migrated as part of this public-repository extraction.
Scout, Guided Scout, and community-summarizer install private Azure-hosted
`odin_engine` wheels (their Dockerfiles pin 0.4.5, 0.4.3, and 0.4.4 respectively),
rather than this public checkout. Community-summarizer's direct bootstrap call
therefore remains on that private contract. No compatibility adapter is added
to the public bootstrapper.

Unit regressions cover same-count mutations, snapshot immutability, one-read
bootstrap, 101 relations, 120 history entries, complete nested persistence,
namespaces, corruption, and conflicting creation/replacement/forced retraining.
Live checks in `tests/integration/test_arango_backend_live.py` create a disposable
database and verify AQL identity, actual revision conflicts, complete artifact
round-trip, and real training followed by reload and scoring.

The persistence contract covers the existing weights-only artifact. It does
not assert identical neural scores after rebuilding a fresh model: embeddings
and scoring-network parameters are not persisted by the existing design.
Cross-backend retrieval parity and Neo4j live tests remain later-phase work.

Run the live tests with an account allowed to create databases:

```bash
ODIN_TEST_ARANGO_URL=http://127.0.0.1:8529 \
ODIN_TEST_ARANGO_PASSWORD=your-test-password \
pytest tests/integration/test_arango_backend_live.py -v
```
