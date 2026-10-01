# Backend contracts

`OdinEngine(backend)` accepts a `GraphBackend` and obtains database operations
through its capabilities. `KnowledgeBootstrapper` receives a training source
and model store. Raw database handles raise a migration error. These contracts
are unreleased; see [Backend migration](../guides/backend-migration.md) for
installation and API changes. Neo4j engine support is not included yet.

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
of database or iteration order. Backend training scope belongs to the
store namespace, not the training-data digest.

Arango extracts relationship and type triples in one AQL query. Entity IDs are
full `_id` strings, matching retrieval. Relationship labels preserve case and
spaces; missing or non-string labels fail extraction. Type triples are
`(entity._id, "has_type", entity.type)`, with a string type value. Dangling
relationships are excluded, as in the previous extractor. Training uses the globally configured node and edge collections;
training is global even when retrieval is scoped to a community.

This intentionally replaces the old bare-key/lowercased training identities.
Existing weights must be retrained; there is no compatibility lookup.
Rules contain unconditional relation priors plus chains and symmetry supported by
observed facts. False premises do not count as support. Exact relation names,
ordered joins, and deterministic rule IDs are preserved for arbitrary vocabularies.

## Model artifacts and concurrency

`ModelStore.load(key)` returns `StoredModel(document, revision)` or `None` only
when absent. `save(key, document, expected_revision=...)` returns the new opaque
revision. A `None` expected revision means create only. An existing revision
means replace only if the artifact has not changed. Bootstrap reads this token
before training, including forced retraining. It does not retry a rejected save. Startup may reload a fully matching winner;
forced retraining still propagates conflicts.

The Arango namespace is the canonical JSON array of database name, global training
scope, node collection, edge collection, relation field, and entity type field.
The storage key hashes `[namespace, logical_model_key]`; both values are checked
on load. Retrieval communities share this global artifact.

The envelope contains `namespace`, `model_key`, and the complete `artifact`,
alongside Arango `_key` / `_rev` metadata. Create uses non-overwriting insert.
Update uses document replacement with `_rev` and `check_rev=True`, so removed
nested fields are actually removed and competing writers cannot silently win.

Artifact version `6.0` requires:

- `model_type`, `storage_type`, `trained_at`, `data_hash`, and `version`;
- `inference_state` with the complete config, seed, scorer recipe, runtime version,
  example count, full loss history, and explicitly excluded embedding paths;
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

Unit regressions cover same-count mutations, snapshot immutability, one-read
bootstrap, 101 relations, 120 history entries, complete nested persistence,
namespaces, corruption, and conflicting creation/replacement/forced retraining.
Live checks in `tests/integration/test_arango_backend_live.py` create a disposable
database and verify AQL identity, actual revision conflicts, complete artifact
round-trip, and real training followed by reload and scoring.

The neural scorer learns observed triples against deterministic unobserved
corruptions, including relation-conditioned scores. Such corruptions are training
examples, not proof that a relationship is false. Learned edge confidence is not
source truth or a calibrated probability. The E-M loop separately learns MLN rule
weights. Its convergence report describes that loop, not scorer calibration.

Internal embeddings are excluded from persistence. Reload replays the complete
scorer training recipe on the same snapshot, verifies its loss history, and applies
the saved rule weights. Replay incurs computation at startup. Changes to the
runtime, profile, or recipe invalidate reuse; malformed artifacts fail clearly.
Cross-backend retrieval parity and Neo4j live tests are not established yet.

Run the live tests with an account allowed to create databases:

```bash
ODIN_TEST_ARANGO_URL=http://127.0.0.1:8529 \
ODIN_TEST_ARANGO_PASSWORD=your-test-password \
pytest tests/integration/test_arango_backend_live.py -v
```
