# Errors

Every exception below is raised, never swallowed. None of them is a sign that the graph is simply empty. Import the backend errors from `odin.backends.base` (or `odin.backends`) and the training errors from `npll`.

Authoritative page: https://odin.developers.prescottdata.io/reference/troubleshooting/

| Exception | Cause | Fix |
| --- | --- | --- |
| `BackendError` | Base class. Also raised for invalid triple identities: an Arango entity ID or relation label that is empty or not a string, or an entity type that is not a string. | Fix the offending documents or the `relation_field` / `entity_type_field` mapping. |
| `BackendConfigurationError` | The backend or mapping is incomplete or inconsistent: a backend missing a required accessor method, a configured provenance collection that does not exist, a partial membership mapping, a non-numeric mapped weight, or an entity with several distinct community memberships. | Read the message, then complete the `ArangoGraphConfig` group it names (see `arango-config.md`) or the missing accessor method. |
| `BackendCapabilityError` | A requested capability the backend does not supply, for example training on a retrieval-only backend or schema inspection without an inspector. | Supply the capability, or pass `auto_train=False` on purpose for structural retrieval. |
| `BackendIOError` | The database call failed (connection, permission, query). Failures are not cached as "no data". | Check connectivity, credentials and permissions, and the original cause attached to the exception. |
| `CorruptModelError` | A stored model artifact fails validation. It is not overwritten automatically; `retrain_model()` does not bypass it. | Export and delete only that `OdinModels` document, then initialize the engine again. See the troubleshooting page. |
| `NewerModelVersionError` | The stored model was written by a newer Odin. It must not be replaced by an older reader. | Upgrade Odin. Keep the artifact. |
| `ModelConflictError` | Another writer created, replaced or deleted the artifact since it was read. Odin does not retry. | Coordinate training across workers and load the current artifact. |
| `npll.TrainingError` | Requested training failed or produced no model: an empty snapshot, no generated rules, a non-finite loss, or a trainer failure. The cause is chained. | Inspect `__cause__` and the graph's triples. The active model and stored artifact are unchanged. |
| `npll.TrainingConvergenceError` | Subclass of `TrainingError`. Training ran out of its safety budget before converging. | Read `error.report` (a `TrainingReport` with the full history). No replacement was saved and the previous model keeps serving. Do not switch to constant scores silently. |

## Rules for handling them

- Do not wrap `OdinEngine(...)` or `retrain_model()` in a bare `except` that falls back to empty results or constant scoring.
- `auto_train=False` is a deliberate choice for structural retrieval, not an error handler. Use it only when the user asked for that.
- Show the user the exception type and message. Include `TrainingConvergenceError.report` values when they are relevant.
- An empty `retrieve()` result is not an exception. See `result-schema.md`.
