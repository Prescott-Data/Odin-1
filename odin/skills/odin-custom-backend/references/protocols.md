# Backend protocols

Sources: `odin/backends/base.py` (protocols and errors), `retrieval/adapters.py` (`GraphAccessor`), `odin/engine.py` (construction checks). Narrative: https://odin.developers.prescottdata.io/development/backend-contracts/

`GraphBackend`, `TrainingBackend`, `TripleSource` and `ModelStore` are importable from `odin.backends`. `GlobalAccessBackend`, `SchemaInspectionBackend` and `SchemaIntrospector` are in `odin.backends.base`.

## Backend capabilities

```python
class GraphBackend(Protocol):
    def accessor(self, community_id: str, community_mode: str) -> GraphAccessor: ...

class TrainingBackend(Protocol):
    def triple_source(self) -> TripleSource: ...
    def model_store(self) -> ModelStore: ...

class GlobalAccessBackend(Protocol):
    def global_accessor(self) -> Optional[GraphAccessor]: ...

class SchemaInspectionBackend(Protocol):
    def schema_inspector(self) -> Optional[SchemaIntrospector]: ...
```

`accessor` is always required. Training needs both `triple_source` and `model_store` to exist and return non-`None`; otherwise the engine raises `BackendCapabilityError`. `inspect_schema(backend)` raises `BackendCapabilityError` when there is no inspector.

## Accessor surface

`OdinEngine` checks these seven methods at construction.

```text
def iter_out(self, node: str) -> Iterable[Tuple[str, str, float]]       # (neighbor, relation, weight)
def iter_in(self, node: str) -> Iterable[Tuple[str, str, float]]        # (neighbor, relation, weight)
def iter_out_edges(self, node: str) -> Iterable[Dict[str, object]]      # edge records, below
def nodes(self, community_id: str) -> Iterable[str]
def degree(self, node: str) -> int                                      # out-degree
def get_node(self, node_id: str, fields: Optional[List[str]] = None) -> Dict[str, object]  # {} when absent
def community_seed_norm(self, community_id: str, seeds: List[str]) -> List[str]
```

A method inherited unchanged from the `GraphAccessor` protocol is treated as missing.

### Edge records from `iter_out_edges`

Required keys: `_id` (stable unique edge identity), `u` (source, the `node` argument), `rel` (exact relation label), `v` (target), `weight` (number, `1.0` when the database has none).

Optional keys, `None` or omitted when unavailable: `created_at`, `valid_from`, `valid_to`, `status`, `provenance`. The Arango accessor also returns `source_confidence`, `npll_posterior` and `calibration`, and keeps the raw stored edge under `provenance["assertion"]`. Edges from `iter_out` and `iter_out_edges` must be the same edges.

## Training source

```python
class TripleSource(Protocol):
    def snapshot(self) -> TrainingSnapshot: ...
```

`TrainingSnapshot(triples)` takes `(source, relation, target)` tuples of non-empty strings. It sorts them, keeps duplicates and computes `data_hash` (SHA-256 of their compact UTF-8 JSON). Read the data once. Training covers the whole configured graph, even when retrieval is scoped to a community. Entity IDs must match the accessor's IDs. Type information, if any, is expressed as `(entity, "has_type", type)` triples.

## Model store

```python
class ModelStore(Protocol):
    def load(self, key: str) -> Optional[StoredModel]: ...
    def save(self, key: str, document: Dict[str, Any], *, expected_revision: Optional[str]) -> str: ...
```

`StoredModel(document, revision)` carries an opaque revision string.

- `load` returns `None` only when the artifact is absent. A damaged current artifact raises `CorruptModelError`. A stored version newer than this Odin raises `NewerModelVersionError` and must never be overwritten. Older versions are returned with their revision so bootstrap can retrain.
- `save` is an atomic compare-and-swap. `expected_revision=None` means create only; a revision means replace only if unchanged. It returns the new revision. A conflict raises `ModelConflictError`, with no retry and no overwrite.
- Backend failures raise `BackendIOError`.
- Scope the store by namespace (database plus training graph mapping) so different graphs never share an artifact. The artifact is global; retrieval communities share it.
- Store the complete JSON document without trimming any field.

## Errors

The full list is in `odin/backends/base.py`: `BackendError`, `BackendConfigurationError`, `BackendCapabilityError`, `BackendIOError`, `CorruptModelError`, `NewerModelVersionError`, `ModelConflictError`. Training raises `npll.TrainingError` and `npll.TrainingConvergenceError`.

## Reference implementations

`odin/backends/arango.py` (`ArangoBackend`, full), `retrieval/adapters.py` (`KGCommunityAccessor`, in-memory accessor), `tests/utils/backend_fakes.py` (in-memory source and store for tests). `JanusGraphAccessor` in `retrieval/adapters_janus.py` is a retrieval accessor only.
