# ArangoGraphConfig

`ArangoGraphConfig` (`odin.backends.arango`) maps your collections and fields to Odin. Start with the minimum and add a group only when you need its feature. Omitted fields disable the matching signal. Odin never falls back to a guessed property name. Every string field must be a non-empty string or the constructor raises `ValueError`.

Authoritative guide: https://odin.developers.prescottdata.io/guides/arangodb/

## 1. Minimum (always required)

```python
graph = ArangoGraphConfig(
    node_collection="CaseRecords",
    edge_collection="EvidenceLinks",
    relation_field="predicate",
)
```

- `node_collection`: document collection whose `_id` values are seeds and node identities.
- `edge_collection`: edge collection whose `_from` and `_to` point at those `_id` values.
- `relation_field`: field holding the relation label. Every edge used for training needs a non-empty string here.

Edges whose endpoints are missing are left out of training. An invalid relation value stops training with an error rather than being coerced.

## 2. Entity types (training)

Add when nodes carry a type and you want it learned.

- `entity_type_field`: node field with the type. Each non-null value must be a non-empty string. Odin trains it as `has_type` triples.

## 3. Edge weight (traversal)

- `edge_weight_field`: numeric edge field used as structural traversal weight. Without it every edge has weight 1.0. Odin never assumes a field named `weight`.

## 4. Time and status (aggregation and filtering)

Add only when the graph has these signals.

- `edge_timestamp_field`: becomes `created_at` on result edges and feeds recency.
- `edge_valid_from_field`, `edge_valid_to_field`: validity window metadata.
- `edge_status_field`: status metadata.
- `allowed_edge_statuses`: tuple of status values to keep. Requires `edge_status_field`.

## 5. Provenance (traceability)

- `edge_provenance_fields`: tuple of edge fields to carry as provenance. The raw stored edge is always kept under `provenance["assertion"]`.
- `provenance_edge_collection`: edge collection linking entities to source records. A configured collection that does not exist raises `BackendConfigurationError`.
- `provenance_target_collections`: tuple restricting those links to source collections.

## 6. Communities (scoped retrieval, optional)

Needed only for `community_mode="mapping"` or `"property"`. Training and model storage do not need them.

- Mapping mode: set `membership_collection`, `membership_entity_field` and `membership_community_field` together, or `BackendConfigurationError` is raised. Each entity must resolve to one community; several distinct communities raise `BackendConfigurationError`.
- Property mode: set `community_property_field`.
- Algorithm filters: `community_algorithm` is the value to match, with `membership_algorithm_field`, `bridge_algorithm_field` or `affinity_algorithm_field` naming the field to filter on.

## 7. Bridges and affinity (cross-community signals, optional)

Each signal is independent. Unconfigured signals issue no queries, and a configured query that fails raises `BackendIOError`.

- Bridges: `bridge_collection`, `bridge_entity_field`, `bridge_strength_field`, `bridge_community_field`. Store full document IDs in the entity field.
- Affinity: `affinity_collection`, `affinity_from_field`, `affinity_to_field`, `affinity_score_field`. It needs mapped membership or a mapped community property.

## 8. What decides the trained model

Training reads the whole configured graph. The stored model is keyed by database, node collection, edge collection, `relation_field` and `entity_type_field`. Changing groups 3 to 7 does not create a second model or force retraining. Changing the graph's triples does.
