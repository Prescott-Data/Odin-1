---
icon: material/file-document
---

# Reference

The complete API surface, parameters, and data shapes.

<div class="grid cards" markdown>

-   :material-engine: **OdinEngine API**

    Every method on the main entry point.

    [Read more →](engine.md)

-   :material-table-search: **SchemaInspector API**

    Runtime schema discovery.

    [Read more →](schema.md)

-   :material-cog: **Configuration**

    All constructor and retrieval parameters with defaults.

    [Read more →](configuration.md)

-   :material-code-json: **Result Schema**

    The exact dictionary returned by `retrieve()`.

    [Read more →](result-schema.md)

-   :material-lifebuoy: **Troubleshooting**

    Common issues and how to resolve them.

    [Read more →](troubleshooting.md)

</div>

---

## Public imports

```python
from odin import OdinEngine, SchemaInspector, inspect_arango_schema
```

| Symbol | Purpose |
|--------|---------|
| `OdinEngine` | Main entry point — retrieval, scoring, anchors | 
| `SchemaInspector` | Runtime ArangoDB schema discovery |
| `inspect_arango_schema` | One-call schema export helper |

`odin.__version__` reports the installed version.
