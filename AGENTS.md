# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

Odin-1 (`odin-engine`) is a Python library for guided knowledge-graph retrieval:
Personalized PageRank, beam search, and NPLL edge scoring over an ArangoDB (or
custom-adapter) graph. Public API: `from odin import OdinEngine, SchemaInspector`.

## Writing Odin code

When writing or reviewing code that uses the engine, follow the Odin skill at
[`odin/skills/odin/SKILL.md`](odin/skills/odin/SKILL.md). It has the exact API signatures,
the real `retrieve()` result shape, and the mistakes to avoid. In particular:

- Paths have no `nodes` field. Read `path["edges"]` (each edge has `u`, `v`,
  `relation`) and derive nodes from them.
- Motif fields are `edge_count` / `path_count`, not `count`. `relation_share[rel]`
  is `{"count", "share"}`.
- `score_edge(src, rel, dst)`; `retrieve(seeds, max_paths=50, hop_limit=3, beam_width=64)`.
- `OdinEngine` takes a connected `db` handle, not a connection string.

## Commands

```bash
pip install -e ".[dev]"          # install with dev tools
pytest tests/unit -v             # fast unit tests (no database)
pytest tests/integration -v      # needs ArangoDB on localhost:8529
flake8 odin npll retrieval       # lint
pip install -r docs/requirements.txt && mkdocs build --strict   # build docs
```

## Documentation conventions

- The docs site lives in `docs/` (MkDocs Material). Keep prose flowing and
  concise; do not use em dashes.
- Code examples in docs must match the real engine. When in doubt, check
  `retrieval/orchestrator.py` and `retrieval/aggregators.py` for the result shape.
