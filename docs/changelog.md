---
icon: material/history
hide:
  - toc
---

# Changelog

All notable changes to **Odin** (`odin-engine`) are documented here. This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html): patch releases carry backward-compatible fixes, minor releases add backward-compatible features, and major releases may break the public API.

---

<div class="changelog-release" markdown>

## Unreleased

The engine now accepts `OdinEngine(backend)`. ArangoDB operations are provided
by `ArangoBackend(db, graph)`, with a required explicit Arango graph mapping,
public `odin.backends` modules, optional driver extras, and generic
`inspect_schema(backend)` inspection. Raw handles and former public adapter
exports require migration.

Training fingerprints cover the exact extracted triples. Model artifacts are
shared by global training scope, complete, and replaced only when their stored revision matches.
Scorers train on known facts and uniform corruptions of both endpoints, excluding
the E-M holdout. Artifact version `7.0` loads learned tensor state without training
and keeps complete loss histories. Older artifact versions and stale config schemas
retrain; newer artifact versions raise without being replaced. Damaged blobs raise.
Device and Torch version differences do not invalidate learned state.
Rule mining and grounding use bounded observed joins and confidence thresholds.
Global accessor traversal bonuses and the unused engine global accessor are removed.
Signal helpers return every record; their limit argument is removed. Membership
ambiguity raises and lookup results are cached. Writers preserve user metadata.
Odin returns complete application documents and vectors without exclusion markers;
model-input filtering belongs to consuming applications.
Training failures raise; constant confidence requires explicit
`auto_train=False`. Failed retraining preserves the active serving state.

See [Backend migration](guides/backend-migration.md) before upgrading.
These changes have not been released to PyPI.

</div>

<div class="changelog-release" markdown>

## 0.3.0 <span class="changelog-date">2026-08-10</span>

<div class="changelog-meta" markdown>
<div class="changelog-contributors">
<a href="https://github.com/ekizito96" title="Muyukani Kizito"><img src="https://github.com/ekizito96.png?size=32" alt="ekizito96"></a>
</div>
</div>

Tooling to make AI coding agents write correct Odin.

**Added**

- The `odin` command-line tool. `odin init --skill` installs the packaged AI-editor skill into `.github/skills/odin/` and `.claude/skills/odin/`, where GitHub Copilot and Claude Code load it automatically. Run `odin init --skill --force` to refresh after upgrades.
- The AI-editor skill is packaged in the wheel, so it is versioned with the code and works offline.
- An **AI Editor Setup** guide and a served [`llms.txt`](https://odin.developers.prescottdata.io/llms.txt) index of the documentation for AI ingestion.

</div>

<div class="changelog-release" markdown>

## 0.2.1 <span class="changelog-date">2026-08-10</span>

<div class="changelog-meta" markdown>
<div class="changelog-contributors">
<a href="https://github.com/ekizito96" title="Muyukani Kizito"><img src="https://github.com/ekizito96.png?size=32" alt="ekizito96"></a>
</div>
</div>

First release from the public [Odin-1 repository](https://github.com/Prescott-Data/Odin-1). This is a packaging and documentation release; the engine code is identical to `0.2.0`, so there are no public API changes.

**Changed**

- Project metadata now points at the public GitHub repository and the documentation site at [odin.developers.prescottdata.io](https://odin.developers.prescottdata.io).
- Trimmed runtime requirements to the packages the engine actually imports: `torch`, `python-arango`, `numpy`, `scipy`, `networkx`, `scikit-learn`, and `gremlinpython`.

**Added**

- Full documentation site built with MkDocs Material, covering concepts, guides, examples, and reference.
- `CHANGELOG.md`, `CONTRIBUTING.md`, and `CODE_OF_CONDUCT.md`.
- GitHub Actions for CI, PyPI publishing via trusted publishing, and documentation deployment.
- The Odin brand-asset system (monogram, wordmark, and combo lockups).

</div>

<div class="changelog-release" markdown>

## 0.2.0 <span class="changelog-date">2026-02-04</span>

<div class="changelog-meta" markdown>
<div class="changelog-contributors">
<a href="https://github.com/ekizito96" title="Muyukani Kizito"><img src="https://github.com/ekizito96.png?size=32" alt="ekizito96"></a>
</div>
</div>

The baseline public release of the Odin engine on PyPI.

**Added**

- Runtime schema introspection via `SchemaInspector` and `inspect_arango_schema`, giving agents live ArangoDB structure for writing valid queries.

**Baseline features**

- The `OdinEngine` retrieval pipeline: Personalized PageRank, then beam search, then NPLL edge scoring, then aggregation into motifs, relation shares, and a triage score.
- A self-managing NPLL lifecycle that trains from your graph on first run and persists the learned weights in ArangoDB.
- Edge plausibility scoring via `score_edge()` and PPR-based anchor discovery via `find_anchors()`.

</div>
