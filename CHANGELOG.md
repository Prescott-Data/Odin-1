# Changelog

All notable changes to **Odin-1** (`odin-engine`) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Reworked the README to distinguish Odin's graph-navigation role from the
  consuming agent's interpretation role and removed performance claims that
  were not linked to reproducible repository evidence.
- Replaced the static test badge with the GitHub Actions CI badge.
- Clarified that the quickstart requires a populated graph and linked to the
  complete setup guide.
- Modernized the package's MIT license metadata.
- Removed the no-op Docker image, which implied a deployable service even
  though Odin is distributed as a Python library.

### Added
- **NPLL training telemetry and convergence diagnostics.** Every training run
  now produces a `TrainingReport` — complete per-iteration ELBO history,
  per-iteration max rule-weight deltas, convergence status, and the criteria
  used — persisted alongside the model weights and rehydrated on
  cached-weight loads. `KnowledgeBootstrapper.ensure_model_ready` returns a
  `BootstrapResult` (model, source, data hash, report); `OdinEngine` exposes
  `engine.training_report`, reports `npll_source` / `npll_converged` /
  `npll_trained_at` in `get_status()`, and logs a warning whenever the active
  model — freshly trained or cached — did not converge.
- Unit test coverage for the NPLL E-M machinery: ELBO computation on
  hand-checkable graphs, variational inference budgets, E-step posterior
  validity, M-step weight-update consistency, and trainer loop termination.
- Security reporting guidance and GitHub issue and pull-request templates.
- Brand Studio-aligned Odin `0.3.0` demo visuals and the complete canonical
  retrieval artifact behind the README example.
- A compact architecture visual showing the seed-to-evidence retrieval loop and
  the responsibility boundary between Odin and the consuming agent.

### Fixed
- `ELBOComputer.compute_elbo` raised `AttributeError` whenever a graph had no
  unknown facts (or no sampled worlds), because its scalar helper was defined
  on a different class. Surfaced by the new E-M unit tests.
- Replaced stale `PRODUCTION_FIXES.md` references in the architecture guide
  with the current production deployment guide.

## [0.3.0] - 2026-08-10

### Added
- `odin` command-line tool. `odin init --skill` installs the packaged AI-editor
  skill into `.github/skills/odin/` and `.claude/skills/odin/`, where GitHub
  Copilot and Claude Code load it automatically. Use `--force` to refresh.
- AI-editor skill packaged in the wheel (`odin/skills/odin/SKILL.md`), an
  **AI Editor Setup** guide, and a served `llms.txt` index of the docs.

## [0.2.1] - 2026-08-10

First release from the public **Odin-1** repository
(<https://github.com/Prescott-Data/Odin-1>).

### Changed
- Project metadata now points at the public GitHub repository and the
  documentation site (`https://odin.developers.prescottdata.io`).
- Trimmed runtime requirements to the packages the engine actually imports
  (`torch`, `python-arango`, `numpy`, `scipy`, `networkx`, `scikit-learn`,
  `gremlinpython`).

### Added
- Full documentation site (MkDocs Material).
- `CHANGELOG.md`, `CONTRIBUTING.md`, and `CODE_OF_CONDUCT.md`.
- GitHub Actions for CI, PyPI publishing (trusted publishing), and docs
  deployment.

### Notes
- No public API changes. Code is identical to the `0.2.0` release on PyPI.

## [0.2.0] - 2026-02-04

### Added
- Runtime schema introspection via `SchemaInspector` and
  `inspect_arango_schema`.

### Features (baseline)
- `OdinEngine` retrieval pipeline: Personalized PageRank -> Beam Search ->
  NPLL edge scoring -> aggregation (motifs, relation shares, triage score).
- Self-managing NPLL lifecycle (auto-train, weights persisted in ArangoDB).
- Edge plausibility scoring and PPR-based anchor discovery.

[Unreleased]: https://github.com/Prescott-Data/Odin-1/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Prescott-Data/Odin-1/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Prescott-Data/Odin-1/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Prescott-Data/Odin-1/releases/tag/v0.2.0
