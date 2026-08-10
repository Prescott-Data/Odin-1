# Contributing to Odin-1

Thanks for your interest in improving **Odin-1**, the open-source graph
intelligence engine. This document explains how to set up your environment,
the standards we hold code to, and how to get changes merged.

## Ways to contribute

- **Bug reports**: open an issue with a minimal reproduction.
- **Features & algorithms**: PPR variants, new aggregators, scalability work.
- **Database adapters**: Neo4j, Neptune, or other graph backends.
- **Benchmarks**: academic dataset comparisons and regression suites.
- **Documentation**: clarifications, examples, and guides.

## Development setup

```bash
git clone https://github.com/Prescott-Data/Odin-1.git
cd Odin-1

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Running a graph database

Integration tests and the retrieval pipeline expect an ArangoDB instance.

```bash
docker run -d --name arango -p 8529:8529 \
  -e ARANGO_NO_AUTH=1 arangodb:3.12
```

## Running tests

```bash
# Fast unit tests (no database required)
pytest tests/unit -v

# Integration tests (require ArangoDB on localhost:8529)
pytest tests/integration -v

# Everything, with coverage
pytest tests/ --cov=odin --cov=npll --cov=retrieval --cov-report=term-missing
```

## Code standards

- **Style**: keep changes idiomatic and minimal; do not reformat unrelated code.
- **Lint**: `flake8 odin npll retrieval` should pass.
- **Security**: `bandit -r odin npll retrieval` should not introduce new issues.
- **Types**: prefer explicit type hints on public functions.
- **Tests**: new behavior needs tests; bug fixes should include a regression test.

## Commit & PR guidelines

1. Branch from `main` (e.g. `feat/neptune-adapter`, `fix/ppr-cache-key`).
2. Keep PRs focused; one logical change per PR.
3. Update `CHANGELOG.md` under `[Unreleased]`.
4. Describe the motivation and testing performed in the PR description.
5. Ensure CI is green.

## Versioning & releases

We follow [Semantic Versioning](https://semver.org/): patch for backward-compatible
fixes, minor for backward-compatible features, major for breaking changes.

### Cutting a release (maintainers)

1. Bump the version in `pyproject.toml` **and** `odin/__init__.py` (`__version__`),
   and in `mkdocs.yml` (`extra.version`). Keep them in sync.
2. Move the `[Unreleased]` notes in `CHANGELOG.md` under a new `## [X.Y.Z]` heading,
   add a matching entry to `docs/changelog.md`, and update the compare links at the
   bottom of `CHANGELOG.md`.
3. Merge to `main` with CI green.
4. Tag and push:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

5. Create a GitHub **Release** for `vX.Y.Z` (use the changelog section as the notes).
   Publishing the release triggers `.github/workflows/publish.yml`, which builds the
   distributions and uploads them to PyPI via trusted publishing.

### One-time setup

- **PyPI trusted publisher**: on the `odin-engine` PyPI project, add a GitHub
  publisher for owner `Prescott-Data`, repository `Odin-1`, workflow `publish.yml`,
  environment `pypi`. No API tokens are stored in the repo.
- **Docs / GitHub Pages**: enable Pages from the `gh-pages` branch. The
  `.github/workflows/docs.yml` workflow deploys the site on every push to `main`
  that touches the docs, and the `docs/CNAME` file points it at
  `odin.developers.prescottdata.io`.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) that covers this project.
