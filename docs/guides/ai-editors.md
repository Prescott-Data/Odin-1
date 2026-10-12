---
icon: material/robot-happy
title: AI Editor Setup - Copilot, Claude Code, Cursor
description: Teach your AI coding agent Odin. Install the version-matched Odin skills with odin init --skill, choose where they go, refresh safely after an upgrade, and point tools without skill discovery at them.
---

# AI Editor Setup

Your coding agent writes better Odin code when it knows the library. Odin ships that knowledge as two skills inside the package, so they always describe the version you have installed:

| Skill | Use it for |
|---|---|
| `odin` | Using the engine: setup, API, the `retrieve()` result, evidence handling, errors, migration from 0.3.0 |
| `odin-custom-backend` | Implementing a backend for another database |

Each skill is a short `SKILL.md` that loads automatically when a task matches it, plus reference files the agent reads only when needed.

## Install the skills

Install Odin first, then run this from your project root:

```bash
odin init --skill
```

By default this writes both skills to every location below. Use `--target` (repeatable) to pick some:

```bash
odin init --skill --target github --target claude
```

| `--target` | Directory | Read by |
|---|---|---|
| `github` | `.github/skills/` | GitHub Copilot |
| `claude` | `.claude/skills/` | Claude Code |
| `cursor` | `.cursor/skills/` | Cursor |
| `agents` | `.agents/skills/` | Tools that use the shared `.agents` location |

Commit the files so everyone who opens the project gets an agent that knows Odin. The command prints one line per file (`installed`, `updated`, `unchanged`, `skipped`, `removed` or `failed`) and exits non-zero if anything failed.

!!! note "Source checkout"
    Until the next release is published, install from the checkout that contains the backend API (`pip install -e ".[arango]"`) and run `odin init --skill` from that environment. The skills come from the installed package, not from the network, so they cannot get ahead of or behind the library.

## Refresh after an upgrade

The skills are versioned with the package. After upgrading Odin, run the same command:

```bash
odin init --skill
```

It records a manifest (`.odin-skill-manifest.json` in each `skills/` directory) with the skill version and a hash of every installed file, then:

- updates files you have not touched,
- skips files you edited and tells you which ones,
- removes files that the new version no longer ships, if you have not edited them.

Add `--force` to overwrite files you edited.

## Tools without skill discovery

Tools that read `AGENTS.md` can be pointed at the skill with one line in your project's `AGENTS.md`:

```markdown
When working with odin (odin-engine) code, read .github/skills/odin/SKILL.md first.
```

## Any tool: llms.txt

The documentation site serves [llms.txt](https://odin.developers.prescottdata.io/llms.txt), a curated index of these docs for AI ingestion. Point research agents or custom tooling at it instead of scraping. The skills fall back to the same index when a reference file is missing.

## What good looks like

With the skill loaded, your agent should construct `OdinEngine(backend, ...)` with a backend such as `ArangoBackend(db, graph)` and an explicit `ArangoGraphConfig`, call `retrieve(seeds, ...)`, derive nodes from `path["edges"]`, keep relation labels and provenance when it formats evidence, and treat an exception differently from an empty result. If you see it reach for `OdinEngine(db)` or depend on undocumented fields such as `path["nodes"]`, the skill is not loaded.

## Next

- [Your First Retrieval](first-retrieval.md) is the code the skill teaches an agent to write.
- [AI Agent Integration](agent-integration.md) covers patterns for putting Odin in an agent loop.
- [Reference](../reference/index.md) is the authoritative source the skills point to.
