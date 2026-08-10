---
icon: material/robot-happy
title: AI Editor Setup - Copilot, Claude Code, Cursor
description: Teach your AI coding editor Odin. Install the Odin skill for GitHub Copilot and Claude Code, wire up Cursor with AGENTS.md, and point any tool at llms.txt.
---

# AI Editor Setup

Your AI coding editor writes better Odin code when it knows the library. Odin ships that knowledge as a skill: the correct API contracts, the exact `retrieve()` result shape, and the mistakes we see most, maintained in the same repo as the code it describes. Load it once and your agent stops inventing a `path["nodes"]` field and starts writing Odin that runs.

## Install the skill

After `pip install odin-engine`, run this from your project root:

```bash
odin init --skill
```

It writes the skill to the locations editors discover automatically:

| Editor | Location | Loaded |
|---|---|---|
| GitHub Copilot | `.github/skills/odin/` | Automatically, when a task matches the skill description |
| Claude Code | `.claude/skills/odin/` | Automatically, same trigger model |

Commit these files. Everyone who opens the project gets an editor that knows Odin.

Prefer not to install the package first? Drop the skill in by hand instead:

```bash
mkdir -p .github/skills/odin .claude/skills/odin
curl -sL https://raw.githubusercontent.com/Prescott-Data/Odin-1/main/odin/skills/odin/SKILL.md \
  -o .github/skills/odin/SKILL.md
cp .github/skills/odin/SKILL.md .claude/skills/odin/SKILL.md
```

## Cursor and other AGENTS.md tools

Tools that read `AGENTS.md` (Cursor, Codex, and others) get the same knowledge with one line in your project's `AGENTS.md`:

```markdown
When working with odin (odin-engine) code, read .github/skills/odin/SKILL.md first.
```

## Any tool: llms.txt

The documentation site serves [llms.txt](https://odin.developers.prescottdata.io/llms.txt), a curated index of these docs for AI ingestion. Point research agents or custom tooling at it instead of scraping.

## Keeping it current

The skill is versioned with the package. After upgrading Odin, refresh it:

```bash
odin init --skill --force
```

## What good looks like

Once the skill is loaded, your agent should construct `OdinEngine(db, ...)` with a connected ArangoDB handle, call `retrieve(seeds, max_paths=..., hop_limit=..., beam_width=...)`, read `result["paths"][i]["edges"]` and derive nodes from `u`/`v`, use `score_edge(src, rel, dst)` in the right order, and read motifs as `edge_count` / `path_count`. If you see it reach for `path["nodes"]`, the skill is not loaded.

## Next

- [Your First Retrieval](first-retrieval.md) is the code the skill teaches an agent to write.
- [Reference](../reference/index.md) is the authoritative source the skill points to.
