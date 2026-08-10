---
icon: material/robot-happy
---

# AI Editor Setup

If you write Odin code with an AI coding agent, Cursor, Claude Code, GitHub Copilot, Windsurf, or a CLI agent, give it Odin context first. Without it, agents fall back on generic graph-library patterns and get the details wrong: inventing a `path["nodes"]` field, mixing up `score_edge` arguments, or guessing the result shape. With it, they write correct Odin from the first line.

Odin ships a ready-made **skill** for exactly this.

## The Odin skill

The repository includes a self-contained brief for AI agents at [`skills/odin/SKILL.md`](https://github.com/Prescott-Data/Odin-1/blob/main/skills/odin/SKILL.md). It covers the exact API signatures, the real `retrieve()` result shape, the edge and motif structures, adapters, and the specific mistakes agents make with Odin. It is short by design, so it fits comfortably in an agent's context.

Point your tool at that file, or paste its contents into whatever rules mechanism your tool uses. The sections below show where.

## Cursor

Add the skill as a project rule. Create `.cursor/rules/odin.mdc` and paste the contents of `SKILL.md` into it, with a short frontmatter so Cursor applies it when you touch Python:

```mdc
---
description: How to write correct Odin (odin-engine) code
globs: ["**/*.py"]
alwaysApply: false
---

<paste the contents of skills/odin/SKILL.md here>
```

Alternatively, add the docs site to Cursor's `@Docs` (URL `https://odin.developers.prescottdata.io`) so you can reference it inline with `@Odin`.

## Claude Code and CLI agents

Most CLI agents read an `AGENTS.md` at the repository root. Create one in your project and either paste the skill or point to it:

```markdown
# AGENTS.md

## Odin (odin-engine)

Follow the Odin skill when writing graph-retrieval code:
<paste the contents of skills/odin/SKILL.md here>
```

If your agent supports installable skills, drop the whole `skills/odin/` folder into its skills directory so it loads on demand.

## GitHub Copilot

Copilot reads `.github/copilot-instructions.md`. Add an Odin section there:

```markdown
# Copilot instructions

When code imports `odin` (the odin-engine package), follow these rules:
<paste the key rules from skills/odin/SKILL.md here>
```

## Any LLM or chat

Two options that need no setup:

- Every page on the [documentation site](https://odin.developers.prescottdata.io) has a **Copy page as Markdown** action (top-right of the content). Copy the pages you need, Getting Started and the Reference section are the highest-signal, and paste them into your chat.
- Point the model directly at the [Reference](../reference/index.md) pages, which carry the authoritative API and result schema.

## What good looks like

Once the skill is loaded, your agent should:

- import from `odin` and construct `OdinEngine(db, ...)` with a connected ArangoDB handle,
- call `retrieve(seeds, max_paths=..., hop_limit=..., beam_width=...)` with correct defaults,
- read `result["paths"][i]["edges"]` and derive nodes from `u`/`v` instead of a non-existent `nodes` field,
- use `score_edge(src, rel, dst)` in the right order,
- and read motifs as `edge_count` / `path_count`, not `count`.

If you see an agent reach for `path["nodes"]` or a bare `relation_share` float, the skill is not loaded.

## Next

- [Your First Retrieval](first-retrieval.md) is the code the skill teaches an agent to write.
- [Reference](../reference/index.md) is the authoritative source the skill points to.
