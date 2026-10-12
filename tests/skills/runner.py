"""Behavioral evals for the Odin skills. Run before releases, not in CI.

A model is asked to write code for each prompt in prompts.json, with and
without the skills. The code is checked statically, executed against an
in-memory backend (no database), and optionally scored by a judge model.

Stages per sample:
  1. activation: the model sees only skill names and descriptions and says which
     skills it would load. Compared with the prompt's expected skills.
  2. generation: the chosen skills' SKILL.md and reference files are given as
     context. All references are loaded, so reference selection is not tested.
  3. checks: deterministic checks in checks.py, plus the judge rubric if any.

Examples (needs `pip install -e ".[evals]"`):
  python -m tests.skills.runner --provider azure --env-file path/to/.env --samples 3
  ANTHROPIC_API_KEY=... python -m tests.skills.runner --provider anthropic --model MODEL
  python -m tests.skills.runner --provider command --command "claude -p" --mode skill

Generated code is executed. Run this only in a disposable environment.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Optional

from tests.skills.checks import CHECKS
from tests.skills.exec_case import MARK
from tests.skills.providers import make_provider

ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = Path(str(files("odin").joinpath("skills")))
PROMPTS = Path(__file__).with_name("prompts.json")
CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.S)
SUFFIX = ("\n\nAssume odin-engine 0.4.0 is installed. The caller supplies the arguments of "
          "`solve`. Reply with one Python code block and nothing else.")


def load_prompts(ids: Optional[List[str]] = None) -> List[dict]:
    entries = json.loads(PROMPTS.read_text(encoding="utf-8"))
    return [e for e in entries if not ids or e["id"] in ids]


def skill_catalog() -> Dict[str, str]:
    """name -> description, read from each SKILL.md frontmatter."""
    out = {}
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        head = path.read_text(encoding="utf-8").split("---")[1]
        name = re.search(r"^name: (.+)$", head, re.M).group(1).strip()
        out[name] = re.search(r"^description: (.+)$", head, re.M).group(1).strip()
    return out


def skill_context(names: List[str]) -> str:
    parts = []
    for name in names:
        for path in sorted((SKILLS_DIR / name).rglob("*.md")):
            parts.append(f"=== {name}/{path.relative_to(SKILLS_DIR / name).as_posix()} ===\n"
                         + path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def choose_skills(provider, entry: dict) -> List[str]:
    catalog = skill_catalog()
    listing = "\n".join(f"- {n}: {d}" for n, d in catalog.items())
    system = ("You are a coding agent with these skills available. Decide which, if any, "
              "you would load before doing the task. Reply with only a JSON list of skill "
              f"names, possibly empty.\n\n{listing}")
    reply = provider.complete(system, entry["prompt"])
    match = re.search(r"\[.*?\]", reply, re.S)
    try:
        return [n for n in json.loads(match.group(0)) if n in catalog] if match else []
    except ValueError:
        return []


def extract_code(reply: str) -> str:
    match = CODE_BLOCK.search(reply)
    return (match.group(1) if match else reply).strip() + "\n"


def run_solution(code: str, entry: dict) -> dict:
    blank = {"returned": None, "stdout": "", "exception": None, "message": "",
             "score_edge_calls": [], "path_counts": [], "post_retrieve": None}
    if not entry["execute"]:
        return blank
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(code, encoding="utf-8")
        try:
            done = subprocess.run(
                [sys.executable, "-m", "tests.skills.exec_case", str(path), json.dumps(entry["args"])],
                cwd=ROOT, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return {**blank, "exception": "Timeout", "message": "solution did not finish in 60s"}
    for line in done.stdout.splitlines():
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    return {**blank, "exception": "HarnessError", "message": done.stderr}


def evaluate(entry: dict, code: str, result: dict) -> Dict[str, dict]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {name: {"passed": False, "message": f"syntax error: {exc}"} for name in entry["checks"]}
    ctx = {"entry": entry, "code": code, "tree": tree, "result": result}
    return {name: dict(zip(("passed", "message"), CHECKS[name](ctx))) for name in entry["checks"]}


def judge(provider, entry: dict, result: dict) -> Dict[str, dict]:
    verdicts = {}
    answer = (result.get("returned") or "") + "\n" + (result.get("stdout") or "")
    for rubric in entry.get("judge", []):
        reply = provider.complete(
            "You grade outputs. Reply with only JSON: {\"pass\": true or false, \"reason\": \"...\"}.",
            f"Task: {entry['prompt']}\n\nOutput to grade:\n{answer}\n\nQuestion: {rubric}")
        match = re.search(r"\{.*\}", reply, re.S)
        try:
            data = json.loads(match.group(0))
            verdicts[rubric] = {"passed": bool(data["pass"]), "message": str(data.get("reason", ""))}
        except (ValueError, KeyError, AttributeError):
            verdicts[rubric] = {"passed": False, "message": "unparseable judge reply"}
    return verdicts


def run_sample(provider, judge_provider, entry: dict, with_skills: bool) -> dict:
    chosen = choose_skills(provider, entry) if with_skills else []
    system = ("You are a coding agent. Follow any skill instructions below.\n\n"
              + skill_context(chosen)) if chosen else "You are a coding agent."
    reply = provider.complete(system, entry["prompt"] + SUFFIX)
    code = extract_code(reply)
    result = run_solution(code, entry)
    sample = {
        "mode": "skill" if with_skills else "baseline",
        "activation_ok": (sorted(chosen) == sorted(entry["skills"])) if with_skills else None,
        "chosen_skills": chosen,
        "reply": reply, "code": code, "execution": result,
        "checks": evaluate(entry, code, result),
        "judge": judge(judge_provider, entry, result) if judge_provider and entry.get("judge") else None,
    }
    sample["passed"] = all(c["passed"] for c in sample["checks"].values()) and \
        all(j["passed"] for j in (sample["judge"] or {}).values())
    return sample


def summarize(records: List[dict]) -> str:
    lines = [f"{'prompt':26} {'mode':9} {'pass':>8} {'activation':>11}"]
    for entry_id in dict.fromkeys(r["id"] for r in records):
        for mode in ("baseline", "skill"):
            rows = [s for r in records if r["id"] == entry_id for s in r["samples"] if s["mode"] == mode]
            if not rows:
                continue
            act = [s["activation_ok"] for s in rows if s["activation_ok"] is not None]
            lines.append(f"{entry_id:26} {mode:9} {sum(s['passed'] for s in rows):>3}/{len(rows):<4} "
                         + (f"{sum(act):>6}/{len(act):<4}" if act else f"{'-':>11}"))
    return "\n".join(lines)


def load_env_file(path: str) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing values or printing them."""
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", choices=("anthropic", "openai", "azure", "command"), required=True)
    ap.add_argument("--env-file", help="file with KEY=VALUE lines, e.g. AZURE_* settings")
    ap.add_argument("--model", default="", help="model, or Azure deployment (default: AZURE_DEPLOYMENT)")
    ap.add_argument("--judge-model", default="")
    ap.add_argument("--command", default="")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--mode", choices=("skill", "baseline", "both"), default="both")
    ap.add_argument("--only", nargs="*", help="prompt ids to run")
    ap.add_argument("--out", default="skill-eval-results.json")
    args = ap.parse_args(argv)
    if args.env_file:
        load_env_file(args.env_file)

    provider = make_provider(args.provider, args.model, args.command)
    judge_provider = make_provider(args.provider, args.judge_model, args.command) \
        if args.judge_model else provider
    modes = {"skill": [True], "baseline": [False], "both": [False, True]}[args.mode]

    records = []
    for entry in load_prompts(args.only):
        samples = [run_sample(provider, judge_provider, entry, m) for m in modes for _ in range(args.samples)]
        records.append({"id": entry["id"], "samples": samples})
    Path(args.out).write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(summarize(records))
    print(f"\nFull results (complete replies, code and execution output): {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
