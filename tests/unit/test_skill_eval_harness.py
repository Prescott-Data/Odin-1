"""The skill eval harness must itself be trustworthy: good code passes, bad code fails."""
import pytest

from tests.skills import runner
from tests.skills.checks import CHECKS
from tests.skills.golden import GOLDEN

PROMPTS = runner.load_prompts()
BY_ID = {p["id"]: p for p in PROMPTS}


def test_prompt_set_is_well_formed():
    assert 10 <= len(PROMPTS) <= 15
    catalog = runner.skill_catalog()
    assert len(BY_ID) == len(PROMPTS)
    assert sum(1 for p in PROMPTS if not p["skills"]) >= 3, "need prompts where no skill should activate"
    for p in PROMPTS:
        assert set(p["checks"]) <= set(CHECKS), p["id"]
        assert set(p["skills"]) <= set(catalog), p["id"]
        assert p["id"] in GOLDEN, f"{p['id']} has no golden solution"
        assert (not p["execute"]) or p["args"] is not None


@pytest.mark.parametrize("entry_id", sorted(GOLDEN))
def test_good_solution_passes_every_check(entry_id):
    entry = BY_ID[entry_id]
    code = GOLDEN[entry_id]["good"]
    result = runner.run_solution(code, entry)
    checks = runner.evaluate(entry, code, result)
    failed = {k: v["message"] for k, v in checks.items() if not v["passed"]}
    assert not failed, failed


@pytest.mark.parametrize("entry_id", sorted(GOLDEN))
def test_bad_solution_is_caught(entry_id):
    entry = BY_ID[entry_id]
    code = GOLDEN[entry_id]["bad"]
    result = runner.run_solution(code, entry)
    checks = runner.evaluate(entry, code, result)
    for name in GOLDEN[entry_id]["fails"]:
        assert not checks[name]["passed"], f"{name} did not catch the bad solution"


class ScriptedProvider:
    """Answers activation with the expected skills and generation with golden code."""

    def __init__(self, entries, quality):
        self.entries, self.quality = entries, quality

    def _entry(self, user):
        return next(e for e in self.entries if user.startswith(e["prompt"]))

    def complete(self, system, user):
        if system.startswith("You grade outputs"):
            return '{"pass": true, "reason": "scripted"}'
        entry = self._entry(user)
        if system.startswith("You are a coding agent with these skills"):
            import json
            return json.dumps(entry["skills"])
        return "```python\n" + GOLDEN[entry["id"]][self.quality] + "\n```"


def test_runner_scores_good_and_bad_providers_differently():
    good = runner.run_sample(ScriptedProvider(PROMPTS, "good"), ScriptedProvider(PROMPTS, "good"),
                             BY_ID["score_one_edge"], with_skills=True)
    assert good["passed"] and good["activation_ok"] and good["judge"]

    bad = runner.run_sample(ScriptedProvider(PROMPTS, "bad"), None, BY_ID["score_one_edge"],
                            with_skills=False)
    assert not bad["passed"] and bad["mode"] == "baseline" and bad["activation_ok"] is None


def test_skill_context_loads_skill_and_references():
    text = runner.skill_context(["odin"])
    assert "=== odin/SKILL.md ===" in text and "=== odin/references/errors.md ===" in text
