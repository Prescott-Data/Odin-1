"""Tests for `odin init --skill`: AI-editor skill installation."""
import importlib.resources as ir

from odin.cli import copy_skill


def _skill_text() -> str:
    return ir.files("odin").joinpath("skills", "odin", "SKILL.md").read_text(encoding="utf-8")


def test_packaged_skill_exists_with_valid_frontmatter():
    text = _skill_text()
    assert text.startswith("---\n")
    assert "name: odin" in text
    assert "description:" in text
    assert "\u2014" not in text  # house style: no em dashes


def test_skill_teaches_the_real_api_surface():
    text = _skill_text()
    for token in ("OdinEngine", "score_edge(src, rel, dst)", '"edges"', "edge_count"):
        assert token in text, f"skill is missing '{token}'"


def test_skill_installs_to_both_editor_locations(tmp_path):
    assert copy_skill(tmp_path, force=False) is True
    for editor_dir in (".github", ".claude"):
        dest = tmp_path / editor_dir / "skills" / "odin" / "SKILL.md"
        assert dest.exists()
        assert "OdinEngine" in dest.read_text(encoding="utf-8")


def test_skill_does_not_overwrite_without_force(tmp_path):
    target = tmp_path / ".github" / "skills" / "odin" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("customised")

    copy_skill(tmp_path, force=False)
    assert target.read_text() == "customised"  # preserved without force
    assert (tmp_path / ".claude" / "skills" / "odin" / "SKILL.md").exists()

    copy_skill(tmp_path, force=True)
    assert target.read_text() != "customised"  # force overwrites
