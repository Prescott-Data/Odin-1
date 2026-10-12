"""Tests for `odin init --skill`: bundle installation, manifest and refresh."""
import hashlib
import json
import os
import stat

import pytest

from odin import cli
from odin.cli import MANIFEST_NAME, TARGETS, bundled_skills, install_skills, main

BUNDLE_V1 = {"odin/SKILL.md": b"v1 skill", "odin/references/a.md": b"v1 a"}


def statuses(results):
    return {(r.status, r.path) for r in results}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(root, target=".github"):
    return json.loads((root / target / "skills" / MANIFEST_NAME).read_text(encoding="utf-8"))


def test_packaged_bundle_has_valid_frontmatter_and_no_em_dashes():
    bundle = bundled_skills()
    assert "odin/SKILL.md" in bundle
    for rel, data in bundle.items():
        text = data.decode("utf-8")
        assert "\u2014" not in text, f"{rel} contains an em dash"
        if rel.endswith("/SKILL.md"):
            assert text.startswith("---\n"), rel
            assert f"name: {rel.split('/')[0]}\n" in text, rel
            assert "description:" in text, rel


def test_fresh_install_writes_bundle_and_manifest_to_each_target(tmp_path):
    results = install_skills(tmp_path, list(TARGETS), bundle=BUNDLE_V1, skill_version="0.4.0")
    assert all(r.status == "installed" for r in results if not r.path.endswith(MANIFEST_NAME))
    for target in TARGETS.values():
        assert (tmp_path / target / "skills" / "odin" / "references" / "a.md").read_bytes() == b"v1 a"
    data = manifest(tmp_path)
    assert data["version"] == "0.4.0"
    assert data["files"]["odin/SKILL.md"] == sha(b"v1 skill")


def test_refresh_updates_unmodified_files(tmp_path):
    install_skills(tmp_path, ["github"], bundle=BUNDLE_V1)
    v2 = {"odin/SKILL.md": b"v2 skill", "odin/references/a.md": b"v1 a"}
    results = install_skills(tmp_path, ["github"], bundle=v2)
    assert ("updated", ".github/skills/odin/SKILL.md") in statuses(results)
    assert ("unchanged", ".github/skills/odin/references/a.md") in statuses(results)
    assert (tmp_path / ".github/skills/odin/SKILL.md").read_bytes() == b"v2 skill"
    assert manifest(tmp_path)["files"]["odin/SKILL.md"] == sha(b"v2 skill")


def test_refresh_skips_user_modified_file(tmp_path):
    install_skills(tmp_path, ["github"], bundle=BUNDLE_V1)
    edited = tmp_path / ".github/skills/odin/SKILL.md"
    edited.write_bytes(b"mine")
    v2 = {"odin/SKILL.md": b"v2 skill", "odin/references/a.md": b"v2 a"}

    results = install_skills(tmp_path, ["github"], bundle=v2)

    assert ("skipped", ".github/skills/odin/SKILL.md") in statuses(results)
    assert ("updated", ".github/skills/odin/references/a.md") in statuses(results)
    assert edited.read_bytes() == b"mine"
    assert manifest(tmp_path)["files"]["odin/SKILL.md"] == sha(b"v1 skill")


def test_force_overwrites_user_modified_file(tmp_path):
    install_skills(tmp_path, ["github"], bundle=BUNDLE_V1)
    edited = tmp_path / ".github/skills/odin/SKILL.md"
    edited.write_bytes(b"mine")

    results = install_skills(tmp_path, ["github"], force=True, bundle=BUNDLE_V1)

    assert ("updated", ".github/skills/odin/SKILL.md") in statuses(results)
    assert edited.read_bytes() == b"v1 skill"


def test_preexisting_untracked_file_is_not_overwritten(tmp_path):
    existing = tmp_path / ".github/skills/odin/SKILL.md"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"someone else's")

    results = install_skills(tmp_path, ["github"], bundle=BUNDLE_V1)

    assert ("skipped", ".github/skills/odin/SKILL.md") in statuses(results)
    assert existing.read_bytes() == b"someone else's"
    assert "odin/SKILL.md" not in manifest(tmp_path)["files"]


def test_obsolete_unmodified_file_is_removed(tmp_path):
    install_skills(tmp_path, ["github"], bundle=BUNDLE_V1)
    v2 = {"odin/SKILL.md": b"v1 skill"}

    results = install_skills(tmp_path, ["github"], bundle=v2)

    assert ("removed", ".github/skills/odin/references/a.md") in statuses(results)
    assert not (tmp_path / ".github/skills/odin/references").exists()
    assert "odin/references/a.md" not in manifest(tmp_path)["files"]


def test_obsolete_modified_file_is_kept(tmp_path):
    install_skills(tmp_path, ["github"], bundle=BUNDLE_V1)
    kept = tmp_path / ".github/skills/odin/references/a.md"
    kept.write_bytes(b"mine")

    results = install_skills(tmp_path, ["github"], force=True, bundle={"odin/SKILL.md": b"v1 skill"})

    assert ("skipped", ".github/skills/odin/references/a.md") in statuses(results)
    assert kept.read_bytes() == b"mine"


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions")
def test_failures_are_reported_and_exit_nonzero(tmp_path, monkeypatch, capsys):
    locked = tmp_path / ".github" / "skills"
    locked.mkdir(parents=True)
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "bundled_skills", lambda: BUNDLE_V1)
        code = main(["init", "--skill", "--target", "github"])
    finally:
        locked.chmod(stat.S_IRWXU)
    out = capsys.readouterr().out
    assert code == 1
    assert "failed: .github/skills/odin/SKILL.md" in out
    assert "installed" not in out


def test_cli_reports_per_file_and_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "bundled_skills", lambda: BUNDLE_V1)
    assert main(["init", "--skill", "--target", "claude", "--target", "agents"]) == 0
    out = capsys.readouterr().out
    assert "installed: .claude/skills/odin/SKILL.md" in out
    assert "installed: .agents/skills/odin/SKILL.md" in out
    assert not (tmp_path / ".github").exists()
    assert out.rstrip().splitlines()[-1].startswith("Summary: ")


def test_default_target_is_all(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "bundled_skills", lambda: BUNDLE_V1)
    assert main(["init", "--skill"]) == 0
    for target in TARGETS.values():
        assert (tmp_path / target / "skills" / "odin" / "SKILL.md").is_file()
