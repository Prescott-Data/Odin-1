"""User-journey acceptance test for the packaged Odin skills.

README -> install the matching version -> skills are discoverable ->
refresh after an upgrade never clobbers user edits.

Builds a real wheel and installs it with --no-deps into a clean venv. The CLI
imports lazily, so no torch or database driver is needed.
"""
import hashlib
import json
import re
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TARGET_DIRS = (".github", ".claude", ".cursor", ".agents")
DEFAULT_DIRS = (".claude", ".agents")
SKILLS = ("odin", "odin-custom-backend")
MANIFEST = ".odin-skill-manifest.json"
LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)#\s]+)")
REFERENCE = re.compile(r"`?(references/[\w./-]+\.md)`?")


def _run(cmd, cwd, check=True):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version = "([^"]+)"', text, re.M).group(1)


@pytest.fixture(scope="session")
def installed(tmp_path_factory):
    """Wheel built from this checkout, installed into a clean venv."""
    work = tmp_path_factory.mktemp("journey")
    dist = work / "dist"
    _run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
          "--outdir", str(dist), str(ROOT)], cwd=work)
    wheel = next(dist.glob("odin_engine-*.whl"))
    env_dir = work / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    py = env_dir / "bin" / "python"
    _run([str(py), "-m", "pip", "install", "--no-deps", "--quiet", str(wheel)], cwd=work)
    return {"python": py, "odin": env_dir / "bin" / "odin", "wheel": wheel}


@pytest.fixture
def project(tmp_path):
    return tmp_path


def _init(installed, project, *args):
    return _run([str(installed["odin"]), "init", "--skill", *args], cwd=project, check=False)


def test_readme_points_at_skill_install():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "odin init --skill" in readme


def test_source_version_matches_package_and_wheel(installed):
    version = _source_version()
    init_text = (ROOT / "odin" / "__init__.py").read_text(encoding="utf-8")
    assert f'__version__ = "{version}"' in init_text
    assert f"odin_engine-{version}-" in installed["wheel"].name
    out = _run([str(installed["odin"]), "--version"], cwd=ROOT)
    assert version in out.stdout


def test_default_install_writes_every_skill_to_claude_and_agents(installed, project):
    result = _init(installed, project)
    assert result.returncode == 0, result.stdout + result.stderr
    for target in DEFAULT_DIRS:
        for skill in SKILLS:
            assert (project / target / "skills" / skill / "SKILL.md").is_file()
    assert not (project / ".github").exists() and not (project / ".cursor").exists()
    assert result.stdout.count("installed") >= len(DEFAULT_DIRS) * len(SKILLS)


def test_target_all_writes_every_skill_to_every_location(installed, project):
    result = _init(installed, project, "--target", "all")
    assert result.returncode == 0, result.stdout + result.stderr
    for target in TARGET_DIRS:
        for skill in SKILLS:
            assert (project / target / "skills" / skill / "SKILL.md").is_file()


def test_installed_skills_reference_only_files_that_exist(installed, project):
    assert _init(installed, project, "--target", "all").returncode == 0
    for target in TARGET_DIRS:
        for skill in SKILLS:
            skill_dir = project / target / "skills" / skill
            text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
            wanted = set(LINK.findall(text)) | set(REFERENCE.findall(text))
            for rel in wanted:
                assert (skill_dir / rel).resolve().is_file(), f"{skill}: missing {rel}"


def test_manifest_records_version_and_hashes(installed, project):
    assert _init(installed, project).returncode == 0
    manifest_path = project / ".claude" / "skills" / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == _source_version()
    assert manifest["files"], "manifest lists no files"
    for rel, digest in manifest["files"].items():
        assert _sha(project / ".claude" / "skills" / rel) == digest


def test_refresh_keeps_user_edits_and_reports_them(installed, project):
    assert _init(installed, project).returncode == 0
    edited = project / ".claude" / "skills" / "odin" / "SKILL.md"
    edited.write_text(edited.read_text(encoding="utf-8") + "\nmine\n", encoding="utf-8")

    result = _init(installed, project)
    assert result.returncode == 0
    assert "mine" in edited.read_text(encoding="utf-8")
    assert any("skipped" in line and ".claude" in line for line in result.stdout.splitlines())

    forced = _init(installed, project, "--force")
    assert forced.returncode == 0
    assert "mine" not in edited.read_text(encoding="utf-8")


def test_single_target_limits_what_is_written(installed, project):
    assert _init(installed, project, "--target", "cursor").returncode == 0
    assert (project / ".cursor" / "skills" / "odin" / "SKILL.md").is_file()
    assert not (project / ".github").exists()
