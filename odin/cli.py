"""Command-line tools for Odin.

Usage:
    odin --version
    odin init --skill [--force]
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from importlib.resources import files as _files
except ImportError:  # Python < 3.9 fallback
    from importlib_resources import files as _files  # type: ignore

EDITOR_DIRS = (".github", ".claude")


def _skill_text() -> str:
    """Read the packaged Odin skill shipped inside the wheel."""
    resource = _files("odin").joinpath("skills", "odin", "SKILL.md")
    return resource.read_text(encoding="utf-8")


def copy_skill(project_root, force: bool = False) -> bool:
    """Install the packaged Odin skill into editor-discovered locations.

    Writes the skill to ``<root>/.github/skills/odin/SKILL.md`` and
    ``<root>/.claude/skills/odin/SKILL.md``. Existing files are preserved
    unless ``force`` is True. Returns True if any file was written.
    """
    text = _skill_text()
    root = Path(project_root)
    wrote = False
    for editor in EDITOR_DIRS:
        dest = root / editor / "skills" / "odin" / "SKILL.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            continue
        dest.write_text(text, encoding="utf-8")
        wrote = True
    return wrote


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("odin-engine")
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="odin", description="Odin command-line tools.")
    parser.add_argument("--version", action="version", version=f"odin {_version()}")
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Set up Odin in your project.")
    p_init.add_argument(
        "--skill",
        action="store_true",
        help="Install the Odin AI-editor skill for GitHub Copilot and Claude Code.",
    )
    p_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing skill file.",
    )

    args = parser.parse_args(argv)

    if args.command == "init" and args.skill:
        copy_skill(Path.cwd(), force=args.force)
        print("Installed the Odin skill to .github/skills/odin/ and .claude/skills/odin/.")
        print("Commit these files so every editor in your project knows Odin.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
