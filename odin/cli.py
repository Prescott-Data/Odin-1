"""Command-line tools for Odin.

Usage:
    odin --version
    odin init --skill [--target NAME ...] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

TARGETS = {
    "github": ".github",
    "claude": ".claude",
    "cursor": ".cursor",
    "agents": ".agents",
}
MANIFEST_NAME = ".odin-skill-manifest.json"
DEFAULT_TARGETS = ("claude", "agents")


class FileResult(NamedTuple):
    status: str  # installed | updated | unchanged | skipped | removed | failed
    path: str
    detail: str = ""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("odin-engine")
    except Exception:
        return "unknown"


def bundled_skills() -> Dict[str, bytes]:
    """Every file in the packaged skills tree, keyed by POSIX path under skills/."""
    found: Dict[str, bytes] = {}

    def walk(node, prefix: str) -> None:
        for child in node.iterdir():
            rel = prefix + child.name
            if child.is_dir():
                if child.name != "__pycache__":
                    walk(child, rel + "/")
            else:
                found[rel] = child.read_bytes()

    walk(files("odin").joinpath("skills"), "")
    return found


def _load_manifest(path: Path) -> Dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.get("files", {}).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def _prune_empty_dirs(path: Path, stop: Path) -> None:
    for parent in path.parents:
        if parent == stop or not parent.is_dir() or any(parent.iterdir()):
            return
        parent.rmdir()


def _install_target(root: Path, target_dir: str, bundle: Dict[str, bytes],
                    skill_version: str, force: bool) -> List[FileResult]:
    skills_dir = root / target_dir / "skills"
    manifest_path = skills_dir / MANIFEST_NAME
    previous = _load_manifest(manifest_path)
    recorded: Dict[str, str] = {}
    results: List[FileResult] = []

    def shown(dest: Path) -> str:
        return dest.relative_to(root).as_posix()

    for rel, data in sorted(bundle.items()):
        dest = skills_dir / rel
        new_hash = _sha256(data)
        try:
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                recorded[rel] = new_hash
                results.append(FileResult("installed", shown(dest)))
                continue
            current = _sha256(dest.read_bytes())
            if current == new_hash:
                recorded[rel] = new_hash
                results.append(FileResult("unchanged", shown(dest)))
            elif force or previous.get(rel) == current:
                dest.write_bytes(data)
                recorded[rel] = new_hash
                results.append(FileResult("updated", shown(dest)))
            else:
                if rel in previous:
                    recorded[rel] = previous[rel]
                results.append(FileResult(
                    "skipped", shown(dest), "modified locally; use --force to overwrite"))
        except OSError as exc:
            if rel in previous:
                recorded[rel] = previous[rel]
            results.append(FileResult("failed", shown(dest), str(exc)))

    for rel, old_hash in sorted(previous.items()):
        if rel in bundle:
            continue
        dest = skills_dir / rel
        try:
            if not dest.exists():
                continue
            if _sha256(dest.read_bytes()) == old_hash:
                dest.unlink()
                _prune_empty_dirs(dest, skills_dir)
                results.append(FileResult("removed", shown(dest), "no longer in the bundle"))
            else:
                recorded[rel] = old_hash
                results.append(FileResult(
                    "skipped", shown(dest), "obsolete but modified locally; left in place"))
        except OSError as exc:
            recorded[rel] = old_hash
            results.append(FileResult("failed", shown(dest), str(exc)))

    try:
        skills_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"version": skill_version, "files": dict(sorted(recorded.items()))},
                       indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        results.append(FileResult("failed", shown(manifest_path), str(exc)))
    return results


def install_skills(project_root, targets: Sequence[str], force: bool = False,
                   bundle: Optional[Dict[str, bytes]] = None,
                   skill_version: Optional[str] = None) -> List[FileResult]:
    """Install the skill bundle into each target and return per-file results."""
    root = Path(project_root)
    bundle = bundled_skills() if bundle is None else bundle
    skill_version = skill_version or _version()
    results: List[FileResult] = []
    for name in targets:
        results.extend(_install_target(root, TARGETS[name], bundle, skill_version, force))
    return results


def _report(results: Sequence[FileResult]) -> int:
    for r in results:
        print(f"{r.status}: {r.path}" + (f" ({r.detail})" if r.detail else ""))
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print("Summary: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    return 1 if counts.get("failed") else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="odin", description="Odin command-line tools.")
    parser.add_argument("--version", action="version", version=f"odin {_version()}")
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Set up Odin in your project.")
    p_init.add_argument("--skill", action="store_true",
                        help="Install the Odin skills for AI coding agents.")
    p_init.add_argument("--target", action="append", choices=[*TARGETS, "all"],
                        help="Where to install (repeatable): github, claude, cursor, agents, or all. "
                             "Default: claude and agents.")
    p_init.add_argument("--force", action="store_true",
                        help="Overwrite skill files you have modified.")

    args = parser.parse_args(argv)

    if args.command == "init" and args.skill:
        chosen = args.target or list(DEFAULT_TARGETS)
        targets = list(TARGETS) if "all" in chosen else list(dict.fromkeys(chosen))
        return _report(install_skills(Path.cwd(), targets, force=args.force))

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
