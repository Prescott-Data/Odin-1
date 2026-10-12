"""Keep the bundled skills true to the code they describe."""
import ast
import dataclasses
import importlib
import inspect
import re
from importlib.resources import files
from pathlib import Path

import pytest

import npll
from odin.backends import base
from odin.backends.arango import ArangoGraphConfig
from odin.engine import OdinEngine
from retrieval.adapters import GraphAccessor

SKILLS = Path(str(files("odin").joinpath("skills")))
SKILL_DIRS = sorted(p for p in SKILLS.iterdir() if (p / "SKILL.md").is_file())
FENCE = re.compile(r"```(\w*)\n(.*?)```", re.S)
LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)#\s]+)")
REFERENCE = re.compile(r"`(references/[\w./-]+\.md)`")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def blocks(text, lang):
    return [body for tag, body in FENCE.findall(text) if tag == lang]


def spec(source: str):
    """(names, defaults) of a function header written as source text."""
    fn = ast.parse(f"def f({source}): pass").body[0].args
    names = [a.arg for a in fn.args if a.arg != "self"] + [a.arg for a in fn.kwonlyargs]
    positional = fn.defaults
    pos_args = [a.arg for a in fn.args if a.arg != "self"]
    defaults = {n: ast.literal_eval(d) for n, d in zip(pos_args[-len(positional):], positional)} \
        if positional else {}
    for a, d in zip(fn.kwonlyargs, fn.kw_defaults):
        if d is not None:
            defaults[a.arg] = ast.literal_eval(d)
    return names, defaults


def real_spec(func):
    params = [p for p in inspect.signature(func).parameters.values() if p.name != "self"]
    names = [p.name for p in params]
    defaults = {p.name: p.default for p in params if p.default is not inspect.Parameter.empty}
    return names, defaults


def test_every_skill_has_a_directory_matching_its_name():
    assert [p.name for p in SKILL_DIRS] == ["odin", "odin-custom-backend"]
    for d in SKILL_DIRS:
        assert f"name: {d.name}\n" in read(d / "SKILL.md")


@pytest.mark.parametrize("skill", SKILL_DIRS, ids=lambda p: p.name)
def test_files_named_in_a_skill_exist(skill):
    text = read(skill / "SKILL.md")
    for rel in set(LINK.findall(text)) | set(REFERENCE.findall(text)):
        assert (skill / rel).is_file(), f"{skill.name} references missing {rel}"
    for ref in (skill / "references").glob("*.md"):
        for rel in LINK.findall(read(ref)):
            assert (ref.parent / rel).resolve().is_file(), f"{ref.name} links to missing {rel}"


def test_engine_signatures_in_the_odin_skill_match_the_code():
    text = read(SKILLS / "odin" / "SKILL.md")
    api = next(b for b in blocks(text, "text") if "OdinEngine(" in b)
    checked = 0
    for line in api.strip().splitlines():
        call, args = re.match(r"(\S+?)\((.*)\)$", line.strip()).groups()
        target = OdinEngine.__init__ if call == "OdinEngine" else getattr(OdinEngine, call.split(".")[1])
        assert spec(args) == real_spec(target), f"{call} drifted from the code"
        checked += 1
    assert checked >= 6


def test_protocol_signatures_in_the_backend_skill_match_the_code():
    text = read(SKILLS / "odin-custom-backend" / "references" / "protocols.md")
    owners = (GraphAccessor, base.GraphBackend, base.TrainingBackend, base.GlobalAccessBackend,
              base.SchemaInspectionBackend, base.TripleSource, base.ModelStore)
    checked = set()
    for block in blocks(text, "python") + blocks(text, "text"):
        for line in block.splitlines():
            m = re.match(r"\s*def (\w+)\((.*?)\)\s*(?:->.*?)?(?::\s*\.\.\.)?\s*(?:#.*)?$", line)
            if not m:
                continue
            name, args = m.groups()
            owner = next(o for o in owners if hasattr(o, name))
            assert spec(args) == real_spec(getattr(owner, name)), f"{name} drifted from the code"
            checked.add(name)
    assert {"iter_out", "iter_out_edges", "community_seed_norm", "accessor", "snapshot",
            "load", "save", "triple_source", "model_store"} <= checked


def test_every_exception_class_is_documented():
    errors = read(SKILLS / "odin" / "references" / "errors.md")
    classes = [n for n, c in inspect.getmembers(base, inspect.isclass)
               if c.__module__ == base.__name__ and issubclass(c, Exception)]
    classes += ["TrainingError", "TrainingConvergenceError"]
    assert len(classes) >= 9
    for name in classes:
        assert f"`{name}`" in errors or f"`npll.{name}`" in errors, f"errors.md omits {name}"
    for name in ("TrainingError", "TrainingConvergenceError"):
        assert hasattr(npll, name)


def test_every_graph_config_field_is_documented():
    doc = read(SKILLS / "odin" / "references" / "arango-config.md")
    for field in dataclasses.fields(ArangoGraphConfig):
        assert f"`{field.name}`" in doc, f"arango-config.md omits {field.name}"


@pytest.mark.parametrize("path", sorted(SKILLS.rglob("*.md")), ids=lambda p: str(p.relative_to(SKILLS)))
def test_odin_imports_in_examples_resolve(path):
    for code in blocks(read(path), "python"):
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.ImportFrom) and node.module.split(".")[0] in ("odin", "npll", "retrieval"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{path.name}: {node.module}.{alias.name}"


def test_backend_skill_example_runs_and_keeps_relation_labels():
    code = blocks(read(SKILLS / "odin-custom-backend" / "SKILL.md"), "python")[0]
    namespace = {}
    exec(compile(code, "SKILL.md", "exec"), namespace)
    edges = [e for p in namespace["result"]["paths"] for e in p["edges"]]
    assert edges and {e["relation"] for e in edges} == {"linked_to"}
