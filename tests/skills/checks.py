"""Deterministic checks over a generated solution and its execution result."""
from __future__ import annotations

import ast
import re
from typing import Callable, Dict, Tuple

from tests.skills.fixtures import EDGES

Check = Callable[[dict], Tuple[bool, str]]
NUMBER = (int, float)


def _text(ctx):
    return (ctx["result"].get("returned") or "") + "\n" + (ctx["result"].get("stdout") or "")


def _ok(passed, message=""):
    return bool(passed), message


def runs(ctx):
    exc = ctx["result"]["exception"]
    return _ok(exc is None, f"raised {exc}: {ctx['result']['message']}" if exc else "")


def labels_and_direction(ctx):
    text = _text(ctx)
    present = [(u, r, v) for u, r, v in EDGES if r in text]
    if not present:
        return False, "no relation label from the retrieved edges appears in the output"
    for u, r, v in present:
        if u not in text or v not in text:
            return False, f"edge {u} {r} {v}: an endpoint is missing"
        if text.index(u) > text.index(v):
            return False, f"edge {u} {r} {v}: direction reversed"
    return True, ""


def all_relations_present(ctx):
    text = _text(ctx)
    missing = [r for _, r, _ in EDGES if r not in text]
    return _ok(not missing, f"relations missing from output: {missing}")


def provenance_kept(ctx):
    return _ok("doc_" in _text(ctx), "provenance (document ids) not present in the output")


def empty_not_error(ctx):
    text = _text(ctx).lower()
    if ctx["result"]["exception"]:
        return False, "raised on an empty result"
    if not text.strip():
        return False, "no message returned"
    bad = [w for w in ("error", "exception", "failed", "failure") if w in text]
    return _ok(not bad, f"empty result described as a failure: {bad}")


def failure_not_hidden(ctx):
    exc = ctx["result"]["exception"]
    if exc == "BackendIOError":
        return True, ""
    if exc:
        return False, f"unexpected exception {exc}"
    text = _text(ctx).lower()
    hides = re.search(r"\bno (evidence|paths?|results?|findings)\b", text)
    reports = "backendioerror" in text or "unreachable" in text or "error" in text or "fail" in text
    return _ok(reports and not hides, "a backend failure was reported as empty or not reported")


def no_hardcoded_threshold(ctx):
    for node in ast.walk(ctx["tree"]):
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            segment = ast.get_source_segment(ctx["code"], node) or ""
            has_number = any(isinstance(s, ast.Constant) and isinstance(s.value, NUMBER)
                             and not isinstance(s.value, bool) for s in sides)
            if has_number and re.search(r"triage|score", segment):
                return False, f"hard-coded cutoff: {segment}"
    return True, ""


def score_edge_order(ctx):
    calls = ctx["result"]["score_edge_calls"]
    return _ok(["claim_1", "billed_by", "provider_9"] in calls,
               f"score_edge was not called as (src, rel, dst); calls: {calls}")


def no_truth_claims(ctx):
    hit = re.search(r"\b(proves?|proven|confirmed|certainly)\b", _text(ctx).lower())
    return _ok(not hit, f"presents a score as proof: {hit.group(0) if hit else ''}")


def backend_works(ctx):
    seen = ctx["result"]["post_retrieve"]
    return _ok(bool(seen) and "billed_by" in seen,
               f"engine returned by solve did not retrieve the stored relations: {seen}")


def no_nodes_key(ctx):
    for node in ast.walk(ctx["tree"]):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and node.slice.value == "nodes":
            return False, 'uses path["nodes"], which is not a documented field'
    return True, ""


def nodes_follow_edges(ctx):
    try:
        nodes = ast.literal_eval(ctx["result"]["returned"] or "")
    except (ValueError, SyntaxError):
        return False, "did not return a list of node ids"
    if not isinstance(nodes, list) or len(nodes) < 2 or nodes[0] != "claim_1":
        return False, f"unexpected node sequence: {nodes}"
    pairs = {(u, v) for u, _, v in EDGES}
    return _ok(all((a, b) in pairs for a, b in zip(nodes, nodes[1:])),
               f"consecutive nodes are not connected by a stored edge: {nodes}")


def truncation_visible(ctx):
    counts = ctx["result"]["path_counts"]
    total = max(counts) if counts else None
    limit = ctx["entry"]["args"][-1]
    text = _text(ctx)
    if total is None:
        return False, "engine.retrieve was never called"
    if total <= limit:
        return True, ""
    pattern = rf"\b{limit}\b\D{{0,20}}\bof\b\D{{0,5}}\b{total}\b"
    return _ok(re.search(pattern, text, re.I) is not None,
               f"output does not say it shows {limit} of {total} paths")


def _calls(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if (isinstance(fn, ast.Name) and fn.id == name) or \
                    (isinstance(fn, ast.Attribute) and fn.attr == name):
                yield node


def explicit_backend(ctx):
    tree, expect = ctx["tree"], ctx["entry"]["expect_kwargs"]
    for call in _calls(tree, "OdinEngine"):
        first = call.args[0] if call.args else None
        if isinstance(first, ast.Name) and first.id in ("db", "database", "client"):
            return False, "OdinEngine is given a raw database handle"
    configs = list(_calls(tree, "ArangoGraphConfig"))
    if not configs or not list(_calls(tree, "ArangoBackend")) or not list(_calls(tree, "OdinEngine")):
        return False, "expected ArangoGraphConfig, ArangoBackend and OdinEngine"
    kwargs = {k.arg: k.value.value for k in configs[0].keywords
              if isinstance(k.value, ast.Constant)}
    wrong = {k: kwargs.get(k) for k, v in expect.items() if kwargs.get(k) != v}
    return _ok(not wrong, f"ArangoGraphConfig mapping is wrong or missing: {wrong}")


def no_literal_password(ctx):
    for node in ast.walk(ctx["tree"]):
        if isinstance(node, ast.keyword) and node.arg == "password" \
                and isinstance(node.value, ast.Constant) and node.value.value:
            return False, "password is a string literal"
    return True, ""


def handles_training_error(ctx):
    caught = set()
    for handler in ast.walk(ctx["tree"]):
        if isinstance(handler, ast.ExceptHandler) and handler.type is not None:
            caught |= {n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)}
    if not caught & {"TrainingConvergenceError", "TrainingError"}:
        return False, "does not catch TrainingError or TrainingConvergenceError"
    if "auto_train=False" in ctx["code"].replace(" ", ""):
        return False, "falls back to auto_train=False to hide the failure"
    return True, ""


def no_odin_import(ctx):
    for node in ast.walk(ctx["tree"]):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        if any(n.split(".")[0] in ("odin", "npll", "retrieval") for n in names):
            return False, "imports Odin for a task that is not about Odin"
    return True, ""


CHECKS: Dict[str, Check] = {
    f.__name__: f for f in (
        runs, labels_and_direction, all_relations_present, provenance_kept, empty_not_error,
        failure_not_hidden, no_hardcoded_threshold, score_edge_order, no_truth_claims,
        backend_works, no_nodes_key, nodes_follow_edges, truncation_visible, explicit_backend,
        no_literal_password, handles_training_error, no_odin_import,
    )
}
