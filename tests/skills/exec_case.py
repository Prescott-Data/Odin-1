"""Run one generated solution in a subprocess and print a JSON result line.

Usage: python -m tests.skills.exec_case CODE_FILE '<json list of arg tokens>'

Generated code is executed. Run evals only in a disposable environment.
"""
import contextlib
import io
import json
import sys

from tests.skills import fixtures

MARK = "@@RESULT@@"


def main(code_file, tokens_json):
    source = open(code_file, encoding="utf-8").read()
    out = {"returned": None, "stdout": "", "exception": None, "message": "",
           "score_edge_calls": [], "path_counts": [], "post_retrieve": None}
    args = [fixtures.resolve(t) for t in json.loads(tokens_json)]
    buffer = io.StringIO()
    try:
        namespace = {}
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, "solution.py", "exec"), namespace)
            returned = namespace["solve"](*args)
        out["returned"] = returned if isinstance(returned, str) else repr(returned)
        if hasattr(returned, "retrieve"):
            result = returned.retrieve(seeds=[fixtures.EDGES[0][0]])
            out["post_retrieve"] = [e["relation"] for p in result["paths"] for e in p["edges"]]
    except BaseException as exc:  # report, never crash the harness
        out["exception"] = type(exc).__name__
        out["message"] = str(exc)
    out["stdout"] = buffer.getvalue()
    for arg in args:
        out["score_edge_calls"] += [list(c) for c in getattr(arg, "score_edge_calls", [])]
        out["path_counts"] += list(getattr(arg, "retrieve_path_counts", []))
    print(MARK + json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
