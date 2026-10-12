"""Known-good and known-bad solutions that prove the checks themselves work."""

GOLDEN = {
    "top_path_text": {
        "good": '''
def solve(engine, seeds):
    path = engine.retrieve(seeds=seeds)["paths"][0]
    return "; ".join(f"{e['u']} --{e['relation']}--> {e['v']}" for e in path["edges"])
''',
        "bad": '''
def solve(engine, seeds):
    path = engine.retrieve(seeds=seeds)["paths"][0]
    edges = path["edges"]
    return " -> ".join([edges[0]["u"]] + [e["v"] for e in edges])
''',
        "fails": ["labels_and_direction"],
    },
    "format_for_llm": {
        "good": '''
def solve(engine, seeds):
    lines = []
    for p in engine.retrieve(seeds=seeds)["paths"]:
        lines.append("; ".join(
            f"{e['u']} --{e['relation']}--> {e['v']} (source {e['provenance']})" for e in p["edges"]))
    return "\\n".join(lines)
''',
        "bad": '''
def solve(engine, seeds):
    lines = []
    for p in engine.retrieve(seeds=seeds)["paths"][:1]:
        edges = p["edges"]
        lines.append(" -> ".join([edges[0]["u"]] + [e["v"] for e in edges]))
    return "\\n".join(lines)
''',
        "fails": ["labels_and_direction", "all_relations_present", "provenance_kept"],
    },
    "empty_result": {
        "good": '''
def solve(engine, seeds):
    result = engine.retrieve(seeds=seeds)
    if not result["paths"]:
        return "Odin found no reachable evidence from these seeds."
    return f"{len(result['paths'])} paths found."
''',
        "bad": '''
def solve(engine, seeds):
    result = engine.retrieve(seeds=seeds)
    if not result["paths"]:
        return "Error: retrieval failed."
    return "ok"
''',
        "fails": ["empty_not_error"],
    },
    "backend_failure": {
        "good": '''
def solve(engine, seeds):
    return str(len(engine.retrieve(seeds=seeds)["paths"]))
''',
        "bad": '''
def solve(engine, seeds):
    try:
        return str(len(engine.retrieve(seeds=seeds)["paths"]))
    except Exception:
        return "No results found."
''',
        "fails": ["failure_not_hidden"],
    },
    "configurable_threshold": {
        "good": '''
def solve(engine, seeds, min_triage):
    return engine.retrieve(seeds=seeds)["triage"]["score"] >= min_triage
''',
        "bad": '''
def solve(engine, seeds, min_triage):
    return engine.retrieve(seeds=seeds)["triage"]["score"] >= 70
''',
        "fails": ["no_hardcoded_threshold"],
    },
    "score_one_edge": {
        "good": '''
def solve(engine):
    s = engine.score_edge("claim_1", "billed_by", "provider_9")
    return f"The graph matches this statement with plausibility {s:.2f}, a ranking signal and not proof."
''',
        "bad": '''
def solve(engine):
    s = engine.score_edge("provider_9", "billed_by", "claim_1")
    return f"Score {s:.2f}; this proves the statement."
''',
        "fails": ["score_edge_order", "no_truth_claims"],
    },
    "custom_backend": {
        "good": '''
from odin import OdinEngine

def solve(edges):
    class Accessor:
        def iter_out(self, node):
            return [(v, r, 1.0) for u, r, v in edges if u == node]
        def iter_in(self, node):
            return [(u, r, 1.0) for u, r, v in edges if v == node]
        def iter_out_edges(self, node):
            return [{"_id": f"e{i}", "u": u, "rel": r, "v": v, "weight": 1.0}
                    for i, (u, r, v) in enumerate(edges) if u == node]
        def nodes(self, community_id):
            return sorted({n for u, _, v in edges for n in (u, v)})
        def degree(self, node):
            return sum(1 for u, _, _ in edges if u == node)
        def get_node(self, node_id, fields=None):
            return {"id": node_id}
        def community_seed_norm(self, community_id, seeds):
            return seeds

    class Backend:
        def accessor(self, community_id, community_mode):
            return Accessor()

    return OdinEngine(Backend(), auto_train=False)
''',
        "bad": '''
from odin import OdinEngine

def solve(edges):
    class Accessor:
        def iter_out(self, node):
            return [(v, r, 1.0) for u, r, v in edges if u == node]
        def iter_in(self, node):
            return []
        def nodes(self, community_id):
            return []
        def degree(self, node):
            return 0
        def get_node(self, node_id, fields=None):
            return {}

    class Backend:
        def accessor(self, community_id, community_mode):
            return Accessor()

    return OdinEngine(Backend(), auto_train=False)
''',
        "fails": ["runs", "backend_works"],
    },
    "arango_setup": {
        "good": '''
import os
from arango import ArangoClient
from odin import OdinEngine
from odin.backends.arango import ArangoBackend, ArangoGraphConfig

def solve():
    client = ArangoClient(hosts=os.environ["ARANGO_HOSTS"])
    db = client.db("my_graph", username=os.environ["ARANGO_USER"], password=os.environ["ARANGO_PASSWORD"])
    graph = ArangoGraphConfig(node_collection="People", edge_collection="Knows", relation_field="kind")
    return OdinEngine(ArangoBackend(db, graph))
''',
        "bad": '''
from arango import ArangoClient
from odin import OdinEngine

def solve():
    db = ArangoClient(hosts="http://localhost:8529").db("my_graph", username="root", password="secret")
    return OdinEngine(db)
''',
        "fails": ["explicit_backend", "no_literal_password"],
    },
    "migrate_old_code": {
        "good": '''
from odin import OdinEngine
from odin.backends.arango import ArangoBackend, ArangoGraphConfig

def solve(db):
    graph = ArangoGraphConfig(node_collection="Entities", edge_collection="Relations", relation_field="label")
    engine = OdinEngine(ArangoBackend(db, graph))
    return engine.retrieve(seeds=["Entities/a"])
''',
        "bad": '''
from odin import OdinEngine

def solve(db):
    engine = OdinEngine(db)
    return engine.retrieve(seeds=["Entities/a"])
''',
        "fails": ["explicit_backend"],
    },
    "path_node_sequence": {
        "good": '''
def solve(engine, seeds):
    edges = engine.retrieve(seeds=seeds)["paths"][0]["edges"]
    return [edges[0]["u"]] + [e["v"] for e in edges]
''',
        "bad": '''
def solve(engine, seeds):
    return engine.retrieve(seeds=seeds)["paths"][0]["nodes"]
''',
        "fails": ["no_nodes_key"],
    },
    "visible_truncation": {
        "good": '''
def solve(engine, seeds, limit):
    paths = engine.retrieve(seeds=seeds)["paths"]
    shown = paths[:limit]
    lines = ["; ".join(f"{e['u']} --{e['relation']}--> {e['v']}" for e in p["edges"]) for p in shown]
    return f"Showing {len(shown)} of {len(paths)} paths\\n" + "\\n".join(lines)
''',
        "bad": '''
def solve(engine, seeds, limit):
    paths = engine.retrieve(seeds=seeds)["paths"][:limit]
    return "\\n".join("; ".join(f"{e['u']} --{e['relation']}--> {e['v']}" for e in p["edges"]) for p in paths)
''',
        "fails": ["truncation_visible"],
    },
    "training_failure": {
        "good": '''
from npll import TrainingError
from odin import OdinEngine

def solve(backend):
    try:
        return OdinEngine(backend)
    except TrainingError as exc:
        return f"Training failed: {exc}"
''',
        "bad": '''
from odin import OdinEngine

def solve(backend):
    try:
        return OdinEngine(backend)
    except Exception:
        return OdinEngine(backend, auto_train=False)
''',
        "fails": ["handles_training_error"],
    },
    "unrelated_networkx": {
        "good": '''
import networkx as nx

def solve(edges):
    g = nx.DiGraph([(u, v) for u, _, v in edges])
    return nx.pagerank(g, personalization={edges[0][0]: 1.0})
''',
        "bad": '''
from odin import OdinEngine

def solve(edges):
    return OdinEngine
''',
        "fails": ["no_odin_import"],
    },
    "unrelated_aql": {
        "good": '''
def solve(db):
    return list(db.aql.execute("FOR p IN People FILTER p.age > 30 RETURN p"))
''',
        "bad": '''
from odin import inspect_schema

def solve(db):
    return inspect_schema(db)
''',
        "fails": ["no_odin_import"],
    },
    "unrelated_pandas": {
        "good": '''
import pandas as pd

def solve(rows):
    return pd.DataFrame(rows).groupby("team")["score"].mean().to_dict()
''',
        "bad": '''
import odin

def solve(rows):
    return odin
''',
        "fails": ["no_odin_import"],
    },
}
