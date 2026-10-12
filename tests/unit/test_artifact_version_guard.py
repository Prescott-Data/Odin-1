"""Stored-model semantics are pinned to ARTIFACT_VERSION.

Older readers treat config, scorer-recipe, and rule changes within the same
version as stale and replace the artifact. A semantic change must therefore bump
ARTIFACT_VERSION so older workers raise NewerModelVersionError instead.
"""
import hashlib
import json
from dataclasses import asdict, fields

from npll.bootstrap import RUNTIME_CONFIG_FIELDS, SCORER_RECIPE, KnowledgeBootstrapper
from npll.core.knowledge_graph import load_knowledge_graph_from_triples
from npll.utils.config import NPLLConfig, get_config
from odin.backends.base import ARTIFACT_VERSION

# Update these together, and only after bumping ARTIFACT_VERSION.
PINNED_ARTIFACT_VERSION = "7.0"
PINNED_SEMANTICS_FINGERPRINT = "fc140e65bed424018454429096ee1fd6c4ec63609d9d9d6b5c3f84d7e2f58994"

FIXTURE_TRIPLES = [
    ("People/a1", "works_for", "Orgs/c1"), ("People/a2", "works_for", "Orgs/c1"),
    ("People/a3", "works_for", "Orgs/c2"),
    ("Orgs/c1", "located_in", "Places/l1"), ("Orgs/c2", "located_in", "Places/l1"),
    ("People/a1", "based_in", "Places/l1"), ("People/a2", "based_in", "Places/l1"),
    ("People/a1", "knows", "People/a2"), ("People/a2", "knows", "People/a1"),
    ("People/a2", "knows", "People/a3"), ("People/a3", "knows", "People/a2"),
    ("People/a1", "has_type", "Person"), ("Orgs/c1", "has_type", "Organization"),
]


def model_semantics():
    config = asdict(get_config("OdinTriples"))
    kg = load_knowledge_graph_from_triples(FIXTURE_TRIPLES, "version-guard")
    rules = KnowledgeBootstrapper(None, None)._generate_smart_rules(kg)
    return {
        "config_fields": sorted(f.name for f in fields(NPLLConfig)),
        "config_values": {k: v for k, v in sorted(config.items())
                          if k not in RUNTIME_CONFIG_FIELDS},
        "scorer_recipe": SCORER_RECIPE,
        "rules": [[r.rule_id, str(r), r.confidence] for r in rules],
    }


def fingerprint(semantics):
    payload = json.dumps(semantics, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_fixture_exercises_every_rule_family():
    rule_types = {rule[0].split("_", 1)[0] for rule in model_semantics()["rules"]}
    assert rule_types == {"prior", "chain", "symmetry"}


def test_model_semantics_changes_require_an_artifact_version_bump():
    actual = fingerprint(model_semantics())
    assert (ARTIFACT_VERSION, actual) == (PINNED_ARTIFACT_VERSION, PINNED_SEMANTICS_FINGERPRINT), (
        "Stored-model semantics or ARTIFACT_VERSION changed. If config fields or values, "
        "SCORER_RECIPE, or rule generation changed, bump ARTIFACT_VERSION so older workers "
        "refuse the new artifact, then pin the new version and fingerprint "
        f"{actual} here."
    )
