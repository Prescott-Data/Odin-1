"""Rules must reflect observed evidence for arbitrary relation vocabularies."""
from npll.bootstrap import KnowledgeBootstrapper
from npll.core.knowledge_graph import load_knowledge_graph_from_triples
from npll.core.logical_rules import RuleGenerator, RuleType
from tests.utils.backend_fakes import MemorySource, MemoryStore


def test_arbitrary_vocabulary_has_non_tautological_priors_and_supported_chains():
    triples = [("Records/a", "submitted_by", "Records/b"),
               ("Records/b", "works for", "Records/c"),
               ("Records/a", "handled_by", "Records/c")]
    kg = load_knowledge_graph_from_triples(triples)
    rules = KnowledgeBootstrapper(MemorySource(), MemoryStore())._generate_smart_rules(kg)
    assert {r.head.predicate.name for r in rules if r.rule_type == RuleType.PRIOR} == \
        {"submitted_by", "works for", "handled_by"}
    chains = [r for r in rules if r.rule_type == RuleType.TRANSITIVITY]
    assert len(chains) == 1
    assert [a.predicate.name for a in chains[0].body] == ["submitted_by", "works for"]
    assert chains[0].head.predicate.name == "handled_by"
    assert chains[0].support == 1
    assert all(r.body != [r.head] for r in rules)


def test_false_premises_do_not_count_as_rule_support():
    kg = load_knowledge_graph_from_triples([("a", "first", "b"), ("c", "second", "d")])
    generator = RuleGenerator(kg)
    assert generator.generate_simple_rules(min_support=1) == []
    assert generator.generate_symmetry_rules(min_support=1) == []


def test_same_relation_transitivity_and_tail_support_are_mined_deterministically():
    triples = []
    for i in range(150):
        triples.extend([(f"a/{i}", "linked Via", f"b/{i}"),
                        (f"b/{i}", "linked Via", f"c/{i}"),
                        (f"a/{i}", "linked Via", f"c/{i}")])
    first = RuleGenerator(load_knowledge_graph_from_triples(triples)).generate_simple_rules()
    second = RuleGenerator(load_knowledge_graph_from_triples(reversed(triples))).generate_simple_rules()
    assert len(first) == 1
    assert first[0].support == 150
    assert first[0].confidence == 1.0
    assert [(r.rule_id, str(r), r.support) for r in first] == \
        [(r.rule_id, str(r), r.support) for r in second]
