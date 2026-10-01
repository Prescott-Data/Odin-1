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
    chains = RuleGenerator(kg).generate_simple_rules(min_support=1)
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


def test_groundings_include_observed_bodies_and_unbiased_prior_heads():
    triples = []
    for i in range(120):
        triples.extend([(f'Z/{i}', 'r1', f'M/{i}'),
                        (f'M/{i}', 'r2', f'T/{i}'),
                        (f'Z/{i}', 'r3', f'T/{i}')])
    kg = load_knowledge_graph_from_triples(triples)
    generator = RuleGenerator(kg)
    chain = generator.generate_simple_rules()[0]
    groundings = chain.generate_ground_rules(kg, max_groundings=300)
    true_bodies = [g for g in groundings if all(f in kg.known_facts for f in g.body_facts)]
    assert len(true_bodies) == 120
    assert any(g.head_fact.head.name == 'Z/119' for g in true_bodies)
    assert len(groundings) == 300
    prior = next(r for r in generator.generate_relation_priors() if r.head.predicate.name == 'r1')
    grounded = prior.generate_ground_rules(kg, max_groundings=300)
    assert sum(g.head_fact in kg.known_facts for g in grounded) == 120
    assert len({g.head_fact.head.name for g in grounded}) > 120
    assert [str(g) for g in groundings] == [str(g) for g in chain.generate_ground_rules(kg, 300)]


def test_hub_mining_is_bounded_and_reports_sample_population():
    triples = [(f'A/{i}', 'in', 'hub') for i in range(200)]
    triples += [('hub', 'out', f'B/{i}') for i in range(200)]
    generator = RuleGenerator(load_knowledge_graph_from_triples(triples))
    assert generator.generate_simple_rules(max_paths_per_node=75) == []
    assert generator.mining_report['possible_paths'] == 40000
    assert generator.mining_report['sampled_paths'] == 75


def test_low_confidence_coincidences_are_not_mined_as_rules():
    triples = [('a', 'r1', 'hub'), ('a', 'r3', 'b/0')]
    triples += [('hub', 'r2', f'b/{i}') for i in range(30)]
    generator = RuleGenerator(load_knowledge_graph_from_triples(triples))
    assert generator.generate_simple_rules(min_support=1, min_confidence=0.2) == []
