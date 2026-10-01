import random
import torch
from odin.backends.base import TrainingSnapshot
from npll.bootstrap import create_snapshot_trained_model, KnowledgeBootstrapper
from npll.core.knowledge_graph import load_knowledge_graph_from_triples
from npll.utils.config import get_config


def test_scorer_rejects_independent_random_corruptions_across_vocabulary():
    entities = [f'Nodes/{i:04}' for i in range(80)]
    triples = [(h, f'predicate {r}', entities[(i + r + 1) % 80])
               for i, h in enumerate(entities) for r in range(3)]
    snapshot = TrainingSnapshot(tuple(triples))
    kg = load_knowledge_graph_from_triples(snapshot.triples, 'test')
    rules = KnowledgeBootstrapper(None, None)._generate_smart_rules(kg)
    config = get_config('OdinTriples')
    model = create_snapshot_trained_model(snapshot, kg, rules, config)
    rng = random.Random(9321)
    facts = set(triples)
    negatives = []
    while len(negatives) < 500:
        candidate = (rng.choice(entities), f'predicate {rng.randrange(3)}', rng.choice(entities))
        if candidate not in facts:
            negatives.append(candidate)
    def mean_score(rows):
        with torch.no_grad():
            return model.scoring_module.forward_with_names(
                [h for h, _, _ in rows], [r for _, r, _ in rows], [t for _, _, t in rows]
            ).sigmoid().mean().item()
    positive, negative = mean_score(triples), mean_score(negatives)
    assert positive > 0.8, (positive, negative)
    assert negative < 0.3, (positive, negative)
    assert positive - negative > 0.5


def test_scorer_does_not_supervise_em_holdout(monkeypatch):
    snapshot = TrainingSnapshot((('N/a', 'r', 'N/b'), ('N/b', 'r', 'N/c')))
    kg = load_knowledge_graph_from_triples(snapshot.triples, 'test')
    held = next(f for f in kg.known_facts if f.head.name == 'N/b')
    kg.hold_out_fact(held)
    config = get_config('OdinTriples')
    config.scorer_epochs = 1
    model = create_snapshot_trained_model(snapshot, kg, [], config)
    # One supervised positive, two distinct unobserved candidates on each side.
    assert model.scorer_training['example_count'] == 5
