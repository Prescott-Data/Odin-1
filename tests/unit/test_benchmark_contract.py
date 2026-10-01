from types import SimpleNamespace
from unittest.mock import patch
import pytest
from benchmarks.run_npll_benchmark import create_npll_model, train_npll, evaluate_npll
from npll.core.knowledge_graph import load_knowledge_graph_from_triples
from npll.utils.config import NPLLConfig


def test_benchmark_uses_current_training_and_scoring_contracts():
    config = NPLLConfig(entity_embedding_dim=4, relation_embedding_dim=4,
                        rule_embedding_dim=4, scoring_hidden_dim=4, scorer_epochs=4,
                        max_ground_rules=8, device="cpu")
    model = create_npll_model(load_knowledge_graph_from_triples([("a", "made up", "b")]), config)
    result = train_npll(model, epochs=1)
    assert result["final_elbo"] is not None
    dataset = SimpleNamespace(entities=["a", "b"], test_triples=[("a", "made up", "b")],
                              valid_triples=[], get_train_set=lambda: {("a", "made up", "b")})
    assert evaluate_npll(model, dataset)["triples_evaluated"] == 1
    with patch.object(model, "predict_single_triple", side_effect=RuntimeError("scoring failed")):
        with pytest.raises(RuntimeError, match="scoring failed"):
            evaluate_npll(model, dataset)
