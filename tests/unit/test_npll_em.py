"""
Unit tests for the NPLL E-M machinery: ELBO computation, E-step, M-step,
and trainer loop termination. Pure unit tests on tiny hand-checkable
graphs; no database required.
"""

import math

import pytest
import torch

from npll.core import Relation, load_knowledge_graph_from_triples
from npll.core.logical_rules import Atom, LogicalRule, RuleType, Variable
from npll.core.mln import create_mln_from_kg_and_rules
from npll.inference.e_step import create_e_step_runner
from npll.inference.elbo import ELBOComputer, VariationalInference
from npll.inference.m_step import MStepRunner
from npll.npll_model import create_initialized_npll_model
from npll.scoring import create_scoring_module
from npll.training.npll_trainer import TrainingConfig, create_trainer
from npll.utils.config import NPLLConfig


LN2 = math.log(2.0)


def small_config(**overrides) -> NPLLConfig:
    """Small dimensions and few iterations so tests stay fast on CPU."""
    params = dict(
        entity_embedding_dim=16,
        relation_embedding_dim=16,
        rule_embedding_dim=32,
        scoring_hidden_dim=32,
        em_iterations=3,
        mean_field_iterations=3,
        device="cpu",
        num_workers=0,
        pin_memory=False,
    )
    params.update(overrides)
    return NPLLConfig(**params)


def chain_rule() -> LogicalRule:
    """r1(x,y) ∧ r2(y,z) → r3(x,z)"""
    r1, r2, r3 = Relation("r1"), Relation("r2"), Relation("r3")
    x, y, z = Variable("?x"), Variable("?y"), Variable("?z")
    return LogicalRule(
        rule_id="chain_rule",
        body=[Atom(r1, (x, y)), Atom(r2, (y, z))],
        head=Atom(r3, (x, z)),
        rule_type=RuleType.TRANSITIVITY,
        confidence=0.7,
    )


@pytest.fixture(autouse=True)
def deterministic_torch():
    torch.manual_seed(42)


@pytest.fixture
def config():
    return small_config()


@pytest.fixture
def tiny_kg():
    """A --r1--> B --r2--> C with unknown fact A --r3--> C (rule head)."""
    kg = load_knowledge_graph_from_triples(
        [("A", "r1", "B"), ("B", "r2", "C")], "TinyKG"
    )
    kg.add_unknown_fact("A", "r3", "C")
    return kg


@pytest.fixture
def tiny_mln(tiny_kg, config):
    return create_mln_from_kg_and_rules(tiny_kg, [chain_rule()], config)


class TestELBOComputer:
    def test_entropy_of_uniform_bernoulli_is_ln2(self, config):
        computer = ELBOComputer(config)
        entropy = computer._compute_entropy_term(torch.tensor([0.5]))
        assert entropy.item() == pytest.approx(LN2, abs=1e-6)

    def test_entropy_sums_over_facts(self, config):
        computer = ELBOComputer(config)
        entropy = computer._compute_entropy_term(torch.tensor([0.5, 0.5, 0.5]))
        assert entropy.item() == pytest.approx(3 * LN2, abs=1e-6)

    def test_entropy_of_near_certain_fact_is_near_zero(self, config):
        computer = ELBOComputer(config)
        for q in (0.0, 1.0):
            entropy = computer._compute_entropy_term(torch.tensor([q]))
            assert torch.isfinite(entropy)
            assert entropy.item() == pytest.approx(0.0, abs=1e-5)

    def test_entropy_is_maximal_at_half(self, config):
        computer = ELBOComputer(config)
        at_half = computer._compute_entropy_term(torch.tensor([0.5])).item()
        for q in (0.1, 0.3, 0.7, 0.9):
            other = computer._compute_entropy_term(torch.tensor([q])).item()
            assert other < at_half

    def test_elbo_without_unknown_facts_equals_known_joint(self, tiny_kg, config):
        mln = create_mln_from_kg_and_rules(tiny_kg, [chain_rule()], config)
        computer = ELBOComputer(config)
        known_facts = list(tiny_kg.known_facts)

        components = computer.compute_elbo(mln, known_facts, [], torch.tensor([]))

        expected = mln.compute_joint_probability({f: True for f in known_facts})
        assert components.elbo.item() == pytest.approx(expected.item(), abs=1e-5)
        assert components.entropy_term.item() == 0.0
        assert components.num_samples == 0

    def test_elbo_decomposes_into_joint_plus_entropy(self, tiny_kg, tiny_mln, config):
        computer = ELBOComputer(config)
        known = list(tiny_kg.known_facts)
        unknown = list(tiny_kg.unknown_facts)
        q_probs = torch.tensor([0.8])

        components = computer.compute_elbo(tiny_mln, known, unknown, q_probs)

        assert components.num_samples > 0
        assert torch.isfinite(components.elbo)
        assert torch.isfinite(components.joint_term)
        assert torch.isfinite(components.entropy_term)
        assert components.elbo.item() == pytest.approx(
            components.joint_term.item() + components.entropy_term.item(), abs=1e-5
        )

    def test_elbo_gradient_matches_rule_count(self, tiny_kg, tiny_mln, config):
        computer = ELBOComputer(config)
        gradients = computer.compute_elbo_gradient(
            tiny_mln,
            list(tiny_kg.known_facts),
            list(tiny_kg.unknown_facts),
            torch.tensor([0.8]),
        )
        assert gradients.shape == (len(tiny_mln.logical_rules),)
        assert torch.all(torch.isfinite(gradients))


class TestVariationalInference:
    def test_no_unknown_facts_converges_immediately(self, tiny_mln, config):
        vi = VariationalInference(config)
        result = vi.optimize_approximate_posterior(tiny_mln, [], [])
        assert result["converged"] is True
        assert result["iterations"] == 0
        assert len(result["optimized_probs"]) == 0

    def test_optimized_probs_are_valid_probabilities(self, tiny_kg, tiny_mln, config):
        vi = VariationalInference(config)
        result = vi.optimize_approximate_posterior(
            tiny_mln, list(tiny_kg.known_facts), list(tiny_kg.unknown_facts)
        )
        probs = result["optimized_probs"]
        assert len(probs) == len(tiny_kg.unknown_facts)
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)

    def test_respects_iteration_budget(self, tiny_kg, tiny_mln, config):
        vi = VariationalInference(config)
        result = vi.optimize_approximate_posterior(
            tiny_mln, list(tiny_kg.known_facts), list(tiny_kg.unknown_facts)
        )
        assert 0 < result["iterations"] <= config.em_iterations
        assert len(result["elbo_history"]) == result["iterations"]
        assert all(math.isfinite(v) for v in result["elbo_history"])


class TestEStep:
    def test_no_unknown_facts_short_circuits(self, config):
        kg = load_knowledge_graph_from_triples([("A", "r1", "B")], "KnownOnly")
        mln = create_mln_from_kg_and_rules(kg, [chain_rule()], config)
        scoring = create_scoring_module(config, kg)
        runner = create_e_step_runner(config)

        result = runner.run_e_step(mln, scoring, kg)

        assert result.convergence_info["converged"] is True
        assert result.convergence_info["reason"] == "no_unknown_facts"
        assert len(result.approximate_posterior_probs) == 0
        assert result.iteration_count == 0

    def test_posterior_probs_are_normalized(self, tiny_kg, tiny_mln, config):
        scoring = create_scoring_module(config, tiny_kg)
        runner = create_e_step_runner(config)

        result = runner.run_e_step(tiny_mln, scoring, tiny_kg)

        assert len(result.approximate_posterior_probs) == len(tiny_kg.unknown_facts)
        assert torch.all(result.approximate_posterior_probs >= 0.0)
        assert torch.all(result.approximate_posterior_probs <= 1.0)
        assert torch.isfinite(result.elbo_value)
        assert torch.isfinite(result.entropy)

    def test_fact_probabilities_cover_all_unknown_facts(self, tiny_kg, tiny_mln, config):
        scoring = create_scoring_module(config, tiny_kg)
        runner = create_e_step_runner(config)

        result = runner.run_e_step(tiny_mln, scoring, tiny_kg)

        assert set(result.fact_probabilities.keys()) == set(tiny_kg.unknown_facts)
        for prob in result.fact_probabilities.values():
            assert 0.0 <= prob <= 1.0

    def test_ground_rule_expectations_match_rule_count(self, tiny_kg, tiny_mln, config):
        scoring = create_scoring_module(config, tiny_kg)
        runner = create_e_step_runner(config)

        result = runner.run_e_step(tiny_mln, scoring, tiny_kg)

        assert len(result.ground_rule_expectations) == len(tiny_mln.logical_rules)
        assert torch.all(result.ground_rule_expectations >= 0.0)


class TestMStep:
    @pytest.fixture
    def e_step_result(self, tiny_kg, tiny_mln, config):
        scoring = create_scoring_module(config, tiny_kg)
        return create_e_step_runner(config).run_e_step(tiny_mln, scoring, tiny_kg)

    def test_updates_all_rule_weights(self, tiny_mln, e_step_result, config):
        runner = MStepRunner(config)
        result = runner.run_m_step(tiny_mln, e_step_result)

        assert len(result.updated_rule_weights) == len(tiny_mln.logical_rules)
        assert torch.all(torch.isfinite(result.updated_rule_weights))
        assert torch.isfinite(result.pseudo_likelihood)

    def test_weight_changes_are_consistent(self, tiny_kg, config):
        mln = create_mln_from_kg_and_rules(tiny_kg, [chain_rule()], config)
        initial = mln.rule_weights.data.clone()
        scoring = create_scoring_module(config, tiny_kg)
        e_step_result = create_e_step_runner(config).run_e_step(mln, scoring, tiny_kg)

        result = MStepRunner(config).run_m_step(mln, e_step_result)

        expected_changes = result.updated_rule_weights - initial
        assert torch.allclose(result.weight_changes, expected_changes, atol=1e-6)

    def test_records_optimization_history(self, tiny_mln, e_step_result, config):
        result = MStepRunner(config).run_m_step(tiny_mln, e_step_result)

        assert len(result.optimization_history) == result.iteration_count
        assert all(math.isfinite(v) for v in result.optimization_history)
        assert result.convergence_info["reason"] in ("converged", "max_iterations")


class TestTrainerLoop:
    @pytest.fixture
    def model(self, tiny_kg, config):
        return create_initialized_npll_model(tiny_kg, [chain_rule()], config)

    def test_train_epoch_respects_em_budget(self, model):
        epoch_result = model.train_epoch(max_em_iterations=2)

        assert 0 < epoch_result["em_iterations"] <= 2
        assert len(epoch_result["iteration_results"]) == epoch_result["em_iterations"]
        for iteration in epoch_result["iteration_results"]:
            assert math.isfinite(iteration["elbo"])
            assert isinstance(iteration["converged"], bool)

    def test_trainer_terminates_and_reports_consistently(self, model):
        training_config = TrainingConfig(
            num_epochs=2,
            max_em_iterations_per_epoch=2,
            early_stopping_patience=2,
            save_checkpoints=False,
        )
        trainer = create_trainer(model, training_config)

        result = trainer.train()

        assert 1 <= result.total_epochs <= training_config.num_epochs
        assert result.total_em_iterations == len(result.elbo_history)
        assert math.isfinite(result.final_elbo)
        assert result.best_elbo >= result.final_elbo or math.isclose(
            result.best_elbo, result.final_elbo
        )
        assert result.best_elbo == pytest.approx(max(result.elbo_history))
        assert result.total_training_time > 0.0
        if result.converged:
            assert result.convergence_epoch is not None
