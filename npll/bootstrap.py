"""
Bootstrap module for NPLL.
Handles the end-to-end lifecycle of the NPLL model:
1. Reading a backend-neutral training snapshot
2. Generating domain-appropriate logical rules
3. Training the model
4. Persisting learned scorer state, rule weights, schema, and complete reports

Architecture:
- The injected ModelStore owns persistence
- Model architecture rebuilt from its vocabulary, then learned tensors loaded
- No external files needed
"""

import base64
import hashlib
import io
import logging
import random
import torch
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional, Any
from odin.backends.base import (
    ARTIFACT_VERSION, MODEL_KEY, CorruptModelError, ModelConflictError, ModelStore, StoredModel,
    TrainingSnapshot, TripleSource, validate_model_artifact,
)

from .core.knowledge_graph import KnowledgeGraph, load_knowledge_graph_from_triples
from .core.logical_rules import LogicalRule, RuleGenerator
from .npll_model import create_initialized_npll_model, NPLLModel
from .training.npll_trainer import TrainingConfig, TrainingResult, create_trainer
from .utils.config import get_config
from .utils.config import NPLLConfig

logger = logging.getLogger(__name__)


def scoring_initialization_seed(snapshot: TrainingSnapshot) -> int:
    """Derive a stable scorer initialization from the exact graph snapshot."""
    return int(snapshot.data_hash[:16], 16) % (2 ** 63 - 1)


def create_snapshot_initialized_model(snapshot: TrainingSnapshot, kg: KnowledgeGraph,
                                      rules: List[LogicalRule], config: NPLLConfig) -> NPLLModel:
    """Create the untrained scorer deterministically without mutating caller RNG state."""
    cuda_devices = list(range(torch.cuda.device_count())) if config.device.startswith("cuda") else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(scoring_initialization_seed(snapshot))
        if cuda_devices:
            torch.cuda.manual_seed_all(scoring_initialization_seed(snapshot))
        return create_initialized_npll_model(kg, rules, config)


SCORER_RECIPE = "uniform-endpoint-corruptions-v2"


def create_snapshot_trained_model(snapshot, kg, rules, config):
    """Learn a scorer from supervised known facts and uniform endpoint corruptions.

    Negative examples are unobserved corruptions, not assertions of source falsity.
    Latent E-M holdout facts are excluded from supervision and negative examples.
    """
    if config.scorer_epochs <= 0 or config.scorer_learning_rate <= 0:
        raise TrainingError("Scorer training requires positive epochs and learning rate")
    devices = list(range(torch.cuda.device_count())) if config.device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(scoring_initialization_seed(snapshot))
        model = create_snapshot_initialized_model(snapshot, kg, rules, config)
        facts = set(snapshot.triples)
        entities = sorted({v for h, _, t in facts for v in (h, t)})
        rng = random.Random(scoring_initialization_seed(snapshot))
        positives = sorted((f.head.name, f.relation.name, f.tail.name) for f in kg.known_facts)
        examples = []
        labels = []
        for h, r, t in positives:
            examples.append((h, r, t))
            labels.append(1.0)
            for axis in (0, 2):
                selected = set()
                for _ in range(config.scorer_negatives_per_side * 20):
                    candidate = (rng.choice(entities), r, t) if axis == 0 else (h, r, rng.choice(entities))
                    if candidate not in facts:
                        selected.add(candidate)
                    if len(selected) == config.scorer_negatives_per_side:
                        break
                if len(selected) < config.scorer_negatives_per_side:
                    shuffled = list(entities)
                    rng.shuffle(shuffled)
                    for other in shuffled:
                        candidate = (other, r, t) if axis == 0 else (h, r, other)
                        if candidate not in facts:
                            selected.add(candidate)
                        if len(selected) == config.scorer_negatives_per_side:
                            break
                for candidate in sorted(selected):
                    examples.append(candidate)
                    labels.append(0.0)
        if not any(label == 0.0 for label in labels):
            raise TrainingError("Snapshot has no unobserved corruptions for scorer training")
        scorer = model.scoring_module
        scorer.train()
        optimizer = torch.optim.Adam(scorer.parameters(), lr=config.scorer_learning_rate)
        device = next(scorer.parameters()).device
        losses = []
        for _ in range(config.scorer_epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            total = 0.0
            for start in range(0, len(examples), config.batch_size):
                indices = order[start:start + config.batch_size]
                batch = [examples[i] for i in indices]
                targets = torch.tensor([labels[i] for i in indices], device=device)
                optimizer.zero_grad()
                logits = scorer.forward_with_names([h for h, _, _ in batch],
                                                   [r for _, r, _ in batch],
                                                   [t for _, _, t in batch])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=torch.tensor(labels.count(0.0) / labels.count(1.0), device=device))
                if not torch.isfinite(loss):
                    raise TrainingError("Scorer training produced a non-finite loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            losses.append(total / len(examples))
        scorer.eval()
        scorer.requires_grad_(False)
        model.scorer_training = {"recipe": SCORER_RECIPE, "torch_version": str(torch.__version__),
                                 "loss_history": losses, "example_count": len(examples)}
        return model


class TrainingError(RuntimeError):
    """Requested NPLL training failed; callers must not serve substitute scores."""


@dataclass
class TrainingReport:
    """
    Audit record for an NPLL training run.

    Persisted alongside the model weights so convergence provenance survives
    cached-weight loads. The complete per-iteration histories are kept —
    they are small at bootstrap scale and required for auditability.
    """
    converged: bool
    convergence_epoch: Optional[int]
    final_elbo: float
    best_elbo: float
    total_epochs: int
    total_em_iterations: int
    elbo_history: List[float]
    rule_weight_delta_history: List[Optional[float]]
    training_time_seconds: float
    trained_at: str
    early_stopping_triggered: bool
    convergence_criteria: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_training_result(cls, result: TrainingResult, trained_at: str) -> "TrainingReport":
        config = get_config("OdinTriples")
        return cls(
            converged=result.converged,
            convergence_epoch=result.convergence_epoch,
            final_elbo=float(result.final_elbo),
            best_elbo=float(result.best_elbo),
            total_epochs=result.total_epochs,
            total_em_iterations=result.total_em_iterations,
            elbo_history=[float(v) for v in result.elbo_history],
            rule_weight_delta_history=list(result.rule_weight_delta_history),
            training_time_seconds=float(result.total_training_time),
            trained_at=trained_at,
            early_stopping_triggered=result.early_stopping_triggered,
            convergence_criteria={
                "elbo_rel_tol": config.elbo_rel_tol,
                "weight_abs_tol": config.weight_abs_tol,
                "convergence_patience": config.convergence_patience,
            },
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingReport":
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BootstrapResult:
    """Outcome of KnowledgeBootstrapper.ensure_model_ready."""
    model: Optional[NPLLModel]
    source: str  # "trained" | "cached_weights" | "failed"
    data_hash: str
    report: Optional[TrainingReport]


class KnowledgeBootstrapper:
    """
    Manages the lifecycle of the NPLL model.
    
    Storage Strategy:
    - Learned scorer tensors, rule weights and audit metadata are saved through ModelStore
    - Model parameters are loaded without training on worker startup
    - No external .pt files needed
    """
    
    def __init__(self, triple_source: TripleSource, model_store: ModelStore):
        self.triple_source = triple_source
        self.model_store = model_store

    def ensure_model_ready(self, force_retrain: bool = False) -> BootstrapResult:
        """
        Ensure a trained NPLL model is available.
        
        Flow:
        1. Extract a snapshot with its content fingerprint
        2. Read the model artifact and its revision from ModelStore
        3. If compatible: initialize architecture, load learned tensors and rule weights
        4. If not found: train new model, save weights to DB
        
        Args:
            force_retrain: If True, retrains using the current store revision.
                Corrupt artifacts and store failures still raise.
            
        Returns:
            BootstrapResult with the model (or None for an empty snapshot),
            its source, and the complete rule TrainingReport. Errors raise.
            Backend errors propagate without converting them into cache misses.
        """
        snapshot = self.triple_source.snapshot()
        current_hash = snapshot.data_hash
        logger.info("Current data hash: %s", current_hash)
        # Read the revision before training, including forced retraining.
        stored = self.model_store.load(MODEL_KEY)
        if stored is not None:
            validate_model_artifact(stored.document)
        
        if not force_retrain:
            # Try to load existing weights and rebuild model
            model, report = self._load_model_with_weights(snapshot, stored)
            if model:
                return BootstrapResult(
                    model=model, source="cached_weights",
                    data_hash=current_hash, report=report,
                )
        
        # Train new model
        logger.info("Training new NPLL model...")
        try:
            model, report = self._train_and_save_weights(
                snapshot, stored.revision if stored is not None else None,
            )
        except ModelConflictError:
            if force_retrain:
                raise
            winner = self.model_store.load(MODEL_KEY)
            if winner is not None:
                validate_model_artifact(winner.document)
            model, report = self._load_model_with_weights(snapshot, winner)
            if model is None:
                raise ModelConflictError("Concurrent model does not match the training contract")
            return BootstrapResult(model=model, source="cached_weights",
                                   data_hash=current_hash, report=report)
        source = "trained" if model else "failed"
        return BootstrapResult(model=model, source=source, data_hash=current_hash, report=report)

    def _load_model_with_weights(
        self, snapshot: TrainingSnapshot, stored: Optional[StoredModel],
    ) -> Tuple[Optional[NPLLModel], Optional[TrainingReport]]:
        if stored is None or stored.document.get("version") != ARTIFACT_VERSION:
            return None, None
        if stored.document.get("data_hash") != snapshot.data_hash:
            return None, None
        doc = stored.document
        if not snapshot.triples:
            return None, None
        kg = load_knowledge_graph_from_triples(snapshot.triples, "Odin_KG")
        rules = self._generate_smart_rules(kg)
        expected_rules = [(r.rule_id, str(r), r.confidence) for r in rules]
        saved_rules = [(r["rule_id"], r["rule_text"], r["confidence"]) for r in doc["rules"]]
        if expected_rules != saved_rules:
            # Rules are code-generated: a mismatch means the generation code
            # changed since training — staleness, not corruption. Retrain.
            logger.info("Saved rules do not match current rule generation; retraining")
            return None, None
        try:
            report = TrainingReport.from_dict(doc["training_report"])
        except (TypeError, ValueError) as exc:
            raise CorruptModelError("Invalid training report") from exc
        inference_state = doc["inference_state"]
        if inference_state["initialization_seed"] != scoring_initialization_seed(snapshot):
            raise CorruptModelError("Model initialization seed does not match the graph snapshot")
        config_data = dict(inference_state["config"])
        current = asdict(get_config("OdinTriples"))
        # Device and loader/runtime settings do not change learned semantics.
        runtime_fields = {"device", "num_workers", "pin_memory", "eval_batch_size",
                          "log_interval", "save_interval", "checkpoint_dir"}
        if {k: v for k, v in config_data.items() if k not in runtime_fields} != {
                k: v for k, v in current.items() if k not in runtime_fields}:
            return None, None
        config = NPLLConfig(**{**config_data, "device": current["device"]})
        recipe = inference_state["scorer_training"]
        if recipe["recipe"] != SCORER_RECIPE:
            return None, None
        model = create_snapshot_initialized_model(snapshot, kg, rules, config)
        blob = inference_state["scorer_state"]
        try:
            payload = base64.b64decode(blob["data"], validate=True)
            if hashlib.sha256(payload).hexdigest() != blob["sha256"]:
                raise ValueError("Scorer state checksum differs")
            tensors = torch.load(io.BytesIO(payload), map_location=config.device, weights_only=True)
            expected = model.scoring_module.state_dict()
            if set(tensors) != set(expected) or any(
                    not isinstance(tensors[k], torch.Tensor) or tensors[k].shape != expected[k].shape or
                    not torch.isfinite(tensors[k]).all() for k in expected):
                raise ValueError("Invalid scorer tensors")
            model.scoring_module.load_state_dict(tensors, strict=True)
        except Exception as exc:
            raise CorruptModelError("Invalid persisted scorer state") from exc
        model.scorer_training = recipe
        model.scoring_module.eval()
        model.scoring_module.requires_grad_(False)
        with torch.no_grad():
            model.mln.rule_weights.copy_(torch.tensor(doc["rule_weights"], dtype=torch.float32))
        logger.info("Model rebuilt with saved weights (trained: %s)", doc["trained_at"])
        return model, report

    def _train_and_save_weights(
        self, snapshot: TrainingSnapshot, expected_revision: Optional[str],
    ) -> Tuple[Optional[NPLLModel], Optional[TrainingReport]]:
        """
        Train a new NPLL model and save its complete learned state to the store.
        """
        # 1. Extract Triples
        triples = snapshot.triples
        if not triples:
            logger.error("No triples extracted. Cannot train.")
            return None, None
        
        # 2. Build KG
        kg = load_knowledge_graph_from_triples(triples, "Odin_KG")
        logger.info(f"Built KG: {len(kg.entities)} entities, {len(kg.relations)} relations, {len(kg.known_facts)} facts")
        
        # Create unknown facts for training (10%)
        known_facts_list = sorted(kg.known_facts, key=lambda fact: (
            fact.head.name, fact.relation.name, fact.tail.name,
        ))
        sampling_rng = random.Random(42)
        num_unknown = min(len(known_facts_list) - 1, max(1, len(known_facts_list) // 10))
        unknown_facts = sampling_rng.sample(known_facts_list, num_unknown)
        
        for fact in unknown_facts:
            kg.hold_out_fact(fact)
        
        # 3. Generate Rules
        rules = self._generate_smart_rules(kg)
        logger.info(f"Generated {len(rules)} logical rules")
        
        if not rules:
            logger.error("No rules generated. Cannot train.")
            return None, None
        
        # 4. Initialize Model
        config = get_config("OdinTriples")
        model = create_snapshot_trained_model(snapshot, kg, rules, config)
        
        # 5. Train
        train_config = TrainingConfig(
            num_epochs=10,
            max_em_iterations_per_epoch=5,
            early_stopping_patience=3,
            save_checkpoints=False
        )
        trainer = create_trainer(model, train_config)
        
        training_result = None
        try:
            logger.info("Starting NPLL training...")
            training_result = trainer.train()
            logger.info(f"Training completed. Final ELBO: {training_result.final_elbo}")
        except Exception as e:
            raise TrainingError("NPLL trainer failed") from e
        
        trained_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        report = TrainingReport.from_training_result(training_result, trained_at)
        if not report.converged:
            logger.warning(
                "NPLL training finished WITHOUT convergence "
                f"(epochs={report.total_epochs}, em_iterations={report.total_em_iterations}, "
                f"final_elbo={report.final_elbo:.6f}). Edge confidences may be poorly calibrated."
            )
        
        # 6. Save complete learned state and training evidence.
        self._save_weights(model, kg, rules, snapshot, report, expected_revision)
        
        return model, report

    def _save_weights(self, model: NPLLModel, kg: KnowledgeGraph,
                      rules: List[LogicalRule], snapshot: TrainingSnapshot,
                      report: TrainingReport, expected_revision: Optional[str]):
        buffer = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in model.scoring_module.state_dict().items()}, buffer)
        payload = buffer.getvalue()
        doc = {
            "model_type": "npll",
            "storage_type": "learned_scorer",
            "inference_state": {
                "config": {**asdict(model.config), "temperature": float(model.config.temperature)},
                "initialization_seed": scoring_initialization_seed(snapshot),
                "scorer_training": model.scorer_training,
                "scorer_state": {"format": "torch-state-dict-v1",
                                 "data": base64.b64encode(payload).decode("ascii"),
                                 "sha256": hashlib.sha256(payload).hexdigest()},
            },
            "trained_at": report.trained_at,
            "data_hash": snapshot.data_hash,
            "rule_weights": model.mln.rule_weights.detach().cpu().tolist(),
            "schema_snapshot": {
                "entity_count": len(kg.entities),
                "relation_count": len(kg.relations),
                "fact_count": len(kg.known_facts | kg.unknown_facts),
                "relation_names": sorted(r.name for r in kg.relations),
            },
            "training_report": report.to_dict(),
            "rules": [
                {"rule_id": r.rule_id, "rule_text": str(r), "confidence": r.confidence}
                for r in rules
            ],
            "version": ARTIFACT_VERSION,
        }
        validate_model_artifact(doc)
        self.model_store.save(MODEL_KEY, doc, expected_revision=expected_revision)

    def _generate_smart_rules(self, kg: KnowledgeGraph) -> List[LogicalRule]:
        """Generate unconditional priors and supported motifs for the exact vocabulary."""
        generator = RuleGenerator(kg)
        return (generator.generate_relation_priors() +
                generator.generate_simple_rules(min_support=2, min_confidence=0.2) +
                generator.generate_symmetry_rules(min_support=2, min_confidence=0.2))


def create_bootstrapper(triple_source: TripleSource, model_store: ModelStore) -> KnowledgeBootstrapper:
    """Factory function to create a KnowledgeBootstrapper."""
    return KnowledgeBootstrapper(triple_source, model_store)
