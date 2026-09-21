"""
Bootstrap module for NPLL.
Handles the end-to-end lifecycle of the NPLL model:
1. Reading a backend-neutral training snapshot
2. Generating domain-appropriate logical rules
3. Training the model
4. Persisting rule weights, rules, schema, and the complete training report

Architecture:
- The injected ModelStore owns persistence
- Model rebuilt from the same training snapshot on each load
- No external files needed
"""

import logging
import random
import torch
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional, Any
from retrieval.backends.base import (
    ARTIFACT_VERSION, MODEL_KEY, CorruptModelError, ModelStore, StoredModel,
    TrainingSnapshot, TripleSource, validate_model_artifact,
)

from .core.knowledge_graph import KnowledgeGraph, load_knowledge_graph_from_triples
from .core.logical_rules import LogicalRule, Atom, Variable, RuleType
from .npll_model import create_initialized_npll_model, NPLLModel
from .training.npll_trainer import TrainingConfig, TrainingResult, create_trainer
from .utils.config import get_config

logger = logging.getLogger(__name__)


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
    - Rule weights and audit metadata are saved through ModelStore
    - Model is rebuilt from KG data on each load
    - No external .pt files needed
    """
    
    def __init__(self, triple_source: TripleSource, model_store: ModelStore):
        self.triple_source = triple_source
        self.model_store = model_store

    def ensure_model_ready(self, force_retrain: bool = False) -> BootstrapResult:
        """
        Ensures a trained NPLL model is available.
        
        Flow:
        1. Extract a snapshot with its content fingerprint
        2. Read the model artifact and its revision from ModelStore
        3. If found: rebuild model from KG, apply saved weights
        4. If not found: train new model, save weights to DB
        
        Args:
            force_retrain: If True, retrains using the current store revision.
                Corrupt artifacts and store failures still raise.
            
        Returns:
            BootstrapResult with the model (or None on failure), the source
            of the weights, and the TrainingReport audit record when available.
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
        model, report = self._train_and_save_weights(
            snapshot, stored.revision if stored is not None else None,
        )
        source = "trained" if model else "failed"
        return BootstrapResult(model=model, source=source, data_hash=current_hash, report=report)

    def _load_model_with_weights(
        self, snapshot: TrainingSnapshot, stored: Optional[StoredModel],
    ) -> Tuple[Optional[NPLLModel], Optional[TrainingReport]]:
        if stored is None or stored.document["data_hash"] != snapshot.data_hash:
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
        config = get_config("OdinTriples")
        model = create_initialized_npll_model(kg, rules, config)
        with torch.no_grad():
            model.mln.rule_weights.copy_(torch.tensor(doc["rule_weights"], dtype=torch.float32))
        logger.info("Model rebuilt with saved weights (trained: %s)", doc["trained_at"])
        return model, report

    def _train_and_save_weights(
        self, snapshot: TrainingSnapshot, expected_revision: Optional[str],
    ) -> Tuple[Optional[NPLLModel], Optional[TrainingReport]]:
        """
        Train a new NPLL model and save ONLY the weights to database.
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
        known_facts_list = list(kg.known_facts)
        random.seed(42)
        num_unknown = max(1, len(known_facts_list) // 10)
        unknown_facts = random.sample(known_facts_list, num_unknown)
        
        for fact in unknown_facts:
            kg.known_facts.remove(fact)
            kg.add_unknown_fact(fact.head.name, fact.relation.name, fact.tail.name)
        
        # 3. Generate Rules
        rules = self._generate_smart_rules(kg)
        logger.info(f"Generated {len(rules)} logical rules")
        
        if not rules:
            logger.error("No rules generated. Cannot train.")
            return None, None
        
        # 4. Initialize Model
        config = get_config("OdinTriples")
        model = create_initialized_npll_model(kg, rules, config)
        
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
            logger.error(f"Training failed: {e}", exc_info=True)
            return None, None
        
        trained_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        report = TrainingReport.from_training_result(training_result, trained_at)
        if not report.converged:
            logger.warning(
                "NPLL training finished WITHOUT convergence "
                f"(epochs={report.total_epochs}, em_iterations={report.total_em_iterations}, "
                f"final_elbo={report.final_elbo:.6f}). Edge confidences may be poorly calibrated."
            )
        
        # 6. Save ONLY weights to database
        self._save_weights(model, kg, rules, snapshot.data_hash, report, expected_revision)
        
        return model, report

    def _save_weights(self, model: NPLLModel, kg: KnowledgeGraph,
                      rules: List[LogicalRule], data_hash: str,
                      report: TrainingReport, expected_revision: Optional[str]):
        doc = {
            "model_type": "npll",
            "storage_type": "weights_only",
            "trained_at": report.trained_at,
            "data_hash": data_hash,
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
        """
        Generates domain-appropriate rules based on available relations.
        """
        rules = []
        relations = {r.name: r for r in kg.relations}
        x, y, z = Variable("?x"), Variable("?y"), Variable("?z")
        
        logger.info(f"Generating rules for {len(relations)} relation types...")
        
        # --- HEALTHCARE DOMAIN ---
        if 'has_claim' in relations and 'submitted_by_provider' in relations and 'treated_by' in relations:
            rules.append(LogicalRule(
                rule_id="hc_claim_provider_link",
                body=[
                    Atom(relations['has_claim'], (x, y)),
                    Atom(relations['submitted_by_provider'], (y, z))
                ],
                head=Atom(relations['treated_by'], (x, z)),
                confidence=0.7
            ))
            logger.info("  + Added: hc_claim_provider_link")
        
        if 'diagnosed_with' in relations and 'indicates' in relations:
            target_rel = relations.get('recommended_procedure') or relations.get('related_to')
            if target_rel:
                rules.append(LogicalRule(
                    rule_id="hc_diagnosis_procedure",
                    body=[
                        Atom(relations['diagnosed_with'], (x, y)),
                        Atom(relations['indicates'], (y, z))
                    ],
                    head=Atom(target_rel, (x, z)),
                    confidence=0.6
                ))
                logger.info("  + Added: hc_diagnosis_procedure")

        if 'works_at' in relations and 'located_at' in relations:
            target_rel = relations.get('affiliated_with') or relations.get('related_to')
            if target_rel:
                rules.append(LogicalRule(
                    rule_id="hc_provider_facility",
                    body=[
                        Atom(relations['works_at'], (x, y)),
                        Atom(relations['located_at'], (y, z))
                    ],
                    head=Atom(target_rel, (x, z)),
                    confidence=0.6
                ))
                logger.info("  + Added: hc_provider_facility")

        # --- INSURANCE DOMAIN ---
        if 'policyholder' in relations and 'claim_number' in relations and 'related_to' in relations:
            rules.append(LogicalRule(
                rule_id="ins_policy_claim",
                body=[
                    Atom(relations['policyholder'], (x, y)),
                    Atom(relations['claim_number'], (x, z))
                ],
                head=Atom(relations['related_to'], (y, z)),
                confidence=0.8
            ))
            logger.info("  + Added: ins_policy_claim")

        if 'assessor' in relations and 'insurer' in relations and 'related_to' in relations:
            rules.append(LogicalRule(
                rule_id="ins_assessor_insurer",
                body=[
                    Atom(relations['assessor'], (x, y)),
                    Atom(relations['insurer'], (z, y))
                ],
                head=Atom(relations['related_to'], (x, z)),
                confidence=0.7
            ))
            logger.info("  + Added: ins_assessor_insurer")

        # --- GENERIC RULES ---
        if 'related_to' in relations:
            rules.append(LogicalRule(
                rule_id="gen_transitivity",
                body=[
                    Atom(relations['related_to'], (x, y)),
                    Atom(relations['related_to'], (y, z))
                ],
                head=Atom(relations['related_to'], (x, z)),
                rule_type=RuleType.TRANSITIVITY,
                confidence=0.5
            ))
            logger.info("  + Added: gen_transitivity")

        if 'has_type' in relations and 'related_to' in relations:
            rules.append(LogicalRule(
                rule_id="gen_type_cooccurrence",
                body=[
                    Atom(relations['has_type'], (x, y)),
                    Atom(relations['has_type'], (z, y))
                ],
                head=Atom(relations['related_to'], (x, z)),
                confidence=0.3
            ))
            logger.info("  + Added: gen_type_cooccurrence")

        # Fallback
        if not rules:
            logger.warning("No domain rules matched. Creating fallback.")
            rel = min(kg.relations, key=lambda relation: relation.name)
            rules.append(LogicalRule(
                rule_id="fallback_self",
                body=[Atom(rel, (x, y))],
                head=Atom(rel, (x, y)),
                confidence=0.5
            ))
        
        logger.info(f"Total rules: {len(rules)}")
        return rules


def create_bootstrapper(triple_source: TripleSource, model_store: ModelStore) -> KnowledgeBootstrapper:
    """Factory function to create a KnowledgeBootstrapper."""
    return KnowledgeBootstrapper(triple_source, model_store)
