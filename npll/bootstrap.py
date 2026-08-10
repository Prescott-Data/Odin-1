"""
Bootstrap module for NPLL.
Handles the end-to-end lifecycle of the NPLL model:
1. Extracting data from ArangoDB
2. Generating domain-appropriate logical rules
3. Training the model
4. Storing ONLY WEIGHTS in database (not full model)

Architecture:
- Weights stored in OdinModels collection (~1 KB)
- Model rebuilt from KG on each load (~30 sec)
- No external files needed
"""

import os
import hashlib
import json
import logging
import random
import time
import torch
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
from arango.database import StandardDatabase

from .core.knowledge_graph import KnowledgeGraph, load_knowledge_graph_from_triples
from .core.logical_rules import LogicalRule, Atom, Variable, RuleType
from .npll_model import create_initialized_npll_model, NPLLModel
from .training.npll_trainer import TrainingConfig, create_trainer
from .utils.config import get_config

logger = logging.getLogger(__name__)

# Collection name for storing model weights
ODIN_MODELS_COLLECTION = "OdinModels"
NPLL_MODEL_KEY = "npll_current"


class KnowledgeBootstrapper:
    """
    Manages the lifecycle of the NPLL model.
    
    Storage Strategy:
    - Only rule weights are saved to database (~1 KB)
    - Model is rebuilt from KG data on each load (~30 sec)
    - No external .pt files needed
    """
    
    def __init__(self, db: StandardDatabase):
        """
        Initialize the bootstrapper.
        
        Args:
            db: An already-connected ArangoDB database instance
        """
        self.db = db
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure OdinModels collection exists."""
        try:
            if not self.db.has_collection(ODIN_MODELS_COLLECTION):
                self.db.create_collection(ODIN_MODELS_COLLECTION)
                logger.info(f"Created {ODIN_MODELS_COLLECTION} collection")
        except Exception as e:
            # Suppress permission errors (common in read-only environments)
            logger.warning(f"Could not verify/create {ODIN_MODELS_COLLECTION} (Permission Error?): {e}")

    def ensure_model_ready(self, force_retrain: bool = False) -> Optional[NPLLModel]:
        """
        Ensures a trained NPLL model is available.
        
        Flow:
        1. Compute current data hash
        2. Check OdinModels for saved weights with matching hash
        3. If found: rebuild model from KG, apply saved weights
        4. If not found: train new model, save weights to DB
        
        Args:
            force_retrain: If True, ignores cached weights and retrains
            
        Returns:
            Loaded NPLLModel ready for inference, or None if failed
        """
        current_hash = self._compute_data_hash()
        logger.info(f"Current data hash: {current_hash[:16]}...")
        
        if not force_retrain:
            # Try to load existing weights and rebuild model
            model = self._load_model_with_weights(current_hash)
            if model:
                return model
        
        # Train new model
        logger.info("Training new NPLL model...")
        return self._train_and_save_weights(current_hash)

    def _compute_data_hash(self) -> str:
        """
        Compute a hash of the current schema to detect when retraining is needed.
        """
        try:
            # Get relation names
            rel_query = """
            FOR e IN ExtractedRelationships
              COLLECT rel = e.relationship WITH COUNT INTO cnt
              SORT rel
              RETURN {rel: rel, count: cnt}
            """
            relations = list(self.db.aql.execute(rel_query))
            relation_names = sorted([r['rel'] for r in relations if r['rel']])
            
            # Get counts
            entity_count = self.db.collection("ExtractedEntities").count()
            fact_count = self.db.collection("ExtractedRelationships").count()
            
            hash_input = {
                "relations": relation_names,
                "entity_count": entity_count,
                "fact_count": fact_count,
            }
            
            hash_str = json.dumps(hash_input, sort_keys=True)
            return hashlib.sha256(hash_str.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Could not compute data hash: {e}")
            return hashlib.sha256(str(time.time()).encode()).hexdigest()

    def _load_model_with_weights(self, expected_hash: str) -> Optional[NPLLModel]:
        """
        Load saved weights from DB and rebuild the model.
        
        Flow:
        1. Check if saved weights exist with matching hash
        2. Extract triples from DB → build KG
        3. Generate rules (same code = same rules)
        4. Initialize fresh model
        5. Apply saved weights
        """
        try:
            collection = self.db.collection(ODIN_MODELS_COLLECTION)
            doc = collection.get(NPLL_MODEL_KEY)
            
            if not doc:
                logger.info("No saved weights found in database")
                return None
            
            stored_hash = doc.get("data_hash", "")
            if stored_hash != expected_hash:
                logger.info(f"Data has changed. Stored: {stored_hash[:16]}..., Current: {expected_hash[:16]}...")
                return None
            
            # Get saved weights
            saved_weights = doc.get("rule_weights")
            if not saved_weights:
                logger.warning("No rule_weights in saved document")
                return None
            
            logger.info("Rebuilding model from KG and applying saved weights...")
            
            # 1. Extract triples
            triples = self._extract_triples()
            if not triples:
                return None
            
            # 2. Build KG
            kg = load_knowledge_graph_from_triples(triples, "ArangoDB_KG")
            
            # 3. Generate rules (same code = same rules)
            rules = self._generate_smart_rules(kg)
            
            if len(rules) != len(saved_weights):
                logger.warning(f"Rule count mismatch: {len(rules)} rules, {len(saved_weights)} weights. Retraining.")
                return None
            
            # 4. Initialize model
            config = get_config("ArangoDB_Triples")
            model = create_initialized_npll_model(kg, rules, config)
            
            # 5. Apply saved weights
            with torch.no_grad():
                model.mln.rule_weights.copy_(torch.tensor(saved_weights, dtype=torch.float32))
            
            trained_at = doc.get("trained_at", "unknown")
            logger.info(f"✓ Model rebuilt with saved weights (trained: {trained_at})")
            
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return None

    def _train_and_save_weights(self, data_hash: str) -> Optional[NPLLModel]:
        """
        Train a new NPLL model and save ONLY the weights to database.
        """
        # 1. Extract Triples
        triples = self._extract_triples()
        if not triples:
            logger.error("No triples extracted. Cannot train.")
            return None
        
        # 2. Build KG
        kg = load_knowledge_graph_from_triples(triples, "ArangoDB_KG")
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
            return None
        
        # 4. Initialize Model
        config = get_config("ArangoDB_Triples")
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
            return None
        
        # 6. Save ONLY weights to database
        self._save_weights_to_db(model, kg, rules, data_hash, training_result)
        
        return model

    def _save_weights_to_db(self, model: NPLLModel, kg: KnowledgeGraph, 
                            rules: List[LogicalRule], data_hash: str,
                            training_result: Any):
        """
        Save ONLY the learned weights to OdinModels collection.
        This is tiny (~1 KB) compared to the full model (280 MB).
        """
        try:
            # Extract just the rule weights
            rule_weights = model.mln.rule_weights.detach().cpu().tolist()
            
            doc = {
                "_key": NPLL_MODEL_KEY,
                "model_type": "npll",
                "storage_type": "weights_only",  # Mark this as weights-only storage
                "trained_at": datetime.utcnow().isoformat() + "Z",
                "data_hash": data_hash,
                "rule_weights": rule_weights,  # The learned weights - this is all we need!
                "schema_snapshot": {
                    "entity_count": len(kg.entities),
                    "relation_count": len(kg.relations),
                    "fact_count": len(kg.known_facts),
                    "relation_names": sorted([r.name for r in kg.relations])[:50],
                },
                "training_result": {
                    "final_elbo": float(training_result.final_elbo) if training_result else 0,
                    "best_elbo": float(training_result.best_elbo) if training_result else 0,
                    "converged": training_result.converged if training_result else False,
                    "training_time_seconds": training_result.total_training_time if training_result else 0,
                },
                "rules": [
                    {
                        "rule_id": r.rule_id,
                        "rule_text": str(r),
                        "confidence": r.confidence,
                    }
                    for r in rules
                ],
                "version": "2.0",  # Version 2 = weights-only storage
            }
            
            # Upsert
            collection = self.db.collection(ODIN_MODELS_COLLECTION)
            if collection.has(NPLL_MODEL_KEY):
                collection.update(doc)
            else:
                collection.insert(doc)
            
            weights_size = len(json.dumps(rule_weights))
            logger.info(f"✓ Saved rule weights to database ({weights_size} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to save weights: {e}", exc_info=True)

    def _extract_triples(self) -> List[Tuple[str, str, str]]:
        """Extracts S-P-O triples from ArangoDB."""
        logger.info("Extracting triples from database...")
        triples = []
        
        # Extract Relationships
        query = """
        FOR rel IN ExtractedRelationships
          LET source = DOCUMENT(rel._from)
          LET target = DOCUMENT(rel._to)
          FILTER source != null AND target != null 
          FILTER source._key != null AND target._key != null
          RETURN {
            source: source._key,
            target: target._key,
            relation: rel.relationship || "related_to"
          }
        """
        try:
            cursor = self.db.aql.execute(query)
            for doc in cursor:
                s, t = doc['source'], doc['target']
                r = str(doc['relation']).replace(' ', '_').lower()
                triples.append((s, r, t))
            logger.info(f"Extracted {len(triples)} relationship triples")
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return []
        
        # Extract Entity Types
        query_types = """
        FOR entity IN ExtractedEntities
          FILTER entity._key != null AND entity.type != null
          RETURN { key: entity._key, type: entity.type }
        """
        try:
            cursor = self.db.aql.execute(query_types)
            type_count = 0
            for doc in cursor:
                triples.append((doc['key'], 'has_type', doc['type']))
                type_count += 1
            logger.info(f"Extracted {type_count} entity type triples")
        except Exception as e:
            logger.error(f"Type extraction error: {e}")
        
        logger.info(f"Total triples: {len(triples)}")
        return triples

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
            rel = next(iter(kg.relations))
            rules.append(LogicalRule(
                rule_id="fallback_self",
                body=[Atom(rel, (x, y))],
                head=Atom(rel, (x, y)),
                confidence=0.5
            ))
        
        logger.info(f"Total rules: {len(rules)}")
        return rules


def create_bootstrapper(db: StandardDatabase) -> KnowledgeBootstrapper:
    """Factory function to create a KnowledgeBootstrapper."""
    return KnowledgeBootstrapper(db)
