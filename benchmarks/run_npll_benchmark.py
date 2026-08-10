#!/usr/bin/env python3
"""
NPLL Benchmark Runner

Evaluates NPLL on standard KG completion benchmarks (FB15k-237, WN18RR)
and reports standard metrics (MRR, Hits@K).

Usage:
    python -m benchmarks.run_npll_benchmark --dataset fb15k237
    python -m benchmarks.run_npll_benchmark --dataset wn18rr
    python -m benchmarks.run_npll_benchmark --dataset fb15k237 --test-subset 1000
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch

from benchmarks.datasets import load_fb15k237, load_wn18rr, dataset_to_kg, BenchmarkDataset
from benchmarks.metrics import LinkPredictionEvaluator, evaluate_rankings
from npll import NPLLModel
from npll.core import KnowledgeGraph, RuleGenerator
from npll.utils import NPLLConfig, get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset(name: str) -> BenchmarkDataset:
    """Load dataset by name."""
    if name.lower() == "fb15k237":
        return load_fb15k237()
    elif name.lower() == "wn18rr":
        return load_wn18rr()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def create_npll_model(kg: KnowledgeGraph, config: NPLLConfig) -> NPLLModel:
    """Create and initialize NPLL model."""
    model = NPLLModel(config)
    
    # Generate rules from the knowledge graph
    logger.info("Generating logical rules...")
    rule_gen = RuleGenerator()
    rules = rule_gen.generate_rules(kg, max_chain_length=2)
    
    # If no rules generated, add universal rules
    if not rules:
        logger.warning("No rules generated, adding universal fallback rules")
        rules = rule_gen.generate_universal_rules(kg)
    
    logger.info(f"Generated {len(rules)} rules")
    
    # Initialize with KG and rules
    model.initialize(kg, rules)
    
    return model


def train_npll(model: NPLLModel, epochs: int = 10) -> Dict[str, Any]:
    """Train NPLL model."""
    logger.info(f"Training NPLL for {epochs} epochs...")
    start_time = time.time()
    
    training_state = model.train_model(
        num_epochs=epochs,
        em_iterations=5,
        verbose=True,
    )
    
    training_time = time.time() - start_time
    
    return {
        "epochs": epochs,
        "training_time_seconds": training_time,
        "final_elbo": training_state.best_elbo if training_state else None,
    }


def evaluate_npll(
    model: NPLLModel,
    dataset: BenchmarkDataset,
    test_subset: int = None,
) -> Dict[str, float]:
    """
    Evaluate NPLL on link prediction task.
    
    Args:
        model: Trained NPLL model
        dataset: Benchmark dataset
        test_subset: Limit test to first N triples (for faster evaluation)
    
    Returns:
        Metrics dictionary
    """
    logger.info("Evaluating on link prediction task...")
    
    # Prepare test triples
    test_triples = dataset.test_triples
    if test_subset:
        test_triples = test_triples[:test_subset]
        logger.info(f"Using subset of {len(test_triples)} test triples")
    
    # Create evaluator
    evaluator = LinkPredictionEvaluator(
        all_entities=dataset.entities,
        train_triples=dataset.get_train_set(),
        valid_triples=set(dataset.valid_triples),
    )
    
    # Define scoring function using NPLL
    def score_fn(h: str, r: str, t: str) -> float:
        try:
            # Use NPLL model to score the triple
            scores = model.score_triples([(h, r, t)])
            return float(scores[0]) if scores else 0.0
        except Exception:
            return 0.0
    
    # Run evaluation
    start_time = time.time()
    metrics = evaluator.evaluate_batch(test_triples, score_fn, verbose=True)
    eval_time = time.time() - start_time
    
    metrics["evaluation_time_seconds"] = eval_time
    metrics["triples_evaluated"] = len(test_triples)
    
    return metrics


def run_benchmark(
    dataset_name: str,
    epochs: int = 10,
    test_subset: int = None,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Run full NPLL benchmark.
    
    Args:
        dataset_name: Name of dataset (fb15k237 or wn18rr)
        epochs: Training epochs
        test_subset: Limit test evaluation (for speed)
        output_dir: Directory to save results
    
    Returns:
        Full results dictionary
    """
    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "config": {},
        "dataset_stats": {},
        "training": {},
        "evaluation": {},
    }
    
    # Load dataset
    logger.info(f"Loading {dataset_name}...")
    dataset = load_dataset(dataset_name)
    logger.info(f"\n{dataset}")
    
    results["dataset_stats"] = {
        "num_entities": dataset.num_entities,
        "num_relations": dataset.num_relations,
        "num_train": dataset.num_train,
        "num_valid": dataset.num_valid,
        "num_test": dataset.num_test,
    }
    
    # Convert to KnowledgeGraph
    logger.info("Converting to KnowledgeGraph...")
    kg = dataset_to_kg(dataset)
    
    # Create NPLL config
    config = get_config()
    config.embedding_dim = 100
    config.hidden_dim = 200
    results["config"] = {
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
    }
    
    # Create and train model
    logger.info("Creating NPLL model...")
    model = create_npll_model(kg, config)
    
    training_results = train_npll(model, epochs=epochs)
    results["training"] = training_results
    
    # Evaluate
    eval_results = evaluate_npll(model, dataset, test_subset=test_subset)
    results["evaluation"] = eval_results
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"NPLL BENCHMARK RESULTS: {dataset_name}")
    print("=" * 60)
    print(f"Dataset: {dataset.num_entities:,} entities, {dataset.num_relations} relations")
    print(f"Training: {training_results['training_time_seconds']:.1f}s")
    print("-" * 60)
    print("METRICS (Filtered Setting):")
    print(f"  MRR:      {eval_results['mrr']:.4f}")
    print(f"  Hits@1:   {eval_results['hits@1']:.4f}")
    print(f"  Hits@3:   {eval_results['hits@3']:.4f}")
    print(f"  Hits@10:  {eval_results['hits@10']:.4f}")
    print(f"  Mean Rank: {eval_results['mean_rank']:.1f}")
    print("=" * 60)
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"npll_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="NPLL Benchmark Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fb15k237",
        choices=["fb15k237", "wn18rr"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--test-subset",
        type=int,
        default=None,
        help="Limit test evaluation to first N triples (for speed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results",
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        dataset_name=args.dataset,
        epochs=args.epochs,
        test_subset=args.test_subset,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
