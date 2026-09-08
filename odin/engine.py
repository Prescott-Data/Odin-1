"""
OdinEngine: The main entry point for the Odin KG Intelligence Library.

This class orchestrates all components:
- Graph access (with caching)
- NPLL model management (auto-train if needed)
- Retrieval (PPR + Beam Search + Scoring)
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

from arango.database import StandardDatabase

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npll.bootstrap import KnowledgeBootstrapper, TrainingReport
from npll.npll_model import NPLLModel
from retrieval.orchestrator import RetrievalOrchestrator, OrchestratorParams
from retrieval.adapters_arango import ArangoCommunityAccessor, GlobalGraphAccessor
from retrieval.cache import CachedGraphAccessor
from retrieval.confidence import NPLLConfidence, ConstantConfidence
from retrieval.ppr.anchors import APPRAnchors, APPRAnchorParams

logger = logging.getLogger("odin")


class OdinEngine:
    """
    Main entry point for the Odin Knowledge Graph Intelligence Library.
    
    Handles:
    - Graph access (with caching)
    - NPLL model loading (auto-trains if needed)
    - Retrieval orchestration (PPR + Beam Search + NPLL Scoring)
    
    Example:
        from odin import OdinEngine
        from arango import ArangoClient
        
        client = ArangoClient(hosts="http://localhost:8529")
        db = client.db("KG-test", username="root", password="")
        
        engine = OdinEngine(db)
        results = engine.retrieve(seeds=["Patient_123"])
    """
    
    def __init__(
        self,
        db: StandardDatabase,
        community_id: str = "global",
        cache_size: int = 5000,
        auto_train: bool = True,
        community_mode: str = "none",  # "none" = global, "mapping" = scoped
    ):
        """
        Initialize the Odin Engine.
        
        Args:
            db: Connected ArangoDB database instance
            community_id: Community to scope queries to (default: "global")
            cache_size: Size of the graph accessor cache (default: 5000)
            auto_train: If True, automatically train NPLL if no model exists (default: True)
            community_mode: "none" for global exploration, "mapping" for community-scoped
        """
        self.db = db
        self.community_id = community_id
        
        logger.info(f"Initializing OdinEngine for community '{community_id}' (mode: {community_mode})...")
        
        # 1. Setup Graph Accessor (with caching)
        base_accessor = ArangoCommunityAccessor(
            db=db,
            community_id=community_id,
            community_mode=community_mode,
        )
        self.accessor = CachedGraphAccessor(base_accessor, cache_size=cache_size)
        
        # Global accessor for cross-community queries
        self.global_accessor = GlobalGraphAccessor(db=db, algorithm="gnn")
        
        # 2. Load/Train NPLL Model
        self.npll_model: Optional[NPLLModel] = None
        self.training_report: Optional[TrainingReport] = None
        self.npll_source: str = "disabled"
        self.confidence = self._initialize_intelligence(auto_train)
        
        # 3. Setup Orchestrator
        self.orchestrator = RetrievalOrchestrator(
            accessor=self.accessor,
            edge_confidence=self.confidence,
        )
        
        # 4. Setup PPR Anchor Engine
        self.anchor_engine = APPRAnchors(self.accessor)
        
        mode = "NPLL" if self.npll_model else "Fallback"
        logger.info(f"✓ OdinEngine initialized (Intelligence: {mode})")

    def _initialize_intelligence(self, auto_train: bool):
        """Load or train NPLL model."""
        if not auto_train:
            logger.info("Auto-train disabled. Using constant confidence.")
            return ConstantConfidence(0.8)
        
        try:
            bootstrapper = KnowledgeBootstrapper(db=self.db)
            result = bootstrapper.ensure_model_ready()
            self.npll_model = result.model
            self.training_report = result.report
            self.npll_source = result.source
            self._warn_if_not_converged()
            
            if self.npll_model:
                return NPLLConfidence(self.npll_model, cache_size=10000)
            else:
                logger.warning("NPLL training failed. Using constant confidence.")
                return ConstantConfidence(0.8)
                
        except Exception as e:
            logger.error(f"Failed to initialize NPLL: {e}")
            return ConstantConfidence(0.8)

    def _warn_if_not_converged(self):
        """Surface non-convergence so users know confidences may be miscalibrated."""
        if self.npll_model is None or self.training_report is None:
            return
        if not self.training_report.converged:
            logger.warning(
                "NPLL model in use did NOT converge during training "
                f"(source: {self.npll_source}, trained_at: {self.training_report.trained_at}, "
                f"final_elbo: {self.training_report.final_elbo:.6f}). "
                "Edge confidences may be poorly calibrated. "
                "Inspect engine.training_report or retrain with engine.retrain_model()."
            )

    def retrieve(
        self,
        seeds: List[str],
        max_paths: int = 50,
        hop_limit: int = 3,
        beam_width: int = 64,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant paths from seed nodes.
        
        Uses PPR + Beam Search + NPLL Scoring to find the most relevant
        paths in the knowledge graph starting from the given seeds.
        
        Args:
            seeds: List of starting node IDs (e.g., ["Patient_123", "Claim_456"])
            max_paths: Maximum number of paths to return (default: 50)
            hop_limit: Maximum path length (default: 3)
            beam_width: Beam search width (default: 64)
            
        Returns:
            Dict containing:
            - topk_ppr: Top nodes by PageRank importance
            - paths: Discovered paths with scores
            - insight_score: Overall quality score
            - aggregates: Motifs, relations, anchors
        """
        params = OrchestratorParams(
            community_id=self.community_id,
            max_paths=max_paths,
            hop_limit=hop_limit,
            beam_width=beam_width,
        )
        return self.orchestrator.retrieve(seeds=seeds, params=params)

    def score_edge(self, src: str, rel: str, dst: str) -> float:
        """
        Score how plausible an edge is (0.0 to 1.0).
        
        Uses the trained NPLL model to estimate the probability
        that the given edge (src --rel--> dst) is valid.
        
        Args:
            src: Source node ID
            rel: Relationship type
            dst: Destination node ID
            
        Returns:
            Probability score between 0.0 and 1.0
        """
        return self.confidence.confidence(src, rel, dst)

    def find_anchors(self, seeds: List[str], topn: int = 20) -> List[tuple]:
        """
        Use PPR (PageRank) to find the most important nodes relative to seeds.
        
        Args:
            seeds: Starting node IDs
            topn: Number of top nodes to return (default: 20)
            
        Returns:
            List of (node_id, ppr_score) tuples sorted by importance
        """
        params = APPRAnchorParams(topn=topn)
        return self.anchor_engine.build_for_community(
            community_id=self.community_id,
            seed_set=seeds,
            params=params,
        )

    def get_neighbors(self, node_id: str) -> Dict[str, Any]:
        """
        Get all neighbors of a node with relationship types.
        
        Args:
            node_id: The node to inspect
            
        Returns:
            Dict with node info and list of neighbors
        """
        node = self.accessor.get_node(node_id)
        
        neighbors = []
        for neighbor_id, relation, weight in self.accessor.iter_out(node_id):
            neighbors.append({
                "id": neighbor_id,
                "rel": relation,
                "weight": weight,
                "direction": "out"
            })
        
        for neighbor_id, relation, weight in self.accessor.iter_in(node_id):
            neighbors.append({
                "id": neighbor_id,
                "rel": relation,
                "weight": weight,
                "direction": "in"
            })
        
        return {
            "node": node,
            "neighbors": neighbors,
            "degree": len(neighbors),
        }

    def retrain_model(self) -> bool:
        """
        Force retrain the NPLL model.
        
        Useful after significant data changes.
        
        Returns:
            True if training succeeded, False otherwise
        """
        try:
            bootstrapper = KnowledgeBootstrapper(db=self.db)
            result = bootstrapper.ensure_model_ready(force_retrain=True)
            self.npll_model = result.model
            self.training_report = result.report
            self.npll_source = result.source
            self._warn_if_not_converged()
            
            if self.npll_model:
                self.confidence = NPLLConfidence(self.npll_model, cache_size=10000)
                self.orchestrator = RetrievalOrchestrator(
                    accessor=self.accessor,
                    edge_confidence=self.confidence,
                )
                logger.info("✓ Model retrained successfully")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return False

    @property
    def has_npll(self) -> bool:
        """Check if NPLL model is loaded."""
        return self.npll_model is not None

    def get_status(self) -> Dict[str, Any]:
        """Get engine status information."""
        return {
            "community_id": self.community_id,
            "npll_loaded": self.has_npll,
            "intelligence_mode": "NPLL" if self.has_npll else "Constant",
            "npll_source": self.npll_source,
            "npll_converged": self.training_report.converged if self.training_report else None,
            "npll_trained_at": self.training_report.trained_at if self.training_report else None,
            "cache_size": getattr(self.accessor, 'cache_size', 'unknown'),
        }
