"""
Standard Knowledge Graph Benchmark Datasets

Provides loaders for:
- FB15k-237: Freebase subset (14,541 entities, 237 relations)
- WN18RR: WordNet subset (40,943 entities, 11 relations)

Datasets are downloaded from standard sources and cached locally.
"""

import os
import urllib.request
import tarfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Dataset URLs - using villmow/datasets_knowledge_embedding (reliable raw files)
DATASET_BASE_URLS = {
    "fb15k237": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237",
    "wn18rr": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR",
}

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "odin_benchmarks"


@dataclass
class BenchmarkDataset:
    """Container for a benchmark dataset."""
    name: str
    train_triples: List[Tuple[str, str, str]]
    valid_triples: List[Tuple[str, str, str]]
    test_triples: List[Tuple[str, str, str]]
    entities: List[str]
    relations: List[str]
    
    @property
    def num_entities(self) -> int:
        return len(self.entities)
    
    @property
    def num_relations(self) -> int:
        return len(self.relations)
    
    @property
    def num_train(self) -> int:
        return len(self.train_triples)
    
    @property
    def num_valid(self) -> int:
        return len(self.valid_triples)
    
    @property
    def num_test(self) -> int:
        return len(self.test_triples)
    
    def get_train_set(self) -> Set[Tuple[str, str, str]]:
        return set(self.train_triples)
    
    def get_all_triples(self) -> Set[Tuple[str, str, str]]:
        return set(self.train_triples + self.valid_triples + self.test_triples)
    
    def __repr__(self) -> str:
        return (
            f"BenchmarkDataset({self.name})\n"
            f"  Entities: {self.num_entities:,}\n"
            f"  Relations: {self.num_relations}\n"
            f"  Train: {self.num_train:,}\n"
            f"  Valid: {self.num_valid:,}\n"
            f"  Test: {self.num_test:,}"
        )


def _ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, dest: Path):
    """Download a file with progress."""
    import ssl
    import certifi
    
    logger.info(f"Downloading {url}...")
    
    # Try with certifi SSL context first, fall back to unverified
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(dest, 'wb') as out_file:
                out_file.write(response.read())
    except (ImportError, ssl.SSLError):
        # Fallback: disable SSL verification (for development only)
        logger.warning("SSL verification disabled - using unverified context")
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(dest, 'wb') as out_file:
                out_file.write(response.read())
    
    logger.info(f"Downloaded to {dest}")


def _extract_tar_gz(archive: Path, dest_dir: Path):
    """Extract a tar.gz archive."""
    logger.info(f"Extracting {archive}...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest_dir)


def _extract_zip(archive: Path, dest_dir: Path):
    """Extract a zip archive."""
    logger.info(f"Extracting {archive}...")
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)


def _load_triples(filepath: Path) -> List[Tuple[str, str, str]]:
    """Load triples from a TSV file (head, relation, tail)."""
    triples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                h, r, t = parts
                triples.append((h, r, t))
    return triples


def _extract_entities_and_relations(
    triples: List[Tuple[str, str, str]]
) -> Tuple[List[str], List[str]]:
    """Extract unique entities and relations from triples."""
    entities = set()
    relations = set()
    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    return sorted(entities), sorted(relations)


def _download_dataset_files(base_url: str, dataset_dir: Path):
    """Download train/valid/test files for a dataset."""
    _ensure_dir(dataset_dir)
    
    for split in ["train", "valid", "test"]:
        file_path = dataset_dir / f"{split}.txt"
        if not file_path.exists():
            url = f"{base_url}/{split}.txt"
            _download_file(url, file_path)


def load_fb15k237(cache_dir: Optional[Path] = None) -> BenchmarkDataset:
    """
    Load FB15k-237 dataset.
    
    FB15k-237 is a subset of Freebase with:
    - 14,541 entities
    - 237 relations
    - 310,116 triples
    
    This version removes inverse relations from FB15k to prevent
    data leakage during evaluation.
    
    Args:
        cache_dir: Directory to cache downloaded data
    
    Returns:
        BenchmarkDataset object
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    _ensure_dir(cache_dir)
    
    dataset_dir = cache_dir / "FB15k-237"
    
    # Download if not cached
    if not (dataset_dir / "train.txt").exists():
        _download_dataset_files(DATASET_BASE_URLS["fb15k237"], dataset_dir)
    
    # Load splits
    train = _load_triples(dataset_dir / "train.txt")
    valid = _load_triples(dataset_dir / "valid.txt")
    test = _load_triples(dataset_dir / "test.txt")
    
    # Extract vocab
    all_triples = train + valid + test
    entities, relations = _extract_entities_and_relations(all_triples)
    
    return BenchmarkDataset(
        name="FB15k-237",
        train_triples=train,
        valid_triples=valid,
        test_triples=test,
        entities=entities,
        relations=relations,
    )


def load_wn18rr(cache_dir: Optional[Path] = None) -> BenchmarkDataset:
    """
    Load WN18RR dataset.
    
    WN18RR is a subset of WordNet with:
    - 40,943 entities
    - 11 relations
    - 93,003 triples
    
    This version removes inverse relations from WN18 to prevent
    data leakage during evaluation.
    
    Args:
        cache_dir: Directory to cache downloaded data
    
    Returns:
        BenchmarkDataset object
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    _ensure_dir(cache_dir)
    
    dataset_dir = cache_dir / "WN18RR"
    
    # Download if not cached
    if not (dataset_dir / "train.txt").exists():
        _download_dataset_files(DATASET_BASE_URLS["wn18rr"], dataset_dir)
    
    # Load splits
    train = _load_triples(dataset_dir / "train.txt")
    valid = _load_triples(dataset_dir / "valid.txt")
    test = _load_triples(dataset_dir / "test.txt")
    
    # Extract vocab
    all_triples = train + valid + test
    entities, relations = _extract_entities_and_relations(all_triples)
    
    return BenchmarkDataset(
        name="WN18RR",
        train_triples=train,
        valid_triples=valid,
        test_triples=test,
        entities=entities,
        relations=relations,
    )


def dataset_to_kg(dataset: BenchmarkDataset):
    """
    Convert BenchmarkDataset to Odin KnowledgeGraph.
    
    Args:
        dataset: BenchmarkDataset object
    
    Returns:
        KnowledgeGraph object suitable for NPLL training
    """
    from npll.core import KnowledgeGraph
    
    kg = KnowledgeGraph()
    
    # Add all training triples as known facts
    for h, r, t in dataset.train_triples:
        kg.add_known_fact(h, r, t)
    
    return kg


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Loading FB15k-237...")
    fb = load_fb15k237()
    print(fb)
    print()
    
    print("Loading WN18RR...")
    wn = load_wn18rr()
    print(wn)
