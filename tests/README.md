# Odin KG Engine - Test Suite

Comprehensive testing strategy for production-ready Odin engine.

---

## Test Structure

```
tests/
├── unit/                    # Fast, isolated component tests
│   ├── test_cache.py       # CachedGraphAccessor tests
│   ├── test_confidence_cache.py  # NPLLConfidence cache tests
│   ├── test_ppr_engines.py # PageRank algorithm tests
│   ├── test_scoring.py     # Path/insight scoring tests
│   └── test_linker.py      # Entity linking tests
│
├── integration/             # End-to-end pipeline tests
│   ├── test_orchestrator.py  # Full retrieval pipeline
│   └── test_production_fixes.py  # New features integration
│
├── performance/             # Performance & regression tests
│   └── test_regression.py  # Memory leaks, latency benchmarks
│
└── smoke_test.py           # Quick sanity check
```

---

## Test Categories

### 1. Unit Tests (Fast, Isolated)

**Purpose:** Test individual components in isolation  
**When to run:** On every commit, before merge  
**Expected duration:** <10 seconds

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific component tests
pytest tests/unit/test_cache.py -v
pytest tests/unit/test_confidence_cache.py -v
```

**Critical Tests:**
- ✅ `test_cache.py` - Verifies CachedGraphAccessor LRU behavior
- ✅ `test_confidence_cache.py` - Verifies NPLLConfidence memory bounds

---

### 2. Integration Tests (Medium, End-to-End)

**Purpose:** Test complete workflows with all components  
**When to run:** Before deployment, after major changes  
**Expected duration:** <60 seconds

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/integration/ --cov=retrieval --cov-report=html
```

**Critical Tests:**
- ✅ `test_production_fixes.py` - Verifies new features work in pipeline

---

### 3. Performance Tests (Slow, Regression Detection)

**Purpose:** Catch memory leaks, latency regressions  
**When to run:** Before production deployment, weekly in CI  
**Expected duration:** 2-5 minutes

```bash
# Run performance tests
pytest tests/performance/ -v -m performance

# Run memory leak tests specifically
pytest tests/performance/test_regression.py::TestMemoryLeaks -v
```

**Critical Tests:**
- ✅ `test_npll_confidence_no_memory_leak` - Simulates 24-hour operation (100K inferences)
- ✅ `test_cache_accessor_no_memory_leak` - Verifies bounded cache growth
- ✅ `test_cache_speedup` - Ensures cache provides >10x speedup

---

### 4. Smoke Test (Ultra-Fast)

**Purpose:** Quick sanity check that nothing is broken  
**When to run:** On every change, as pre-commit hook  
**Expected duration:** <1 second

```bash
pytest tests/smoke_test.py -v
```

---

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov psutil

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=retrieval --cov=npll --cov-report=html

# Run only fast tests (unit + smoke)
pytest tests/unit/ tests/smoke_test.py -v
```

### CI/CD Pipeline

```yaml
# Recommended GitHub Actions workflow

stages:
  - smoke_test:    # <1s - Run on every commit
      pytest tests/smoke_test.py
  
  - unit_tests:    # <10s - Run on every PR
      pytest tests/unit/ -v
  
  - integration:   # <60s - Run before merge
      pytest tests/integration/ -v
  
  - performance:   # 2-5min - Run nightly
      pytest tests/performance/ -v -m performance
```

---

## Test Markers

Tests are marked for selective execution:

```bash
# Run only performance tests
pytest -m performance

# Run only benchmark tests
pytest -m benchmark

# Skip slow tests
pytest -m "not performance"
```

**Available markers:**
- `@pytest.mark.performance` - Performance regression tests
- `@pytest.mark.benchmark` - Latency benchmarks
- `@pytest.mark.slow` - Tests that take >10s

---

## Critical Test Results (Baselines)

These are the expected results for key tests. **Deviations indicate regressions.**

| Test | Metric | Expected | Alert Threshold |
|------|--------|----------|-----------------|
| `test_cache_speedup` | Speedup | >10x | <5x |
| `test_cache_hit_rate_on_ppr_pattern` | Hit Rate | >80% | <70% |
| `test_npll_confidence_no_memory_leak` | Memory Growth (100K) | <50MB | >100MB |
| `test_cache_accessor_no_memory_leak` | Memory Growth (100K) | <50MB | >100MB |
| `test_cached_retrieval_latency` | Latency (small graph) | <1000ms | >2000ms |

---

## Production Readiness Checklist

Before deploying Odin to production, ensure:

### ✅ Unit Tests
- [ ] All unit tests pass
- [ ] New features have unit tests
- [ ] Edge cases are covered

### ✅ Integration Tests
- [ ] End-to-end retrieval works
- [ ] CachedGraphAccessor integrates properly
- [ ] Content hydration works in pipeline

### ✅ Performance Tests
- [ ] No memory leaks detected (NPLLConfidence)
- [ ] No memory leaks detected (CachedGraphAccessor)
- [ ] Cache provides >10x speedup
- [ ] Cache hit rate >80% on PPR pattern
- [ ] Latency within acceptable bounds

### ✅ Manual Testing
- [ ] Test with real ArangoDB connection
- [ ] Test with actual NPLL model
- [ ] Test discovery methods (get_top_entities, get_recent_entities)
- [ ] Test content hydration (get_document_content, get_entity_sources)
- [ ] Verify cache statistics in logs
- [ ] Verify NPLL errors are logged (not silent)

---

## Testing New Features

When adding new features to Odin, follow this testing pattern:

### 1. Write Unit Tests First (TDD)
```python
def test_new_feature():
    """Test that new feature works in isolation."""
    component = NewComponent()
    result = component.do_something()
    assert result == expected
```

### 2. Add Integration Test
```python
def test_new_feature_in_pipeline():
    """Test that new feature works in full pipeline."""
    orchestrator = RetrievalOrchestrator(...)
    result = orchestrator.retrieve(...)
    # Verify new feature was used
```

### 3. Add Performance Test (if applicable)
```python
@pytest.mark.performance
def test_new_feature_no_memory_leak():
    """Verify new feature doesn't leak memory."""
    # Run 100K iterations
    # Assert memory growth < threshold
```

---

## Debugging Test Failures

### Memory Leak Tests Failing?

```bash
# Run with memory profiling
pytest tests/performance/test_regression.py::test_npll_confidence_no_memory_leak -v -s

# Use memory_profiler for detailed analysis
pip install memory_profiler
python -m memory_profiler tests/performance/test_regression.py
```

### Cache Tests Failing?

```bash
# Check cache statistics
pytest tests/unit/test_cache.py -v -s

# The `-s` flag shows print statements with cache stats
```

### Integration Tests Failing?

```bash
# Run with full traceback
pytest tests/integration/ -v --tb=long

# Run single test for debugging
pytest tests/integration/test_production_fixes.py::TestCachedAccessorIntegration::test_retrieval_with_cache -v -s
```

---

## Continuous Monitoring

### In Production

Monitor these metrics to detect regressions:

1. **Cache Hit Rate** (from `cache_stats()`)
   - Expected: >80% after warm-up
   - Alert if: <70%

2. **Memory Growth** (process RSS)
   - Expected: Plateau after warm-up
   - Alert if: Linear growth over 24 hours

3. **NPLL Errors** (from logs)
   - Expected: <1% failure rate
   - Alert if: >5% failures

4. **Retrieval Latency** (from `orchestrator.retrieve()`)
   - Expected: <2s for typical queries
   - Alert if: >5s

### Monitoring Setup

```python
# In production code
import logging
logger = logging.getLogger(__name__)

# After each retrieval
cache_stats = cached_accessor.cache_stats()
logger.info(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")

npll_stats = confidence.cache_stats()
logger.info(f"NPLL cache size: {npll_stats['size']}")
```

---

## Test Dependencies

Required packages:
```
pytest>=7.0.0
pytest-cov>=3.0.0
psutil>=5.9.0
torch>=1.9.0
```

Install:
```bash
pip install pytest pytest-cov psutil
```

---

## Contributing

When contributing to Odin:

1. **Write tests first** (TDD approach)
2. **Run unit tests locally** before committing
3. **Ensure all tests pass** before submitting PR
4. **Add performance tests** for any caching/memory-intensive features
5. **Update this README** if adding new test categories

---

## Questions?

- Memory leak in tests? → Check `psutil` is installed
- Tests too slow? → Run only unit tests: `pytest tests/unit/`
- Need to debug? → Use `-v -s` flags to see output
- Coverage report? → Use `--cov=retrieval --cov-report=html`

For production issues, check `PRODUCTION_FIXES.md` for debugging guide.
