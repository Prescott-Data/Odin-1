from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time, json, os, threading
from .utils.pii_redaction import redact_dict


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks: Dict[str, float] = {}

    def mark(self, name: str):
        self.marks[name] = (time.perf_counter() - self.t0) * 1000.0

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)


class JSONLSink:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: Dict[str, Any]):
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')


@dataclass
class RetrievalMetrics:
    query_id: Optional[str]
    community_id: Optional[str]
    seeds_count: int
    ppr_mass: float
    topk: int
    used_budget: Dict[str, Any]
    latency_ms: int
    early_stop_reason: Optional[str]
    engine: str
    notes: Optional[Dict[str, Any]] = None

    def to_event(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsLogger:
    def __init__(self, sink: Optional[JSONLSink] = None, redact_pii: bool = True):
        self.sink = sink
        self.redact_pii = redact_pii

    def log(self, metrics: RetrievalMetrics):
        if self.sink:
            event = metrics.to_event()
            if self.redact_pii:
                event = redact_dict(event)
            self.sink.write(event)


def aggregate_latency_and_budget(jsonl_path: str) -> Dict[str, Any]:
    import numpy as np
    latencies, budget_hits = [], 0
    total = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if 'latency_ms' in ev:
                latencies.append(ev['latency_ms'])
            used = ev.get('used_budget', {})
            bud = used if isinstance(used, dict) else {}
            # Budget hit if any dimension equals its cap
            max_nodes = bud.get('max_nodes')
            max_edges = bud.get('max_edges')
            max_ms = bud.get('max_ms')
            max_paths = bud.get('max_paths')
            u_nodes = bud.get('nodes', -1)
            u_edges = bud.get('edges', -1)
            u_ms = bud.get('ms', -1)
            u_paths = bud.get('paths', -1)
            hit = (
                (max_nodes is not None and u_nodes >= max_nodes)
                or (max_edges is not None and u_edges >= max_edges)
                or (max_ms is not None and u_ms >= max_ms)
                or (max_paths is not None and u_paths >= max_paths)
            )
            if hit:
                budget_hits += 1
    if not latencies:
        return {"count": total, "p50_ms": None, "p95_ms": None, "budget_hit_rate": None}
    arr = np.array(latencies)
    return {
        "count": total,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "budget_hit_rate": (budget_hits / max(total, 1)),
    }


