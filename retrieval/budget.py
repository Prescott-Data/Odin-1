from __future__ import annotations
from dataclasses import dataclass
import time


@dataclass
class SearchBudget:
    max_nodes: int = 2_000
    max_edges: int = 10_000
    max_ms: int = 1_500
    max_paths: int = 200


@dataclass
class Usage:
    nodes: int = 0
    edges: int = 0
    ms: int = 0
    paths: int = 0


class BudgetTracker:
    def __init__(self, budget: SearchBudget):
        self.budget = budget
        self.usage = Usage()
        self._start = time.perf_counter()

    def tick_nodes(self, n=1):
        self.usage.nodes += n

    def tick_edges(self, n=1):
        self.usage.edges += n

    def tick_paths(self, n=1):
        self.usage.paths += n

    def timed_out(self) -> bool:
        self.usage.ms = int((time.perf_counter() - self._start) * 1000)
        return self.usage.ms >= self.budget.max_ms

    def over(self) -> bool:
        self.timed_out()
        b, u = self.budget, self.usage
        return (
            u.nodes >= b.max_nodes
            or u.edges >= b.max_edges
            or u.paths >= b.max_paths
            or u.ms >= b.max_ms
        )

    def left(self) -> SearchBudget:
        self.timed_out()
        return SearchBudget(
            max_nodes=max(0, self.budget.max_nodes - self.usage.nodes),
            max_edges=max(0, self.budget.max_edges - self.usage.edges),
            max_ms=max(0, self.budget.max_ms - self.usage.ms),
            max_paths=max(0, self.budget.max_paths - self.usage.paths),
        )


