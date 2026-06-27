from pydantic import BaseModel, field_validator, model_validator
from typing import Any


class Edge(BaseModel):
    node_i: int
    node_j: int
    weight: float


class GraphProblem(BaseModel):
    n_nodes: int
    edges: list[Edge]
    node_labels: list[str]

    @field_validator("n_nodes")
    @classmethod
    def n_nodes_positive(cls, v: int) -> int:
        if v < 2:
            raise ValueError("n_nodes must be at least 2")
        return v

    @model_validator(mode="after")
    def labels_match_nodes(self) -> "GraphProblem":
        if len(self.node_labels) != self.n_nodes:
            raise ValueError(
                f"node_labels length ({len(self.node_labels)}) must equal n_nodes ({self.n_nodes})"
            )
        return self


class CheckItem(BaseModel):
    name: str
    passed: bool
    message: str
    is_warning: bool = False  # True = warning only, does not block pipeline


class ValidationResult(BaseModel):
    passed: bool
    checks: list[CheckItem]

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            if c.is_warning:
                status = "⚠"
            elif c.passed:
                status = "✓"
            else:
                status = "✗"
            lines.append(f"  [{status}] {c.name}: {c.message}")
        return "\n".join(lines)


class SimulationResult(BaseModel):
    best_bitstring: str
    best_cut_value: float
    counts: dict[str, int]
    optimal_params: list[float]
    convergence_history: list[float] = []


class BruteForceResult(BaseModel):
    best_partition: list[int]
    best_cut_value: float
    node_labels: list[str]

    def group_a(self) -> list[str]:
        return [self.node_labels[i] for i, b in enumerate(self.best_partition) if b == 0]

    def group_b(self) -> list[str]:
        return [self.node_labels[i] for i, b in enumerate(self.best_partition) if b == 1]
