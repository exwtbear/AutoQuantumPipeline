import itertools
from models.schemas import GraphProblem, BruteForceResult


def _cut_value(partition: tuple[int, ...], problem: GraphProblem) -> float:
    cut = 0.0
    for edge in problem.edges:
        if partition[edge.node_i] != partition[edge.node_j]:
            cut += edge.weight
    return cut


def solve_maxcut(problem: GraphProblem) -> BruteForceResult:
    """Enumerate all 2^n partitions and return the maximum cut."""
    n = problem.n_nodes
    best_cut = -1.0
    best_partition: tuple[int, ...] = tuple(0 for _ in range(n))

    # Fix node 0 to group 0 to avoid counting mirror solutions twice
    for bits in itertools.product([0, 1], repeat=n - 1):
        partition = (0,) + bits
        cut = _cut_value(partition, problem)
        if cut > best_cut:
            best_cut = cut
            best_partition = partition

    return BruteForceResult(
        best_partition=list(best_partition),
        best_cut_value=best_cut,
        node_labels=problem.node_labels,
    )
