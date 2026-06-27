import numpy as np
from scipy.optimize import minimize
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.qiskit import QiskitTranspiler
from qamomile.circuit.transpiler.executable import ExecutableProgram
from models.schemas import GraphProblem, SimulationResult


def compute_cut_value(sample: dict[int, int], problem: GraphProblem) -> float:
    """Evaluate Max-Cut objective for one sample (binary dict {qubit_idx: 0/1})."""
    cut = 0.0
    for edge in problem.edges:
        if sample.get(edge.node_i, 0) != sample.get(edge.node_j, 0):
            cut += edge.weight
    return cut


def _expected_cut(sampleset, problem: GraphProblem) -> float:
    total = sum(sampleset.num_occurrences)
    if total == 0:
        return 0.0
    return sum(
        compute_cut_value(s, problem) * n
        for s, n in zip(sampleset.samples, sampleset.num_occurrences)
    ) / total


def optimize_qaoa(
    executable: ExecutableProgram,
    converter: QAOAConverter,
    transpiler: QiskitTranspiler,
    problem: GraphProblem,
    p: int = 1,
    shots: int = 1024,
    max_iter: int = 200,
) -> SimulationResult:
    executor = transpiler.executor()
    rng = np.random.default_rng(42)
    x0 = rng.uniform(0, np.pi, size=2 * p)

    history: list[float] = []

    def cost_fn(params: np.ndarray) -> float:
        gammas = params[:p].tolist()
        betas = params[p:].tolist()
        result = executable.sample(
            executor,
            shots=shots,
            bindings={"gammas": gammas, "betas": betas},
        ).result()
        sampleset = converter.decode_to_binary_sampleset(result)
        expected = _expected_cut(sampleset, problem)
        history.append(expected)
        return -expected

    opt = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": max_iter, "rhobeg": 0.5})

    gammas_opt = opt.x[:p].tolist()
    betas_opt = opt.x[p:].tolist()
    final_result = executable.sample(
        executor,
        shots=shots * 4,
        bindings={"gammas": gammas_opt, "betas": betas_opt},
    ).result()
    final_sampleset = converter.decode_to_binary_sampleset(final_result)

    best_sample, _, _ = final_sampleset.lowest()
    best_cut = compute_cut_value(best_sample, problem)

    n = problem.n_nodes
    counts: dict[str, int] = {}
    for sample, occ in zip(final_sampleset.samples, final_sampleset.num_occurrences):
        bits = "".join(str(sample.get(i, 0)) for i in reversed(range(n)))
        counts[bits] = counts.get(bits, 0) + occ

    best_bitstring = "".join(
        str(best_sample.get(i, 0)) for i in reversed(range(n))
    )

    return SimulationResult(
        best_bitstring=best_bitstring,
        best_cut_value=best_cut,
        counts=counts,
        optimal_params=opt.x.tolist(),
        convergence_history=history,
    )
