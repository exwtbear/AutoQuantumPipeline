from qamomile.optimization.qaoa import QAOAConverter
from qamomile.optimization.binary_model import BinaryModel
from qamomile.qiskit import QiskitTranspiler
from qamomile.circuit.transpiler.executable import ExecutableProgram
from models.schemas import GraphProblem


def _build_maxcut_qubo(problem: GraphProblem) -> dict[tuple[int, int], float]:
    """Build a QUBO dict for MaxCut (minimise → maximise cut).

    For each edge (i, j, w):
      • linear diagonal  (i,i) and (j,j) each get  -w
      • quadratic term   (min,max) gets             +2w

    Minimising x^T Q x over {0,1}^n yields the maximum cut.
    """
    qubo: dict[tuple[int, int], float] = {}
    for edge in problem.edges:
        i, j, w = edge.node_i, edge.node_j, edge.weight
        qubo[(i, i)] = qubo.get((i, i), 0.0) - w
        qubo[(j, j)] = qubo.get((j, j), 0.0) - w
        key = (min(i, j), max(i, j))
        qubo[key] = qubo.get(key, 0.0) + 2.0 * w
    return qubo


def build_qaoa_executable(
    problem: GraphProblem, p: int = 1
) -> tuple[ExecutableProgram, QAOAConverter, QiskitTranspiler]:
    """Convert a MaxCut GraphProblem into a Qamomile ExecutableProgram.

    Returns:
        (executable, converter, transpiler)
        - executable : call .sample(transpiler.executor(), shots=N,
                         bindings={"gammas": [...], "betas": [...]}).result()
        - converter  : call .decode(result) → BinarySampleSet
        - transpiler : needed to create the executor
    """
    qubo = _build_maxcut_qubo(problem)
    model = BinaryModel.from_qubo(qubo)
    converter = QAOAConverter(model)
    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=p)
    return executable, converter, transpiler
