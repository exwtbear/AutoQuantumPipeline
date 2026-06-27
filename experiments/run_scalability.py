"""
experiments/run_scalability.py
Bypass Agent A entirely; construct random connected graphs at n=4..8 and
measure how QAOA circuit resources and performance scale with problem size.

Usage:
    python experiments/run_scalability.py
"""
import sys
import os
import csv
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from models.schemas import GraphProblem, Edge
from quantum.translator import build_qaoa_executable
from quantum.simulator import optimize_qaoa
from quantum.param_advisor import suggest_qaoa_params
from classical.brute_force import solve_maxcut

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCALABILITY_CSV = os.path.join(RESULTS_DIR, "scalability.csv")
N_VALUES = [4, 5, 6, 7, 8]
EDGE_PROB = 0.5
BASE_SEED = 42


def make_random_problem(n: int, seed: int = BASE_SEED) -> GraphProblem:
    G = nx.erdos_renyi_graph(n, EDGE_PROB, seed=seed)
    attempts = 0
    while not nx.is_connected(G) and attempts < 20:
        seed += 1
        attempts += 1
        G = nx.erdos_renyi_graph(n, EDGE_PROB, seed=seed)
    if not nx.is_connected(G):
        G = nx.path_graph(n)  # fallback: simple path
    edges = [Edge(node_i=int(u), node_j=int(v), weight=1.0) for u, v in G.edges()]
    return GraphProblem(
        n_nodes=n,
        edges=edges,
        node_labels=[f"V{i}" for i in range(n)],
    )


def main() -> None:
    rows = []
    print(f"[擴展性分析] n = {N_VALUES}, edge_prob={EDGE_PROB}, seed={BASE_SEED}\n")

    for n in N_VALUES:
        problem = make_random_problem(n)
        m = len(problem.edges)
        p, shots, _ = suggest_qaoa_params(problem)
        print(f"  n={n}, m={m}, AI選參: p={p}, shots={shots}")

        t0 = time.time()
        executable, converter, transpiler = build_qaoa_executable(problem, p=p)
        qc = executable.quantum_circuit

        bf = solve_maxcut(problem)
        bf_val = bf.best_cut_value

        sim = optimize_qaoa(executable, converter, transpiler, problem, p=p, shots=shots)
        elapsed = round(time.time() - t0, 2)

        expected = sim.convergence_history[-1] if sim.convergence_history else 0.0
        r = round(expected / bf_val, 4) if bf_val > 0 else 0.0
        total_w = sum(e.weight for e in problem.edges)
        rand_r = round((total_w / 2) / bf_val, 4) if bf_val > 0 else 0.0

        row = {
            "n_nodes": n,
            "n_edges": m,
            "p_chosen": p,
            "shots_chosen": shots,
            "circuit_depth": qc.depth(),
            "circuit_gates": qc.size(),
            "bf_cut_value": bf_val,
            "qaoa_expected_cut": round(expected, 4),
            "approximation_ratio": r,
            "random_baseline_ratio": rand_r,
            "runtime_seconds": elapsed,
        }
        rows.append(row)
        print(f"     depth={qc.depth()}, gates={qc.size()}, r={r:.4f}, "
              f"C_max={bf_val}  ({elapsed:.1f}s)")

    with open(SCALABILITY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV: {SCALABILITY_CSV}")

    print("\n擴展性摘要：")
    print(f"{'n':>3} {'m':>4} {'p':>3} {'深度':>6} {'閘數':>6} {'r':>8} {'時間':>7}")
    print("-" * 45)
    for r in rows:
        print(f"{r['n_nodes']:>3} {r['n_edges']:>4} {r['p_chosen']:>3} "
              f"{r['circuit_depth']:>6} {r['circuit_gates']:>6} "
              f"{r['approximation_ratio']:>8.4f} {r['runtime_seconds']:>6.1f}s")


if __name__ == "__main__":
    main()
