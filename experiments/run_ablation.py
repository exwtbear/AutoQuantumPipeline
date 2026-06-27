"""
experiments/run_ablation.py
Run TC3 (weighted data-center graph) with p=1, 2, 3 (N_RUNS=5 each)
to show how QAOA depth affects approximation ratio (mean ± std).

Usage:
    python experiments/run_ablation.py
"""
import sys
import os
import csv
import json
import statistics
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.builder_agent import parse_graph_from_text
from agents.checker import validate
from quantum.translator import build_qaoa_executable
from quantum.simulator import optimize_qaoa
from classical.brute_force import solve_maxcut
from experiments.test_cases import TEST_CASES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ABLATION_CSV  = os.path.join(RESULTS_DIR, "ablation_p.csv")
ABLATION_JSON = os.path.join(RESULTS_DIR, "ablation_p.json")

SHOTS = 2048   # Increased to improve gradient estimates
N_RUNS = 5
P_VALUES = [1, 2, 3]


def main() -> None:
    tc3 = next(tc for tc in TEST_CASES if tc.tc_id == "TC3")

    print(f"[p 消融分析] {tc3.name} — p={P_VALUES}, {N_RUNS} 次重複, shots={SHOTS}")
    print("  → Agent A 解析（1 次）...")
    problem, attempts = parse_graph_from_text(tc3.nl_input)
    print(f"     ✓ n={problem.n_nodes}, m={len(problem.edges)}")

    validation = validate(problem)
    if not validation.passed:
        print("  ✗ Checker 未通過，中止")
        sys.exit(1)

    bf = solve_maxcut(problem)
    bf_val = bf.best_cut_value
    print(f"  → C_max = {bf_val}\n")

    rows = []
    json_out = []

    for p in P_VALUES:
        executable, converter, transpiler = build_qaoa_executable(problem, p=p)
        qc = executable.quantum_circuit
        depth = qc.depth()
        gates = qc.size()
        print(f"  p={p}  (depth={depth}, gates={gates})")

        run_ratios = []
        run_histories = []
        for run_id in range(1, N_RUNS + 1):
            t0 = time.time()
            sim = optimize_qaoa(executable, converter, transpiler, problem, p=p, shots=SHOTS)
            elapsed = round(time.time() - t0, 1)
            expected = sim.convergence_history[-1] if sim.convergence_history else 0.0
            r = round(expected / bf_val, 4) if bf_val > 0 else 0.0
            run_ratios.append(r)
            run_histories.append(sim.convergence_history)
            print(f"    Run {run_id}/{N_RUNS}: r={r:.4f}  ({elapsed}s)")

        mean_r = round(statistics.mean(run_ratios), 4)
        std_r  = round(statistics.stdev(run_ratios) if len(run_ratios) > 1 else 0.0, 4)
        print(f"    → mean={mean_r:.4f} ± {std_r:.4f}\n")

        row = {
            "p": p,
            "circuit_depth": depth,
            "circuit_gates": gates,
            "shots": SHOTS,
            "mean_r": mean_r,
            "std_r": std_r,
            "bf_cut_value": bf_val,
        }
        rows.append(row)
        json_out.append({**row, "run_ratios": run_ratios, "convergence_histories": run_histories})

    with open(ABLATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(ABLATION_JSON, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    print(f"  CSV:  {ABLATION_CSV}")
    print(f"  JSON: {ABLATION_JSON}")
    print(f"\n消融摘要（N={N_RUNS} 次平均）：")
    print(f"{'p':>3} {'深度':>6} {'閘數':>6} {'mean_r':>8} {'std_r':>7}")
    print("-" * 35)
    for r in rows:
        print(f"{r['p']:>3} {r['circuit_depth']:>6} {r['circuit_gates']:>6} "
              f"{r['mean_r']:>8.4f} {r['std_r']:>7.4f}")


if __name__ == "__main__":
    main()
