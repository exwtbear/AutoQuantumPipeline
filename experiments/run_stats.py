"""
experiments/run_stats.py
Run each TC 5 times (QAOA only, parse LLM once per TC) and compute mean ± std.

Usage:
    python experiments/run_stats.py
"""
import sys
import os
import csv
import statistics
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.builder_agent import parse_graph_from_text
from agents.checker import validate
from quantum.translator import build_qaoa_executable
from quantum.simulator import optimize_qaoa
from quantum.param_advisor import suggest_qaoa_params
from classical.brute_force import solve_maxcut
from experiments.test_cases import TEST_CASES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

STATS_CSV   = os.path.join(RESULTS_DIR, "stats.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "stats_summary.csv")
N_RUNS = 5


def run_tc_stats(tc) -> list[dict]:
    print(f"\n[{tc.tc_id}] {tc.name} — {N_RUNS} 次重複執行")

    # Parse once (save LLM API cost)
    print("  → Agent A 解析（1 次）...")
    problem, attempts = parse_graph_from_text(tc.nl_input)
    print(f"     ✓ n={problem.n_nodes}, m={len(problem.edges)}")

    validation = validate(problem)
    if not validation.passed:
        print("  ✗ Checker 未通過，跳過")
        return []

    p, shots, _ = suggest_qaoa_params(problem)
    executable, converter, transpiler = build_qaoa_executable(problem, p=p)
    bf = solve_maxcut(problem)
    bf_val = bf.best_cut_value

    rows = []
    for run_id in range(1, N_RUNS + 1):
        t0 = time.time()
        sim = optimize_qaoa(executable, converter, transpiler, problem, p=p, shots=shots)
        elapsed = round(time.time() - t0, 2)
        expected = sim.convergence_history[-1] if sim.convergence_history else 0.0
        r = round(expected / bf_val, 4) if bf_val > 0 else 0.0
        rows.append({
            "tc_id": tc.tc_id,
            "name": tc.name,
            "run_id": run_id,
            "approximation_ratio": r,
            "qaoa_expected_cut": round(expected, 4),
            "bf_cut_value": bf_val,
            "runtime_seconds": elapsed,
        })
        print(f"     Run {run_id}/{N_RUNS}: r={r:.4f}  ({elapsed:.1f}s)")

    return rows


def main() -> None:
    all_rows = []
    for tc in TEST_CASES:
        all_rows.extend(run_tc_stats(tc))

    # Write raw per-run CSV
    with open(STATS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tc_id", "name", "run_id",
                                               "approximation_ratio", "qaoa_expected_cut",
                                               "bf_cut_value", "runtime_seconds"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Raw stats CSV: {STATS_CSV}")

    # Aggregate summary
    summary = []
    for tc in TEST_CASES:
        tc_rows = [r for r in all_rows if r["tc_id"] == tc.tc_id]
        if not tc_rows:
            continue
        ratios = [r["approximation_ratio"] for r in tc_rows]
        summary.append({
            "tc_id": tc.tc_id,
            "name": tc.name,
            "n_runs": len(ratios),
            "mean_r": round(statistics.mean(ratios), 4),
            "std_r": round(statistics.stdev(ratios) if len(ratios) > 1 else 0.0, 4),
            "min_r": round(min(ratios), 4),
            "max_r": round(max(ratios), 4),
        })

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tc_id", "name", "n_runs",
                                               "mean_r", "std_r", "min_r", "max_r"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"  Summary CSV:  {SUMMARY_CSV}")

    print("\n統計摘要：")
    print(f"{'ID':<5} {'名稱':<18} {'mean_r':>8} {'std_r':>7} {'min_r':>7} {'max_r':>7}")
    print("-" * 55)
    for s in summary:
        print(f"{s['tc_id']:<5} {s['name']:<18} {s['mean_r']:>8.4f} "
              f"{s['std_r']:>7.4f} {s['min_r']:>7.4f} {s['max_r']:>7.4f}")


if __name__ == "__main__":
    main()
