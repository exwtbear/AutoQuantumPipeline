"""
experiments/run_all.py
Run all 4 test cases through the full pipeline and record metrics to CSV + JSON.

Usage:
    cd /home/g8bear_/Coding_Place/Claude_AI/AutoQuantumPipeline
    source AQP/bin/activate
    python experiments/run_all.py
"""
import sys
import os
import csv
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.builder_agent import parse_graph_from_text
from agents.checker import validate
from quantum.translator import build_qaoa_executable
from quantum.simulator import optimize_qaoa, compute_cut_value
from quantum.param_advisor import suggest_qaoa_params
from classical.brute_force import solve_maxcut
from experiments.test_cases import TEST_CASES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")
JSON_PATH = os.path.join(RESULTS_DIR, "results.json")

FIELDNAMES = [
    "tc_id", "name", "topology",
    "n_nodes", "n_edges",
    "agent_a_success", "agent_a_attempts",
    "checker_passed", "checker_has_warnings",
    "p_chosen", "shots_chosen",
    "circuit_depth", "circuit_gates",
    "bf_cut_value",
    "qaoa_best_cut", "qaoa_expected_cut",
    "approximation_ratio",
    "random_baseline_ratio",
    "runtime_seconds",
]


def run_one(tc) -> dict:
    print(f"\n{'='*60}")
    print(f"[{tc.tc_id}] {tc.name}  (topology: {tc.topology})")
    print(f"{'='*60}")
    row: dict = {
        "tc_id": tc.tc_id,
        "name": tc.name,
        "topology": tc.topology,
    }
    t_start = time.time()

    # ── Agent A ──────────────────────────────────────────────────────────────
    print("  → Agent A 解析中...")
    try:
        problem, attempts = parse_graph_from_text(tc.nl_input)
        row["agent_a_success"] = True
        row["agent_a_attempts"] = attempts
        row["n_nodes"] = problem.n_nodes
        row["n_edges"] = len(problem.edges)
        print(f"     ✓ 解析成功（第 {attempts} 次嘗試）"
              f"  n={problem.n_nodes}, m={len(problem.edges)}")
        print(f"     節點：{problem.node_labels}")
    except RuntimeError as e:
        print(f"     ✗ 解析失敗：{e}")
        row.update({
            "agent_a_success": False, "agent_a_attempts": 3,
            "n_nodes": tc.expected_n_nodes, "n_edges": "?",
            "checker_passed": False, "checker_has_warnings": False,
            "p_chosen": "N/A", "shots_chosen": "N/A",
            "circuit_depth": "N/A", "circuit_gates": "N/A",
            "bf_cut_value": tc.expected_max_cut,
            "qaoa_best_cut": "N/A", "qaoa_expected_cut": "N/A",
            "approximation_ratio": 0.0,
            "runtime_seconds": round(time.time() - t_start, 2),
        })
        return row

    # ── Agent B (Checker) ─────────────────────────────────────────────────────
    print("  → Checker 驗證中...")
    validation = validate(problem)
    has_warnings = any(c.is_warning for c in validation.checks)
    row["checker_passed"] = validation.passed
    row["checker_has_warnings"] = has_warnings

    for check in validation.checks:
        tag = "⚠" if check.is_warning else ("✓" if check.passed else "✗")
        print(f"     [{tag}] {check.name}: {check.message}")

    if tc.agent_b_should_warn and not has_warnings:
        print("     ⚠  警告：預期觸發 Agent B 警告，但本次未觸發！")

    if not validation.passed:
        print("     ✗ Checker 未通過，跳過量子模擬")
        row.update({
            "p_chosen": "N/A", "shots_chosen": "N/A",
            "circuit_depth": "N/A", "circuit_gates": "N/A",
            "bf_cut_value": tc.expected_max_cut,
            "qaoa_best_cut": "N/A", "qaoa_expected_cut": "N/A",
            "approximation_ratio": 0.0,
            "runtime_seconds": round(time.time() - t_start, 2),
        })
        return row

    # ── AI 參數推薦 ───────────────────────────────────────────────────────────
    p, shots, _ = suggest_qaoa_params(problem)
    row["p_chosen"] = p
    row["shots_chosen"] = shots
    print(f"  → AI 選參：p={p}, shots={shots}")

    # ── 量子電路生成 ──────────────────────────────────────────────────────────
    print("  → 生成 QAOA 電路...")
    executable, converter, transpiler = build_qaoa_executable(problem, p=p)
    qc = executable.quantum_circuit
    row["circuit_depth"] = qc.depth()
    row["circuit_gates"] = qc.size()
    print(f"     電路深度={qc.depth()}, 總閘數={qc.size()}")

    # ── 暴力解 ────────────────────────────────────────────────────────────────
    print("  → 暴力枚舉...")
    bf = solve_maxcut(problem)
    row["bf_cut_value"] = bf.best_cut_value
    # Classical random partition baseline: E[random_cut] = total_weight / 2
    total_weight = sum(e.weight for e in problem.edges)
    row["random_baseline_ratio"] = round((total_weight / 2) / bf.best_cut_value, 4) if bf.best_cut_value > 0 else 0.0
    print(f"     C_max = {bf.best_cut_value}  ({bf.group_a()} | {bf.group_b()})")

    # ── QAOA 模擬 ─────────────────────────────────────────────────────────────
    print(f"  → QAOA 模擬（{shots} shots × 最多 200 輪迭代）...")
    sim = optimize_qaoa(executable, converter, transpiler, problem, p=p, shots=shots)

    qaoa_expected = sim.convergence_history[-1] if sim.convergence_history else 0.0
    approx_ratio = qaoa_expected / bf.best_cut_value if bf.best_cut_value > 0 else 0.0

    row["qaoa_best_cut"] = sim.best_cut_value
    row["qaoa_expected_cut"] = round(qaoa_expected, 4)
    row["approximation_ratio"] = round(approx_ratio, 4)
    row["runtime_seconds"] = round(time.time() - t_start, 2)

    print(f"     QAOA 最佳觀測割值 = {sim.best_cut_value}")
    print(f"     QAOA 期望割值     = {qaoa_expected:.4f}")
    print(f"     逼近比 r          = {approx_ratio:.4f}")
    print(f"     收斂迭代次數       = {len(sim.convergence_history)}")

    # Attach full convergence history to JSON output
    row["_convergence_history"] = sim.convergence_history
    return row


def print_summary(results: list[dict]) -> None:
    print("\n" + "="*72)
    print("  實驗結果摘要")
    print("="*72)
    header = f"{'ID':<5} {'名稱':<18} {'n':>3} {'m':>3} {'p':>3} {'shots':>6} {'bf':>6} {'r':>6} {'深度':>6} {'時間':>7}"
    print(header)
    print("-"*72)
    for r in results:
        if r.get("approximation_ratio", 0) == 0.0 and not r.get("checker_passed", True):
            ratio_str = "  N/A"
        else:
            ratio_str = f"{r.get('approximation_ratio', 0):.4f}"
        shots_str = str(r.get("shots_chosen", "N/A"))
        depth_str = str(r.get("circuit_depth", "N/A"))
        print(
            f"{r['tc_id']:<5} {r['name']:<18} "
            f"{r.get('n_nodes', '?'):>3} {r.get('n_edges', '?'):>3} "
            f"{str(r.get('p_chosen', '-')):>3} {shots_str:>6} "
            f"{r.get('bf_cut_value', '?'):>6} {ratio_str:>6} "
            f"{depth_str:>6} {r.get('runtime_seconds', '?'):>6.1f}s"
        )
    print("="*72)

    success_count = sum(1 for r in results if r.get("agent_a_success") and r.get("checker_passed"))
    first_try = sum(1 for r in results if r.get("agent_a_attempts") == 1)
    warn_count = sum(1 for r in results if r.get("checker_has_warnings"))
    total = len(results)

    print(f"\n  管線成功率        : {success_count}/{total} = {success_count/total:.0%}")
    print(f"  Agent A 首次成功率 : {first_try}/{total} = {first_try/total:.0%}")
    print(f"  觸發警告案例      : {warn_count}/{total}")
    avg_ratio = sum(r["approximation_ratio"] for r in results if isinstance(r.get("approximation_ratio"), float) and r["approximation_ratio"] > 0)
    valid = sum(1 for r in results if isinstance(r.get("approximation_ratio"), float) and r["approximation_ratio"] > 0)
    if valid:
        print(f"  平均期望逼近比    : {avg_ratio/valid:.4f}")


def main() -> None:
    all_results = []
    all_json = []

    for tc in TEST_CASES:
        row = run_one(tc)
        all_results.append(row)

        # Separate convergence history for JSON (not in CSV)
        history = row.pop("_convergence_history", [])
        json_entry = dict(row)
        json_entry["convergence_history"] = history
        all_json.append(json_entry)

    # Write CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  CSV 已儲存：{CSV_PATH}")

    # Write JSON
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_json, f, ensure_ascii=False, indent=2)
    print(f"  JSON 已儲存：{JSON_PATH}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
