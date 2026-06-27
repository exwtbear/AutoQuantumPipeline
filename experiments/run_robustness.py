"""
experiments/run_robustness.py
Test Agent A's semantic robustness: same TC3 graph (D1-D4, 5 weighted edges)
expressed in 3 different phrasings, each called 3 times independently.

Ground truth: n_nodes=4, n_edges=5, weights={1.0, 2.0, 5.0, 8.0, 10.0}

Usage:
    python experiments/run_robustness.py
"""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.builder_agent import parse_graph_from_text

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ROBUSTNESS_CSV = os.path.join(RESULTS_DIR, "robustness.csv")
N_RUNS_PER_PHRASING = 3

# Ground truth for TC3
GT_N_NODES = 4
GT_N_EDGES = 5
GT_WEIGHTS = frozenset([1.0, 2.0, 5.0, 8.0, 10.0])

PHRASINGS = {
    "P1": (
        "原版措辭（資料中心 D1-D4）",
        (
            "我們有四個資料中心 (D1, D2, D3, D4) 之間存在頻寬干擾。"
            "干擾程度如下：D1 和 D2 之間干擾值高達 10；"
            "D2 和 D3 干擾值為 8；D3 和 D4 干擾值為 2；"
            "D1 和 D4 干擾值為 5；D1 和 D3 之間有輕微干擾，數值為 1。"
            "請將這四個中心分流到兩個獨立網段，使被消除的干擾總值最大化。"
        ),
    ),
    "P2": (
        "改名措辭（伺服器 Alpha-Delta）",
        (
            "雲端平台上有四台伺服器：Alpha、Beta、Gamma、Delta。"
            "它們之間的資源搶佔強度如下："
            "Alpha 與 Beta 衝突指數為 10；Beta 與 Gamma 衝突指數為 8；"
            "Gamma 與 Delta 指數為 2；Alpha 與 Delta 指數為 5；"
            "Alpha 與 Gamma 指數為 1。"
            "請將這四台伺服器劃分成兩個群組，使跨群組的衝突指數總和最大化。"
        ),
    ),
    "P3": (
        "口語措辭（模組一~四）",
        (
            "小王管理一個系統，裡面有四個模組（模組一、模組二、模組三、模組四）。"
            "當兩個模組共用同一台機器時會互相干擾："
            "模組一和模組二干擾最嚴重，程度是 10；"
            "模組二和模組三干擾程度是 8；"
            "模組三和模組四只有輕微干擾，程度是 2；"
            "模組一和模組四中等干擾，程度是 5；"
            "模組一和模組三幾乎沒有干擾，程度是 1。"
            "要把這四個模組分配到兩台機器上，使兩台機器之間的干擾程度總和最大，你建議怎麼分？"
        ),
    ),
}


def check_weights(problem) -> bool:
    parsed = frozenset(round(e.weight, 1) for e in problem.edges)
    return parsed == GT_WEIGHTS


def main() -> None:
    all_rows = []

    for phrasing_id, (desc, nl_input) in PHRASINGS.items():
        print(f"\n[{phrasing_id}] {desc}")
        for run_id in range(1, N_RUNS_PER_PHRASING + 1):
            print(f"  Run {run_id}/{N_RUNS_PER_PHRASING} ...", end=" ", flush=True)
            try:
                problem, attempts = parse_graph_from_text(nl_input)
                n_ok   = problem.n_nodes == GT_N_NODES
                m_ok   = len(problem.edges) == GT_N_EDGES
                w_ok   = check_weights(problem)
                status = "✓" if (n_ok and m_ok and w_ok) else "✗"
                print(f"{status}  n={'✓' if n_ok else '✗'}  m={'✓' if m_ok else '✗'}  "
                      f"weights={'✓' if w_ok else '✗'}  attempts={attempts}")
                all_rows.append({
                    "phrasing_id": phrasing_id,
                    "description": desc,
                    "run_id": run_id,
                    "n_nodes_ok": n_ok,
                    "n_edges_ok": m_ok,
                    "weights_match": w_ok,
                    "agent_a_attempts": attempts,
                    "parsed_n_nodes": problem.n_nodes,
                    "parsed_n_edges": len(problem.edges),
                    "parsed_weights": str(sorted(round(e.weight, 1) for e in problem.edges)),
                    "error": "",
                })
            except Exception as e:
                print(f"✗ ERROR: {e}")
                all_rows.append({
                    "phrasing_id": phrasing_id,
                    "description": desc,
                    "run_id": run_id,
                    "n_nodes_ok": False,
                    "n_edges_ok": False,
                    "weights_match": False,
                    "agent_a_attempts": 3,
                    "parsed_n_nodes": "",
                    "parsed_n_edges": "",
                    "parsed_weights": "",
                    "error": str(e),
                })

    with open(ROBUSTNESS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  CSV: {ROBUSTNESS_CSV}")

    print("\n魯棒性摘要（各措辭正確率）：")
    print(f"{'ID':<5} {'描述':<22} {'節點':>6} {'邊':>5} {'權重':>6}")
    print("-" * 48)
    for pid in PHRASINGS:
        tc_rows = [r for r in all_rows if r["phrasing_id"] == pid]
        n = len(tc_rows)
        n_acc = sum(1 for r in tc_rows if r["n_nodes_ok"] is True) / n
        m_acc = sum(1 for r in tc_rows if r["n_edges_ok"] is True) / n
        w_acc = sum(1 for r in tc_rows if r["weights_match"] is True) / n
        print(f"{pid:<5} {PHRASINGS[pid][0]:<22} {n_acc:>5.0%} {m_acc:>5.0%} {w_acc:>6.0%}")


if __name__ == "__main__":
    main()
