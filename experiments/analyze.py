"""
experiments/analyze.py
Read results CSV/JSON and generate report figures (fig1-fig7).

Usage:
    cd /home/g8bear_/Coding_Place/Claude_AI/AutoQuantumPipeline
    source AQP/bin/activate
    python experiments/analyze.py
"""
import sys
import os
import csv
import json
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── Chinese font ───────────────────────────────────────────────────────────────
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = "Noto Sans CJK TC"
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.font_manager import FontProperties
_CN = FontProperties(fname=_FONT_PATH) if os.path.exists(_FONT_PATH) else FontProperties()

# Subscript character guard (U+2080-U+2089 → ASCII 0-9)
_SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def _clean(s: str) -> str:
    return s.translate(_SUBSCRIPT)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATH       = os.path.join(RESULTS_DIR, "results.csv")
JSON_PATH      = os.path.join(RESULTS_DIR, "results.json")
STATS_CSV      = os.path.join(RESULTS_DIR, "stats_summary.csv")
ABLATION_CSV   = os.path.join(RESULTS_DIR, "ablation_p.csv")
SCALABILITY_CSV = os.path.join(RESULTS_DIR, "scalability.csv")
ROBUSTNESS_CSV = os.path.join(RESULTS_DIR, "robustness.csv")

TC_COLORS = {
    "TC1": "#60a5fa",
    "TC2": "#34d399",
    "TC3": "#f59e0b",
    "TC4": "#a78bfa",
}

TC_NAMES_EN = {
    "TC1": "Star Graph",
    "TC2": "K3 Triangle\n(Frustration)",
    "TC3": "Weighted Graph\n(Data Centers)",
    "TC4": "Disconnected",
}


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Figure 1: Approximation Ratio Bar Chart ────────────────────────────────────
def fig_approximation_ratio(rows: list[dict]) -> None:
    valid = [r for r in rows if r["approximation_ratio"] not in ("", "N/A", "0.0", "0")]
    labels = [f"{r['tc_id']}\n{_clean(r['name'])}" for r in valid]
    ratios = [float(r["approximation_ratio"]) for r in valid]
    colors = [TC_COLORS.get(r["tc_id"], "#94a3b8") for r in valid]

    # Classical random baseline per TC (if column exists)
    rand_ratios = []
    for r in valid:
        v = r.get("random_baseline_ratio", "")
        if v not in ("", "N/A"):
            rand_ratios.append(float(v))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, ratios, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(1.0, color="#f87171", linestyle="--", linewidth=1.2, label="完美逼近 r=1.0")

    if rand_ratios:
        avg_rand = statistics.mean(rand_ratios)
        ax.axhline(avg_rand, color="#94a3b8", linestyle=":", linewidth=1.2,
                   label=f"隨機分組基準 r≈{avg_rand:.2f}")

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{ratio:.3f}", ha="center", va="bottom", fontsize=9, fontproperties=_CN)

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("量子期望逼近比 r = E[C_QAOA] / C_max", fontproperties=_CN)
    ax.set_title("各測試案例的 QAOA 量子期望逼近比", fontsize=13, fontproperties=_CN)
    ax.legend(prop=_CN)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    _save(fig, "fig1_approximation_ratio.png", "[圖1]")


# ── Figure 2: Convergence Curves ──────────────────────────────────────────────
def fig_convergence(json_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for entry in json_rows:
        hist = entry.get("convergence_history", [])
        if not hist:
            continue
        tc_id = entry["tc_id"]
        name = _clean(entry["name"])
        bf = float(entry.get("bf_cut_value", 0) or 0)
        color = TC_COLORS.get(tc_id, "#94a3b8")
        ax.plot(hist, color=color, linewidth=1.5, label=f"{tc_id} {name}")
        if bf > 0:
            ax.axhline(bf, color=color, linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("迭代次數", fontproperties=_CN)
    ax.set_ylabel("期望割值", fontproperties=_CN)
    ax.set_title("QAOA 優化收斂曲線（所有測試案例）", fontsize=13, fontproperties=_CN)
    ax.legend(prop=_CN, fontsize=8, loc="lower right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, "fig2_convergence_curves.png", "[圖2]")


# ── Figure 3: Resource Overhead Matrix ────────────────────────────────────────
def fig_resource_overhead(rows: list[dict]) -> None:
    valid = [r for r in rows if r["circuit_depth"] not in ("", "N/A")]
    # Use only TC id on x-axis to prevent overflow
    short_labels = [r["tc_id"] for r in valid]
    full_names = "  |  ".join(_clean(r["name"]) for r in valid)
    depths  = [int(r["circuit_depth"]) for r in valid]
    gates   = [int(r["circuit_gates"]) for r in valid]
    ps      = [int(r["p_chosen"]) for r in valid]
    shots_l = [int(r["shots_chosen"]) for r in valid]
    colors  = [TC_COLORS.get(r["tc_id"], "#94a3b8") for r in valid]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    datasets = [
        (depths,  "電路深度（Circuit Depth）"),
        (gates,   "總閘數（Gate Count）"),
        (ps,      "QAOA 層數（p）"),
        (shots_l, "量測次數（shots）"),
    ]
    for ax, (vals, title) in zip(axes, datasets):
        bars = ax.bar(short_labels, vals, color=colors, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    str(v), ha="center", va="bottom", fontsize=9, fontproperties=_CN)
        ax.set_title(title, fontsize=9, fontproperties=_CN)
        ax.set_ylim(0, max(vals) * 1.25)

    fig.suptitle("量子電路資源開銷矩陣", fontsize=13, fontproperties=_CN)
    # Full TC names as a caption below
    fig.text(0.5, 0.01, full_names, ha="center", fontsize=7.5, color="#555555",
             fontproperties=_CN)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "fig3_resource_overhead.png", "[圖3]")


# ── Figure 4: Stats — mean ± std error bars ────────────────────────────────────
def fig_stats_errorbars(rows: list[dict]) -> None:
    if not os.path.exists(STATS_CSV):
        print("  [圖4] 跳過（stats_summary.csv 尚未生成）")
        return

    stats = {r["tc_id"]: r for r in _load(STATS_CSV)}
    valid = [r for r in rows if r["tc_id"] in stats]
    if not valid:
        print("  [圖4] 無資料，跳過")
        return

    labels  = [f"{r['tc_id']}\n{_clean(r['name'])}" for r in valid]
    means   = [float(stats[r["tc_id"]]["mean_r"]) for r in valid]
    stds    = [float(stats[r["tc_id"]]["std_r"])  for r in valid]
    colors  = [TC_COLORS.get(r["tc_id"], "#94a3b8") for r in valid]

    rand_ratios = [float(r["random_baseline_ratio"]) for r in rows
                   if r.get("random_baseline_ratio", "") not in ("", "N/A")]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.8,
                  error_kw={"elinewidth": 1.5, "ecolor": "#374151"})
    ax.axhline(1.0, color="#f87171", linestyle="--", linewidth=1.2, label="完美逼近 r=1.0")
    if rand_ratios:
        avg_rand = statistics.mean(rand_ratios)
        ax.axhline(avg_rand, color="#94a3b8", linestyle=":", linewidth=1.2,
                   label=f"隨機基準 r≈{avg_rand:.2f}")

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.01,
                f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=8, fontproperties=_CN)

    ax.set_ylim(0, 1.25)
    ax.set_ylabel("量子期望逼近比 r（mean ± std，5 次重複）", fontproperties=_CN)
    ax.set_title("QAOA 逼近比統計信度分析（5 次獨立執行）", fontsize=13, fontproperties=_CN)
    ax.legend(prop=_CN)
    fig.tight_layout()
    _save(fig, "fig4_stats_errorbars.png", "[圖4]")


# ── Figure 5: p Ablation — r(p) line chart ────────────────────────────────────
def fig_ablation_p() -> None:
    if not os.path.exists(ABLATION_CSV):
        print("  [圖5] 跳過（ablation_p.csv 尚未生成）")
        return

    rows = _load(ABLATION_CSV)
    ps   = [int(r["p"]) for r in rows]
    # support both old (approximation_ratio) and new (mean_r) column names
    rs   = [float(r.get("mean_r", r.get("approximation_ratio", 0))) for r in rows]
    errs = [float(r["std_r"]) if "std_r" in r else 0.0 for r in rows]
    depths = [int(r["circuit_depth"]) for r in rows]
    gates  = [int(r["circuit_gates"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.errorbar(ps, rs, yerr=errs if any(e > 0 for e in errs) else None,
                 fmt="o-", color="#f59e0b", linewidth=2, markersize=7, capsize=4,
                 label="逼近比 r（mean±std）")
    ax1.axhline(1.0, color="#f87171", linestyle="--", linewidth=1, label="r=1.0")
    ax2.bar([p - 0.15 for p in ps], depths, width=0.25, color="#60a5fa", alpha=0.5, label="電路深度")
    ax2.bar([p + 0.15 for p in ps], gates,  width=0.25, color="#34d399", alpha=0.5, label="總閘數")

    for p, r in zip(ps, rs):
        ax1.text(p, r + 0.015, f"{r:.3f}", ha="center", fontsize=9, color="#92400e",
                 fontproperties=_CN)

    ax1.set_xlabel("QAOA 層數 p", fontproperties=_CN)
    ax1.set_ylabel("量子期望逼近比 r", fontproperties=_CN)
    ax2.set_ylabel("電路資源", fontproperties=_CN)
    ax1.set_title("QAOA 深度消融分析（TC3 帶權重圖）", fontsize=13, fontproperties=_CN)
    ax1.set_ylim(0, 1.15)
    ax1.set_xticks(ps)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, prop=_CN, fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig5_ablation_p.png", "[圖5]")


# ── Figure 6: Scalability — depth & runtime vs n ──────────────────────────────
def fig_scalability() -> None:
    if not os.path.exists(SCALABILITY_CSV):
        print("  [圖6] 跳過（scalability.csv 尚未生成）")
        return

    rows = _load(SCALABILITY_CSV)
    ns      = [int(r["n_nodes"]) for r in rows]
    depths  = [int(r["circuit_depth"]) for r in rows]
    runtimes = [float(r["runtime_seconds"]) for r in rows]
    ratios  = [float(r["approximation_ratio"]) if r.get("approximation_ratio", "") not in ("", "N/A") else None for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(ns, depths, "o-", color="#60a5fa", linewidth=2, markersize=7)
    for n, d in zip(ns, depths):
        ax1.text(n, d + 0.3, str(d), ha="center", fontsize=9, fontproperties=_CN)
    ax1.set_xlabel("節點數 n", fontproperties=_CN)
    ax1.set_ylabel("電路深度", fontproperties=_CN)
    ax1.set_title("電路深度隨規模成長", fontsize=11, fontproperties=_CN)
    ax1.set_xticks(ns)

    ax2.plot(ns, runtimes, "s-", color="#f59e0b", linewidth=2, markersize=7)
    for n, t in zip(ns, runtimes):
        ax2.text(n, t + 0.3, f"{t:.1f}s", ha="center", fontsize=9, fontproperties=_CN)
    ax2.set_xlabel("節點數 n", fontproperties=_CN)
    ax2.set_ylabel("執行時間（秒）", fontproperties=_CN)
    ax2.set_title("執行時間隨規模成長", fontsize=11, fontproperties=_CN)
    ax2.set_xticks(ns)

    fig.suptitle("QAOA 規模擴展性分析（隨機連通圖，edge_prob=0.5）",
                 fontsize=13, fontproperties=_CN)
    fig.tight_layout()
    _save(fig, "fig6_scalability.png", "[圖6]")


# ── Figure 7: LLM Robustness ─────────────────────────────────────────────────
def fig_robustness() -> None:
    if not os.path.exists(ROBUSTNESS_CSV):
        print("  [圖7] 跳過（robustness.csv 尚未生成）")
        return

    rows = _load(ROBUSTNESS_CSV)
    phrasings = sorted(set(r["phrasing_id"] for r in rows))
    metrics = ["n_nodes_ok", "n_edges_ok", "weights_match"]
    metric_labels = ["節點數正確", "邊數正確", "權重完全正確"]

    # Compute accuracy (success rate) per phrasing × metric
    data = {p: {m: [] for m in metrics} for p in phrasings}
    for r in rows:
        for m in metrics:
            data[r["phrasing_id"]][m].append(1 if r[m].lower() in ("true", "1", "yes") else 0)
    accs = {p: {m: sum(v) / len(v) if v else 0 for m, v in mdict.items()}
            for p, mdict in data.items()}

    x = np.arange(len(phrasings))
    width = 0.25
    colors_metric = ["#60a5fa", "#34d399", "#f59e0b"]
    phrasing_names = {"P1": "P1 原版措辭", "P2": "P2 改名措辭", "P3": "P3 口語措辭"}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (m, label, color) in enumerate(zip(metrics, metric_labels, colors_metric)):
        vals = [accs[p][m] for p in phrasings]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color,
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.0%}", ha="center", va="bottom", fontsize=8.5, fontproperties=_CN)

    ax.set_xticks(x)
    ax.set_xticklabels([phrasing_names.get(p, p) for p in phrasings], fontproperties=_CN)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("正確率（3 次重複）", fontproperties=_CN)
    ax.set_title("Agent A 語意魯棒性分析（TC3 三種措辭）", fontsize=13, fontproperties=_CN)
    ax.legend(prop=_CN)
    fig.tight_layout()
    _save(fig, "fig7_robustness.png", "[圖7]")


def _save(fig, filename: str, tag: str) -> None:
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {tag} 儲存至 {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Report-specific composite figures (compact, < 1/4 page each in LaTeX)
# ══════════════════════════════════════════════════════════════════════════════

def fig_report_1_quality(rows: list[dict]) -> None:
    """Report Fig 1: QAOA algorithm quality (left: mean±std; right: p-ablation)"""
    if not os.path.exists(STATS_CSV):
        print("  [Report Fig1] skipped (stats_summary.csv not found)")
        return

    stats = {r["tc_id"]: r for r in _load(STATS_CSV)}
    valid = [r for r in rows if r["tc_id"] in stats]
    if not valid:
        return

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 3.8))

    # ── Left: mean±std approximation ratio ───────────────────────────────────
    tc_labels = [f"{r['tc_id']}\n{TC_NAMES_EN.get(r['tc_id'], r['tc_id'])}" for r in valid]
    means  = [float(stats[r["tc_id"]]["mean_r"]) for r in valid]
    stds   = [float(stats[r["tc_id"]]["std_r"])  for r in valid]
    colors = [TC_COLORS.get(r["tc_id"], "#94a3b8") for r in valid]
    rand_ratios = [float(r["random_baseline_ratio"]) for r in rows
                   if r.get("random_baseline_ratio", "") not in ("", "N/A")]

    bars = ax_l.bar(tc_labels, means, yerr=stds, capsize=4,
                    color=colors, edgecolor="white", linewidth=0.8,
                    error_kw={"elinewidth": 1.5, "ecolor": "#374151"})
    ax_l.axhline(1.0, color="#f87171", linestyle="--", linewidth=1.1,
                 label="Perfect approx. r=1.0")
    if rand_ratios:
        ax_l.axhline(statistics.mean(rand_ratios), color="#94a3b8", linestyle=":",
                     linewidth=1.1, label=f"Random baseline ≈{statistics.mean(rand_ratios):.2f}")
    for bar, m, s in zip(bars, means, stds):
        ax_l.text(bar.get_x() + bar.get_width() / 2, m + s + 0.015,
                  f"{m:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax_l.set_ylim(0, 1.18)
    ax_l.set_ylabel("Approximation ratio r", fontsize=10)
    ax_l.set_title("(a) QAOA Approximation Ratio (mean±std, N=5)", fontsize=10)
    ax_l.legend(fontsize=7.5)
    ax_l.tick_params(axis="x", labelsize=7)

    # ── Right: p-ablation ─────────────────────────────────────────────────────
    if os.path.exists(ABLATION_CSV):
        ab_rows = _load(ABLATION_CSV)
        has_mean = "mean_r" in ab_rows[0] if ab_rows else False
        ps = [int(r["p"]) for r in ab_rows]

        if has_mean:
            rs   = [float(r["mean_r"]) for r in ab_rows]
            errs = [float(r["std_r"])  for r in ab_rows]
            ax_r.errorbar(ps, rs, yerr=errs, fmt="o-", color="#f59e0b",
                          linewidth=2, markersize=6, capsize=4, elinewidth=1.5,
                          label="mean±std")
        else:
            rs = [float(r["approximation_ratio"]) for r in ab_rows]
            ax_r.plot(ps, rs, "o-", color="#f59e0b", linewidth=2, markersize=6,
                      label="Single run")
        ax_r.axhline(1.0, color="#f87171", linestyle="--", linewidth=1.1)
        for p_val, r_val in zip(ps, rs):
            ax_r.text(p_val, r_val + 0.012, f"{r_val:.3f}", ha="center",
                      fontsize=8, color="#92400e")
        ax_r.set_xlabel("QAOA depth p", fontsize=10)
        ax_r.set_ylabel("Approx. ratio r (TC3)", fontsize=10)
        ax_r.set_title("(b) p-Ablation (TC3 Weighted Graph)", fontsize=10)
        ax_r.set_xticks(ps)
        ax_r.set_ylim(0.60, 1.08)
        ax_r.legend(fontsize=7.5)
    else:
        ax_r.text(0.5, 0.5, "ablation_p.csv not found", ha="center", va="center",
                  transform=ax_r.transAxes)

    fig.tight_layout()
    _save(fig, "fig_report_1_quality.png", "[Report Fig1]")


def fig_report_2_scalability() -> None:
    """Report Fig 2: Circuit scalability (left: depth+gates vs n; right: runtime vs n)"""
    if not os.path.exists(SCALABILITY_CSV):
        print("  [Report Fig2] skipped (scalability.csv not found)")
        return

    rows = _load(SCALABILITY_CSV)
    ns      = [int(r["n_nodes"]) for r in rows]
    depths  = [int(r["circuit_depth"]) for r in rows]
    runtimes = [float(r["runtime_seconds"]) for r in rows]
    gates   = [int(r["circuit_gates"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    # ── Left: circuit depth + gate count vs n ────────────────────────────────
    ax1b = ax1.twinx()
    l1, = ax1.plot(ns, depths, "o-", color="#60a5fa", linewidth=2, markersize=6,
                   label="Circuit Depth")
    l2, = ax1b.plot(ns, gates, "s--", color="#34d399", linewidth=1.5, markersize=5,
                    label="Gate Count")
    for n, d in zip(ns, depths):
        ax1.text(n, d + 0.4, str(d), ha="center", fontsize=8, color="#1d4ed8")
    for n, g in zip(ns, gates):
        ax1b.text(n + 0.08, g + 0.4, str(g), ha="left", fontsize=7.5, color="#065f46")
    ax1.set_xlabel("Graph size n", fontsize=10)
    ax1.set_ylabel("Circuit Depth", fontsize=10, color="#1d4ed8")
    ax1b.set_ylabel("Gate Count", fontsize=10, color="#065f46")
    ax1.set_title("(a) Circuit Resources vs. Graph Size", fontsize=10)
    ax1.set_xticks(ns)
    ax1.legend(handles=[l1, l2], fontsize=7.5, loc="upper left")

    # ── Right: runtime vs n ───────────────────────────────────────────────────
    ax2.plot(ns, runtimes, "s-", color="#f59e0b", linewidth=2, markersize=6)
    for n, t in zip(ns, runtimes):
        ax2.text(n, t + 0.25, f"{t:.1f}s", ha="center", fontsize=8, color="#92400e")
    ax2.set_xlabel("Graph size n", fontsize=10)
    ax2.set_ylabel("Runtime (s)", fontsize=10)
    ax2.set_title("(b) Runtime vs. Graph Size", fontsize=10)
    ax2.set_xticks(ns)
    ax2.set_ylim(0, max(runtimes) * 1.3)

    fig.tight_layout()
    _save(fig, "fig_report_2_scalability.png", "[Report Fig2]")


def fig_report_3_robustness() -> None:
    """Report Fig 3: Agent A semantic robustness (3 phrasings × 3 accuracy metrics)"""
    if not os.path.exists(ROBUSTNESS_CSV):
        print("  [Report Fig3] skipped (robustness.csv not found)")
        return

    rows = _load(ROBUSTNESS_CSV)
    phrasings = sorted(set(r["phrasing_id"] for r in rows))
    metrics = ["n_nodes_ok", "n_edges_ok", "weights_match"]
    metric_labels = ["Node Count Acc.", "Edge Count Acc.", "Weight Acc."]
    phrasing_names = {"P1": "P1\nOriginal", "P2": "P2\nRenamed", "P3": "P3\nColloquial"}

    data = {p: {m: [] for m in metrics} for p in phrasings}
    for r in rows:
        for m in metrics:
            data[r["phrasing_id"]][m].append(1 if r[m].lower() in ("true", "1", "yes") else 0)
    accs = {p: {m: sum(v) / len(v) if v else 0 for m, v in mdict.items()}
            for p, mdict in data.items()}

    x = np.arange(len(phrasings))
    width = 0.26
    colors_m = ["#60a5fa", "#34d399", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (m, label, color) in enumerate(zip(metrics, metric_labels, colors_m)):
        vals = [accs[p][m] for p in phrasings]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color,
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([phrasing_names.get(p, p) for p in phrasings])
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Accuracy (N=3 runs)", fontsize=10)
    ax.set_title("Agent A Semantic Robustness: Parsing Accuracy Across 3 Phrasings",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_report_3_robustness.png", "[Report Fig3]")


def main() -> None:
    if not os.path.exists(CSV_PATH):
        print(f"找不到 {CSV_PATH}，請先執行 experiments/run_all.py")
        sys.exit(1)

    rows = _load(CSV_PATH)
    json_rows = _load_json(JSON_PATH)

    print("\n生成探索性圖表（fig1~fig7）...")
    fig_approximation_ratio(rows)
    fig_convergence(json_rows)
    fig_resource_overhead(rows)
    fig_stats_errorbars(rows)
    fig_ablation_p()
    fig_scalability()
    fig_robustness()

    print("\nGenerating report composite figures (fig_report_1~3) ...")
    fig_report_1_quality(rows)
    fig_report_2_scalability()
    fig_report_3_robustness()

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
