import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import math
import time
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches

# ── Chinese font: register Noto Sans CJK once ─────────────────────────────────
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = "Noto Sans CJK TC"
plt.rcParams["axes.unicode_minus"] = False

from matplotlib.font_manager import FontProperties

from agents.builder_agent import parse_graph_from_text
from agents.checker import validate
from quantum.translator import build_qaoa_executable
from quantum.simulator import optimize_qaoa, compute_cut_value
from quantum.param_advisor import suggest_qaoa_params
from classical.brute_force import solve_maxcut
from models.schemas import GraphProblem

# ── Chinese font property (for NetworkX manual label drawing) ─────────────────
_CN_FONT = FontProperties(fname=_FONT_PATH) if os.path.exists(_FONT_PATH) else FontProperties()


def draw_nx_with_cn_labels(
    G, pos, ax, node_labels: dict,
    node_color, node_size=900, font_size=9, font_color="white",
    edge_color="#888888", edge_widths=None,
):
    """Draw a NetworkX graph with Chinese labels rendered via ax.text()."""
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color,
                           width=edge_widths or 2)
    # Draw node labels manually so Chinese font is respected
    for node_id, (x, y) in pos.items():
        ax.text(x, y, node_labels.get(node_id, str(node_id)),
                ha="center", va="center", fontsize=font_size,
                color=font_color, fontproperties=_CN_FONT)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AutoQuantumPipeline", page_icon="⚛️", layout="wide")

st.title("⚛️ AutoQuantumPipeline")
st.caption("基於多代理語言模型之量子優化自動化與視覺化檢驗管線")

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline animation helper
# ══════════════════════════════════════════════════════════════════════════════
_STAGES = [
    ("📝", "自然語言\n輸入"),
    ("🤖", "Agent A\nNL 解析"),
    ("🔍", "Checker\n驗證"),
    ("⚡", "量子電路\n生成"),
    ("🔬", "QAOA\n模擬"),
    ("📊", "結果\n對比"),
]

def _pipeline_html(active: int, done_up_to: int, error: bool = False) -> str:
    """Render pipeline as animated HTML. active=current stage index (0-based)."""
    stage_cards = []
    for i, (icon, label) in enumerate(_STAGES):
        if i < done_up_to:
            state = "done"
            display_icon = "✅"
        elif i == active and error:
            state = "error"
            display_icon = "❌"
        elif i == active:
            state = "active"
            display_icon = icon
        else:
            state = "pending"
            display_icon = icon

        label_html = label.replace("\n", "<br>")
        stage_cards.append(f"""
        <div class="stage-wrap">
          <div class="stage-circle {state}">{display_icon}</div>
          <div class="stage-label {state}">{label_html}</div>
        </div>""")
        if i < len(_STAGES) - 1:
            arrow_cls = "done" if i < done_up_to else "pending"
            stage_cards.append(f'<div class="arrow {arrow_cls}">→</div>')

    return f"""
<style>
  .pipeline {{
    display:flex; align-items:center; justify-content:center;
    gap:4px; padding:18px 10px; background:#111827;
    border-radius:14px; margin:14px 0; flex-wrap:wrap;
  }}
  .stage-wrap {{
    display:flex; flex-direction:column; align-items:center; gap:6px; min-width:90px;
  }}
  .stage-circle {{
    width:58px; height:58px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:24px; border:3px solid #374151; background:#1f2937;
    transition: all .3s ease;
  }}
  .stage-circle.active {{
    border-color:#60a5fa; background:#1e3a5f;
    animation: pulse 1.2s ease-in-out infinite;
  }}
  .stage-circle.done  {{ border-color:#34d399; background:#064e3b; }}
  .stage-circle.error {{ border-color:#f87171; background:#450a0a; }}
  .stage-label {{
    font-size:11px; color:#6b7280; text-align:center; line-height:1.4;
  }}
  .stage-label.active {{ color:#93c5fd; font-weight:600; }}
  .stage-label.done   {{ color:#6ee7b7; }}
  .stage-label.error  {{ color:#fca5a5; }}
  .arrow {{ font-size:20px; color:#374151; padding:0 2px; margin-bottom:20px; }}
  .arrow.done {{ color:#34d399; }}
  @keyframes pulse {{
    0%,100% {{ box-shadow: 0 0 6px #60a5fa88; }}
    50%      {{ box-shadow: 0 0 20px #60a5faee; }}
  }}
</style>
<div class="pipeline">{''.join(stage_cards)}</div>
"""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 API 設定")
    _ui_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="前往 https://aistudio.google.com 免費取得。金鑰僅在本次 session 中使用，不會儲存。",
    )
    # Fallback: if UI is empty, try environment variable (.env)
    _effective_key: str = _ui_key.strip() or os.getenv("GOOGLE_API_KEY", "")
    if _effective_key:
        st.success("✅ API Key 已設定", icon="🔓")
    else:
        st.warning("請輸入 Google Gemini API Key 以執行管線", icon="⚠️")
    st.divider()
    st.info("⚙️ **QAOA 層數（p）與量測次數（shots）**\n\nAI 將根據問題規模（節點數、圖密度）自動決定最佳參數。")
    st.divider()
    st.markdown("**範例問題：**")
    example_text = (
        "我們公司有業務、工程、財務、人資、行銷、法務六個部門，準備分配到兩個樓層。"
        "請幫我找出最佳分樓方案，讓跨樓層協作最密集。"
        "過去三個月的跨部門會議次數如下：業務與行銷開了 8 次會、業務與法務開了 5 次會、"
        "工程與財務開了 7 次會、工程與人資開了 3 次會、"
        "財務與行銷開了 6 次會、人資與法務開了 4 次會、行銷與工程開了 2 次會。"
    )
    if st.button("載入範例", use_container_width=True):
        st.session_state["user_input"] = example_text

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("📝 輸入問題")
user_input = st.text_area(
    "請輸入問題描述（中文或英文皆可）",
    value=st.session_state.get("user_input", ""),
    height=140,
    placeholder="例：將六個部門分配到兩棟樓，使跨樓層協作最密集...",
)
run_btn = st.button("🚀 執行量子管線", type="primary", use_container_width=True)

if not run_btn:
    st.stop()
if not user_input.strip():
    st.error("請先輸入問題描述！")
    st.stop()
if not _effective_key:
    st.error("請在左側邊欄輸入 Google Gemini API Key，或在 .env 檔案中設定 GOOGLE_API_KEY。")
    st.info("🔗 前往 https://aistudio.google.com 免費取得 API Key")
    st.stop()

# Pipeline display placeholder — sits just below the button, updates throughout
pipeline_ph = st.empty()
pipeline_ph.markdown(_pipeline_html(active=0, done_up_to=0), unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: Agent A — NL parsing
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🤖 階段一：Agent A — 自然語言解析")
pipeline_ph.markdown(_pipeline_html(active=1, done_up_to=1), unsafe_allow_html=True)

with st.spinner("Agent A 正在解析問題結構..."):
    try:
        problem, _attempts = parse_graph_from_text(user_input, api_key=_effective_key)
    except RuntimeError as e:
        pipeline_ph.markdown(_pipeline_html(active=1, done_up_to=1, error=True), unsafe_allow_html=True)
        st.error(f"Agent A 解析失敗：{e}")
        st.stop()

pipeline_ph.markdown(_pipeline_html(active=2, done_up_to=2), unsafe_allow_html=True)

col_info, col_graph = st.columns([1, 2])
with col_info:
    st.success(f"解析完成！識別 **{problem.n_nodes}** 個節點、**{len(problem.edges)}** 條邊")
    st.markdown("**節點清單：**")
    for idx, label in enumerate(problem.node_labels):
        st.markdown(f"- 節點 {idx}：**{label}**")
    st.markdown("**邊清單：**")
    for edge in problem.edges[:8]:
        st.markdown(f"- {problem.node_labels[edge.node_i]} ↔ {problem.node_labels[edge.node_j]}：{edge.weight}")
    if len(problem.edges) > 8:
        st.markdown(f"_（共 {len(problem.edges)} 條）_")

with col_graph:
    G = nx.Graph()
    G.add_nodes_from(range(problem.n_nodes))
    for e in problem.edges:
        G.add_edge(e.node_i, e.node_j, weight=e.weight)

    fig, ax = plt.subplots(figsize=(6, 4))
    pos = nx.spring_layout(G, seed=42)
    node_labels = {i: problem.node_labels[i] for i in range(problem.n_nodes)}
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    ew = [2 + 3 * w / max_w for w in edge_weights]
    draw_nx_with_cn_labels(G, pos, ax, node_labels,
                           node_color="#4C9BE8", node_size=900,
                           font_size=9, font_color="white",
                           edge_color="#888888", edge_widths=ew)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={(e.node_i, e.node_j): e.weight for e in problem.edges},
        ax=ax, font_size=8,
    )
    ax.set_title("問題圖論結構", fontsize=12, fontproperties=_CN_FONT)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

# ── AI parameter recommendation (based on parsed problem) ─────────────────────
p_layers, shots, param_rationale = suggest_qaoa_params(problem)
with st.expander("⚙️ AI 自動決定 QAOA 參數", expanded=True):
    col_p, col_s = st.columns(2)
    with col_p:
        st.metric("QAOA 層數 (p)", p_layers)
    with col_s:
        st.metric("量測次數 (shots)", shots)
    st.markdown(param_rationale)

# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: Checker
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🔍 階段二：Agent B (Checker) — 自動驗證")
pipeline_ph.markdown(_pipeline_html(active=2, done_up_to=2), unsafe_allow_html=True)

validation = validate(problem)

log_placeholder = st.empty()
displayed = ""
log_lines = ["[Checker] 啟動驗證程序...", ""]
for check in validation.checks:
    if check.is_warning:
        icon = "⚠"
    elif check.passed:
        icon = "✓"
    else:
        icon = "✗"
    log_lines += [f"  [{icon}] {check.name}", f"       → {check.message}"]

has_warnings = any(c.is_warning for c in validation.checks)
if validation.passed and has_warnings:
    summary_line = "[Checker] ⚠  關鍵檢驗通過，但偵測到警告項目（見上方 ⚠ 標記）。管線繼續執行。"
elif validation.passed:
    summary_line = "[Checker] ✅ 所有檢驗通過，放行至量子轉換層。"
else:
    summary_line = "[Checker] ❌ 驗證未通過，管線中斷。"
log_lines += ["", summary_line]

for line in log_lines:
    displayed += line + "\n"
    log_placeholder.code(displayed, language=None)
    time.sleep(0.06)

if not validation.passed:
    pipeline_ph.markdown(_pipeline_html(active=2, done_up_to=2, error=True), unsafe_allow_html=True)
    st.error("Checker 發現問題，管線已中斷。")
    st.stop()

pipeline_ph.markdown(_pipeline_html(active=3, done_up_to=3), unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: Quantum circuit
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("⚡ 階段三：量子電路生成（Qamomile + Qiskit）")

with st.spinner(f"Qamomile SDK 正在生成 p={p_layers} 層 QAOA 電路..."):
    try:
        executable, converter, qiskit_transpiler = build_qaoa_executable(problem, p=p_layers)
        qiskit_circuit = executable.quantum_circuit
    except Exception as e:
        pipeline_ph.markdown(_pipeline_html(active=3, done_up_to=3, error=True), unsafe_allow_html=True)
        st.error(f"量子電路生成失敗：{e}")
        st.stop()

pipeline_ph.markdown(_pipeline_html(active=4, done_up_to=4), unsafe_allow_html=True)

col_circ, col_meta = st.columns([3, 1])
with col_meta:
    st.metric("量子位元數 (Qubits)", problem.n_nodes)
    st.metric("QAOA 層數 (p)", p_layers)
    st.metric("電路參數數量", 2 * p_layers)
    st.metric("電路深度", qiskit_circuit.depth())
    st.markdown("**可調參數：**")
    for i in range(p_layers):
        st.markdown(f"- `gamma[{i}]`")
    for i in range(p_layers):
        st.markdown(f"- `beta[{i}]`")

with col_circ:
    try:
        n_q = qiskit_circuit.num_qubits
        depth = qiskit_circuit.depth()

        # Wrap into multiple rows when circuit is deep to keep text legible
        FOLD_POINT = 25
        if depth <= FOLD_POINT:
            fold_val = -1
            n_rows = 1
        else:
            fold_val = FOLD_POINT
            n_rows = math.ceil(depth / FOLD_POINT)

        effective_cols = min(depth, FOLD_POINT) if n_rows > 1 else depth
        fig_w = max(12, effective_cols * 0.6)
        fig_h = max(4.0, (n_q * 0.7 + 1.2) * n_rows)

        fig_c = qiskit_circuit.draw("mpl", fold=fold_val, style="iqp", scale=0.8)
        fig_c.set_size_inches(fig_w, fig_h)
        fig_c.tight_layout()
        st.pyplot(fig_c)
        plt.close(fig_c)
    except Exception as circ_err:
        st.warning(f"matplotlib 電路圖渲染失敗（{circ_err}），改用文字版：")
        st.code(qiskit_circuit.draw("text", fold=80).__str__())

# ── Panel A: Hamiltonian table ────────────────────────────────────────────────
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def _sub(n: int) -> str:
    return str(n).translate(_SUB)

with st.expander("🔬 驗證一：哈密頓量（Hamiltonian）對應表", expanded=True):
    st.markdown(
        "QAOA 的代價函數是從問題圖直接翻譯而來的哈密頓量。"
        "下表列出每條邊對應的 Pauli 算符項：若**表格行數 = 邊數**，"
        "代表哈密頓量完整且正確地編碼了問題的圖結構。"
    )
    try:
        hamiltonian = converter.get_cost_hamiltonian()
        edge_lookup: dict[tuple[int, int], float] = {
            (min(e.node_i, e.node_j), max(e.node_i, e.node_j)): e.weight
            for e in problem.edges
        }
        rows = []
        for pauli_ops, coeff in hamiltonian.terms.items():
            if len(pauli_ops) == 2:
                i, j = pauli_ops[0].index, pauli_ops[1].index
                key = (min(i, j), max(i, j))
                edge_str = (
                    f"{problem.node_labels[key[0]]} ↔ {problem.node_labels[key[1]]}"
                    if key[0] < problem.n_nodes and key[1] < problem.n_nodes
                    else f"q{key[0]} ↔ q{key[1]}"
                )
                rows.append({
                    "Pauli 項": f"Z{_sub(i)} ⊗ Z{_sub(j)}",
                    "對應的邊": edge_str,
                    "係數": round(float(coeff.real), 4),
                })
            elif len(pauli_ops) == 1:
                i = pauli_ops[0].index
                rows.append({
                    "Pauli 項": f"Z{_sub(i)}",
                    "對應的邊": f"（節點 {i} 的線性項）",
                    "係數": round(float(coeff.real), 4),
                })
        if rows:
            st.dataframe(rows, use_container_width=True)
            n_zz = sum(1 for r in rows if "⊗" in r["Pauli 項"])
            st.caption(
                f"共 {len(rows)} 個 Pauli 項，其中 {n_zz} 個 ZZ 項（對應 {len(problem.edges)} 條邊）"
                + ("  ✅ 數量相符" if n_zz == len(problem.edges) else "  ⚠️ 數量不符，請檢查")
            )
        else:
            st.info("哈密頓量為空（無 Pauli 項）")
    except Exception as h_err:
        st.warning(f"無法取得哈密頓量：{h_err}")

# ═════════════════════════════════════════════════════════════════════════════
# Stage 4: Simulation + comparison
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📊 階段四：雙軌驗證與結果對比")

col_classical, col_quantum = st.columns(2)

with col_classical:
    st.markdown("#### 🖥️ 暴力枚舉（傳統演算法）")
    with st.spinner("計算暴力解..."):
        bf_result = solve_maxcut(problem)
    st.success(f"最大割值：**{bf_result.best_cut_value:.1f}**")
    st.markdown(f"**A 棟：** {', '.join(bf_result.group_a())}")
    st.markdown(f"**B 棟：** {', '.join(bf_result.group_b())}")

    colors = ["#E8624C" if bf_result.best_partition[i] == 0 else "#4CE8A0"
              for i in range(problem.n_nodes)]
    fig_bf, ax_bf = plt.subplots(figsize=(5, 3.5))
    draw_nx_with_cn_labels(G, pos, ax_bf, node_labels,
                           node_color=colors, node_size=900,
                           font_size=9, font_color="white",
                           edge_color="#888888",
                           edge_widths=[2 + 3 * w / max_w for w in edge_weights])
    patch_a = mpatches.Patch(color="#E8624C", label="A 棟")
    patch_b = mpatches.Patch(color="#4CE8A0", label="B 棟")
    ax_bf.legend(handles=[patch_a, patch_b], loc="upper right",
                 prop=_CN_FONT)
    ax_bf.set_title(f"最佳分割（割值 = {bf_result.best_cut_value:.1f}）",
                    fontsize=11, fontproperties=_CN_FONT)
    ax_bf.axis("off")
    st.pyplot(fig_bf)
    plt.close(fig_bf)

with col_quantum:
    st.markdown("#### ⚛️ QAOA 量子模擬器")
    with st.spinner(f"執行 QAOA 優化（{shots} shots × 最多 200 次迭代）..."):
        try:
            sim_result = optimize_qaoa(
                executable, converter, qiskit_transpiler,
                problem, p=p_layers, shots=shots,
            )
        except Exception as e:
            pipeline_ph.markdown(_pipeline_html(active=4, done_up_to=4, error=True), unsafe_allow_html=True)
            st.error(f"量子模擬失敗：{e}")
            st.stop()

    n = problem.n_nodes
    bits = [int(sim_result.best_bitstring[n - 1 - i]) for i in range(n)]
    quantum_group_a = [problem.node_labels[i] for i in range(n) if bits[i] == 0]
    quantum_group_b = [problem.node_labels[i] for i in range(n) if bits[i] == 1]

    match_icon = "✅" if abs(sim_result.best_cut_value - bf_result.best_cut_value) < 0.5 else "⚠️"
    st.info(f"{match_icon} 最大割值：**{sim_result.best_cut_value:.1f}** （暴力解：{bf_result.best_cut_value:.1f}）")
    st.markdown(f"**最佳 Bitstring：** `{sim_result.best_bitstring}`")
    st.markdown(f"**A 棟：** {', '.join(quantum_group_a) or '—'}")
    st.markdown(f"**B 棟：** {', '.join(quantum_group_b) or '—'}")

    # Detect complement solution (same cut, labels swapped — mathematically identical)
    bf_bits = bf_result.best_partition
    is_complement = all(bits[i] != bf_bits[i] for i in range(n))
    if is_complement or bits == bf_bits:
        st.success("✅ QAOA 找到與暴力解等價的最佳解！\n\n"
                   "（Max-Cut 具對稱性：A/B 棟對調後割值完全相同，為同一個切割方案）")

    # ── Panel C: Convergence curve ──────────────────────────────────────────
    if sim_result.convergence_history:
        with st.expander("📈 驗證二：QAOA 優化收斂曲線", expanded=True):
            st.markdown(
                "QAOA 利用傳統優化器（COBYLA）逐步調整量子電路參數，"
                "使期望割值越來越接近最優解。"
                "若折線整體呈上升趨勢，代表量子電路**確實在學習**，並非隨機輸出。"
            )
            fig_conv, ax_conv = plt.subplots(figsize=(6, 2.8))
            ax_conv.plot(
                sim_result.convergence_history,
                color="#60a5fa", linewidth=1.5, label="每輪期望割值",
            )
            ax_conv.axhline(
                bf_result.best_cut_value,
                color="#f87171", linestyle="--", linewidth=1.2,
                label=f"暴力最優解 ({bf_result.best_cut_value:.1f})",
            )
            ax_conv.set_xlabel("迭代次數", fontproperties=_CN_FONT)
            ax_conv.set_ylabel("期望割值", fontproperties=_CN_FONT)
            ax_conv.set_title("QAOA 優化收斂曲線", fontsize=11, fontproperties=_CN_FONT)
            ax_conv.legend(prop=_CN_FONT, fontsize=8)
            ax_conv.set_ylim(bottom=0)
            fig_conv.tight_layout()
            st.pyplot(fig_conv)
            plt.close(fig_conv)

    total_shots = sum(sim_result.counts.values())
    sorted_counts = sorted(sim_result.counts.items(), key=lambda x: x[1], reverse=True)[:16]
    labels_hist = [bs for bs, _ in sorted_counts]
    probs = [c / total_shots for _, c in sorted_counts]
    # Mark ALL bitstrings that achieve the optimal cut value in red
    optimal_bitstrings = {
        bs for bs in sim_result.counts
        if abs(compute_cut_value(
            {i: int(bs[n - 1 - i]) for i in range(n)}, problem
        ) - sim_result.best_cut_value) < 0.01
    }
    bar_colors = ["#FF6B6B" if bs in optimal_bitstrings else "#4C9BE8" for bs in labels_hist]

    fig_q, ax_q = plt.subplots(figsize=(5, 3.5))
    ax_q.bar(range(len(labels_hist)), probs, color=bar_colors)
    ax_q.set_xticks(range(len(labels_hist)))
    ax_q.set_xticklabels(labels_hist, rotation=45, ha="right", fontsize=7)
    ax_q.set_ylabel("機率", fontproperties=_CN_FONT)
    ax_q.set_title("QAOA 測量機率分佈（紅色 = 最佳解）", fontsize=10, fontproperties=_CN_FONT)
    ax_q.set_ylim(0, max(probs) * 1.2)
    st.pyplot(fig_q)
    plt.close(fig_q)

# ── Panel D: Energy landscape (n ≤ 10 only) ──────────────────────────────────
n = problem.n_nodes
if n <= 10:
    st.divider()
    with st.expander("🌄 驗證三：能量景觀圖（QAOA 機率 vs. 割值）", expanded=True):
        st.markdown(
            "下圖將所有 2ⁿ 種切割方案依割值分組，顯示 QAOA 在各割值上的**機率總和**（紅色欄 = 最大割值）。\n\n"
            "> ⚠️ **QAOA 是近似演算法**：p 層數越少，電路越淺，最優解的機率柱不一定最高。"
            "p=1 的 QAOA 有時只能找到「很好的近似解」，而非將所有機率集中在最優解。"
            "若想提升集中度，可在側欄調高 QAOA 層數（p）重新執行。"
        )
        total_shots_ld = sum(sim_result.counts.values())
        cut_to_prob: dict[float, float] = {}
        for bs_int in range(2 ** n):
            sample = {i: (bs_int >> i) & 1 for i in range(n)}
            bitstr = "".join(str((bs_int >> i) & 1) for i in reversed(range(n)))
            cv = compute_cut_value(sample, problem)
            prob = sim_result.counts.get(bitstr, 0) / total_shots_ld
            cut_to_prob[cv] = cut_to_prob.get(cv, 0) + prob

        optimal_prob = cut_to_prob.get(bf_result.best_cut_value, 0.0)
        random_baseline = 1.0 / len(cut_to_prob) if cut_to_prob else 0.0
        concentration_ratio = optimal_prob / random_baseline if random_baseline > 0 else 0

        sorted_cuts = sorted(cut_to_prob.keys())
        probs_ld = [cut_to_prob[c] for c in sorted_cuts]
        bar_colors_ld = [
            "#f87171" if c == bf_result.best_cut_value else "#60a5fa"
            for c in sorted_cuts
        ]

        fig_ld, ax_ld = plt.subplots(figsize=(8, 3.2))
        ax_ld.bar([str(int(c)) if c == int(c) else str(c) for c in sorted_cuts],
                  probs_ld, color=bar_colors_ld)
        ax_ld.axhline(random_baseline, color="#9ca3af", linestyle=":", linewidth=1,
                      label=f"均勻隨機基準線（{random_baseline:.3f}）")
        ax_ld.set_xlabel("割值（Cut Value）", fontproperties=_CN_FONT)
        ax_ld.set_ylabel("QAOA 機率質量", fontproperties=_CN_FONT)
        ax_ld.set_title(
            f"能量景觀：QAOA 機率分佈 vs. 所有可能割值（共 {2**n} 種切割方案）",
            fontsize=10, fontproperties=_CN_FONT,
        )
        import matplotlib.patches as _mp
        ax_ld.legend(
            handles=[
                _mp.Patch(color="#f87171", label=f"最大割值 = {bf_result.best_cut_value:.0f}（機率 {optimal_prob:.1%}）"),
                _mp.Patch(color="#60a5fa", label="其他割值"),
                plt.Line2D([0], [0], color="#9ca3af", linestyle=":", label=f"均勻隨機基準（{random_baseline:.3f}）"),
            ],
            prop=_CN_FONT, fontsize=8,
        )
        fig_ld.tight_layout()
        st.pyplot(fig_ld)
        plt.close(fig_ld)
        st.caption(
            f"📊 最優解集中度：QAOA 在最大割值上的機率 = **{optimal_prob:.1%}**，"
            f"為均勻隨機的 **{concentration_ratio:.1f} 倍**。"
            f"（若 = 1 倍表示與亂猜無異；倍數越高代表 QAOA 越有效）"
        )

# All done — show completed pipeline
pipeline_ph.markdown(_pipeline_html(active=5, done_up_to=6), unsafe_allow_html=True)
