import math
import networkx as nx
from models.schemas import GraphProblem, CheckItem, ValidationResult


def _build_nx_graph(problem: GraphProblem) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(problem.n_nodes))
    for edge in problem.edges:
        G.add_edge(edge.node_i, edge.node_j, weight=edge.weight)
    return G


def validate(problem: GraphProblem) -> ValidationResult:
    checks: list[CheckItem] = []
    G = _build_nx_graph(problem)

    # 1. 維度匹配：圖節點數 == 宣告的 n_nodes
    actual_nodes = G.number_of_nodes()
    dim_ok = actual_nodes == problem.n_nodes
    checks.append(CheckItem(
        name="維度匹配檢查",
        passed=dim_ok,
        message=(
            f"圖節點數 {actual_nodes} 與宣告 n_nodes={problem.n_nodes} 一致"
            if dim_ok else
            f"不一致：圖節點數={actual_nodes}，n_nodes={problem.n_nodes}"
        ),
    ))

    # 2. 節點標籤數量 == n_nodes
    label_ok = len(problem.node_labels) == problem.n_nodes
    checks.append(CheckItem(
        name="節點標籤數量檢查",
        passed=label_ok,
        message=(
            f"標籤數量 {len(problem.node_labels)} 與 n_nodes 一致"
            if label_ok else
            f"標籤數量 {len(problem.node_labels)} 與 n_nodes={problem.n_nodes} 不符"
        ),
    ))

    # 3. 孤立節點檢查
    isolated = list(nx.isolates(G))
    iso_ok = len(isolated) == 0
    checks.append(CheckItem(
        name="孤立節點檢查",
        passed=iso_ok,
        message=(
            "無孤立節點"
            if iso_ok else
            f"發現孤立節點（編號 {isolated}），Max-Cut 無法處理孤立節點"
        ),
    ))

    # 4. 邊權重合法性（> 0 且有限）
    bad_weights = [
        (e.node_i, e.node_j, e.weight)
        for e in problem.edges
        if not (math.isfinite(e.weight) and e.weight > 0)
    ]
    weight_ok = len(bad_weights) == 0
    checks.append(CheckItem(
        name="邊權重合法性檢查",
        passed=weight_ok,
        message=(
            f"所有 {len(problem.edges)} 條邊的權重均合法（> 0 且有限）"
            if weight_ok else
            f"發現非法權重：{bad_weights}"
        ),
    ))

    # 5. 節點編號範圍檢查（所有 node_i, node_j 必須在 [0, n_nodes)）
    out_of_range = [
        (e.node_i, e.node_j)
        for e in problem.edges
        if not (0 <= e.node_i < problem.n_nodes and 0 <= e.node_j < problem.n_nodes)
    ]
    range_ok = len(out_of_range) == 0
    checks.append(CheckItem(
        name="節點編號範圍檢查",
        passed=range_ok,
        message=(
            f"所有節點編號均在合法範圍 [0, {problem.n_nodes}) 內"
            if range_ok else
            f"發現超出範圍的節點編號：{out_of_range}"
        ),
    ))

    # 6. 圖連通性檢查（警告等級：不阻斷管線，但記錄異常）
    components = list(nx.connected_components(G))
    n_components = len(components)
    if n_components == 1:
        conn_msg = "圖為連通圖，所有節點均可相互抵達"
        conn_warning = False
    else:
        comp_labels = [
            "{" + ", ".join(problem.node_labels[i] for i in sorted(comp)) + "}"
            for comp in sorted(components, key=min)
        ]
        conn_msg = (
            f"偵測到分離子圖（{n_components} 個連通分量）：{', '.join(comp_labels)}。"
            f"各子圖間無法相互割切，Max-Cut 僅在各子圖內部發生。電路仍可運行。"
        )
        conn_warning = True
    checks.append(CheckItem(
        name="圖連通性檢查",
        passed=True,           # 分離圖仍可求解，不阻斷管線
        message=conn_msg,
        is_warning=conn_warning,
    ))

    # 只有非 warning 的 failed 項目才阻斷
    all_passed = all(c.passed for c in checks if not c.is_warning)
    return ValidationResult(passed=all_passed, checks=checks)
