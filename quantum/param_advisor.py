import math
from models.schemas import GraphProblem


def suggest_qaoa_params(problem: GraphProblem) -> tuple[int, int, str]:
    """Heuristically decide p (QAOA layers) and shots based on problem structure."""
    n = problem.n_nodes
    m = len(problem.edges)
    max_possible_edges = n * (n - 1) // 2
    density = m / max_possible_edges if max_possible_edges > 0 else 0
    search_space = 2 ** n

    if n <= 5:
        p = 2
        p_reason = f"n={n} 小規模問題，p=2 大幅提升近似品質且模擬仍快速"
    elif n <= 9:
        p = 1
        p_reason = f"n={n} 中等規模，p=1 在效率與近似品質間取得平衡"
    else:
        p = 1
        p_reason = f"n={n} 較大規模，p=1 控制模擬時間在可接受範圍"

    if n <= 4:
        shots = 1024
    elif n <= 6:
        shots = 2048
    elif n <= 8:
        shots = 4096
    else:
        shots = 4096
    shots = max(shots, 512 * p)
    shots = ((shots + 127) // 256) * 256

    avg_samples_per_state = shots / search_space
    rationale = (
        f"圖結構：節點 n={n}，邊 m={m}，圖密度 {density:.0%}，搜尋空間 2^{n}={search_space}\n\n"
        f"- **QAOA 層數 p = {p}**：{p_reason}\n"
        f"- **量測次數 shots = {shots}**："
        f"平均每種切割方案採樣 {avg_samples_per_state:.1f} 次，統計量充足"
    )
    return p, shots, rationale
