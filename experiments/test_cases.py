from dataclasses import dataclass, field


@dataclass
class TestCase:
    tc_id: str
    name: str
    topology: str
    nl_input: str
    expected_n_nodes: int
    expected_max_cut: float
    agent_b_should_warn: bool = False


TEST_CASES: list[TestCase] = [
    TestCase(
        tc_id="TC1",
        name="星形圖（無權重）",
        topology="Star Graph",
        expected_n_nodes=5,
        expected_max_cut=4.0,
        nl_input=(
            "我是一個專案經理 (M)，底下有四個工程師 (E1, E2, E3, E4)。"
            "因為管理風格不合，我跟這四個工程師都有直接的嚴重衝突，"
            "但工程師他們彼此之間感情很好，完全沒有衝突。"
            "請幫我把所有人分成兩組，讓衝突的人盡量不在同一組。"
        ),
    ),
    TestCase(
        tc_id="TC2",
        name="K3 三角形（幾何阻挫）",
        topology="Complete Graph K3",
        expected_n_nodes=3,
        expected_max_cut=2.0,
        nl_input=(
            "辦公室裡有 Alice, Bob, Charlie 三個人。"
            "他們是一個三角戀的死對頭：Alice 討厭 Bob，Bob 討厭 Charlie，Charlie 又討厭 Alice。"
            "請幫我把他們分到兩個不同的辦公室，讓仇人碰面的機率降到最低。"
        ),
    ),
    TestCase(
        tc_id="TC3",
        name="帶權重圖（資料中心）",
        topology="Weighted Graph",
        expected_n_nodes=4,
        expected_max_cut=25.0,
        nl_input=(
            "我們有四個資料中心 (D1, D2, D3, D4) 之間存在頻寬干擾。"
            "干擾程度如下：D1 和 D2 之間干擾值高達 10；"
            "D2 和 D3 干擾值為 8；D3 和 D4 干擾值為 2；"
            "D1 和 D4 干擾值為 5；D1 和 D3 之間有輕微干擾，數值為 1。"
            "請將這四個中心分流到兩個獨立網段，使被消除的干擾總值最大化。"
        ),
    ),
    TestCase(
        tc_id="TC4",
        name="分離子圖（連通性警告）",
        topology="Disconnected Graph",
        expected_n_nodes=4,
        expected_max_cut=2.0,
        agent_b_should_warn=True,
        nl_input=(
            "我們公司有台北跟高雄兩個部門。"
            "台北有小明和小華，他們水火不容；"
            "高雄有大寶和二寶，他們也互相討厭。"
            "台北跟高雄的員工互相不認識。"
            "請幫我把這四個人分成兩組。"
        ),
    ),
]
