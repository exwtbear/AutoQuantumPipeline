import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from models.schemas import GraphProblem

load_dotenv()

_SYSTEM_PROMPT = """你是一個圖論問題解析專家。使用者會描述一個需要將節點分成兩組的最佳化問題。

你的任務：
1. 識別所有節點（人名、部門名、城市名等），並從 0 開始編號
2. 識別所有邊（兩個節點之間的連結）及其權重（若未提及，預設為 1.0）
3. 以 JSON 格式輸出，節點編號從 0 開始

輸出規則：
- n_nodes: 節點總數（整數）
- edges: 邊的列表，每條邊包含 node_i（起點編號）、node_j（終點編號）、weight（浮點數）
- node_labels: 節點名稱列表，索引對應節點編號

注意：只輸出合法 JSON，不要任何解釋文字。"""


def parse_graph_from_text(
    user_input: str,
    max_retries: int = 3,
    api_key: str | None = None,
) -> tuple[GraphProblem, int]:
    """Parse natural language into GraphProblem.

    Returns (problem, attempts_used) where attempts_used counts how many LLM
    calls were made (1 = first-try success, up to max_retries).

    api_key: Google Gemini API key. Falls back to GOOGLE_API_KEY env var.
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Google API Key not set. "
            "Enter it in the sidebar or set GOOGLE_API_KEY in your .env file."
        )
    client = genai.Client(api_key=key)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=f"{_SYSTEM_PROMPT}\n\n使用者輸入：\n{user_input}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GraphProblem,
                    temperature=0,
                ),
            )
            result = response.parsed
            if result is None:
                # Fallback: parse text as JSON manually
                raw = response.text.strip()
                result = GraphProblem.model_validate_json(raw)
            return result, attempt
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Agent A 在 {max_retries} 次嘗試後仍無法解析問題：{e}"
                ) from e
            print(f"[Agent A] 第 {attempt} 次嘗試失敗，重試中... ({e})")
