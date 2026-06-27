# ⚛️ AutoQuantumPipeline

**An automated quantum optimization and visualization pipeline powered by multi-agent LLMs.**

基於多代理語言模型之量子優化自動化與視覺化檢驗管線。

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929c4?logo=ibm)](https://qiskit.org)

---

## ✨ What it does / 功能概述

Describe a combinatorial optimization problem **in plain language** (Chinese or English) — the pipeline automatically:

以**自然語言**描述最佳化問題，管線自動完成以下所有步驟：

| Step | Module | Description |
|------|--------|-------------|
| 1 | **Agent A** | Parses natural language → weighted graph (Google Gemini) |
| 2 | **Agent B** | Validates graph structure across 6 criteria |
| 3 | **Quantum Module** | Generates QAOA circuit (Qamomile + Qiskit) |
| 4 | **Optimizer** | Tunes variational parameters via COBYLA |
| 5 | **Visualizer** | Shows circuit, convergence, partition result & energy landscape |

The Max-Cut problem is solved as a QUBO and run on a statevector simulator.

---

## 🚀 Quick Start / 快速開始

### 1. Clone the repository / 複製專案

```bash
git clone https://github.com/exwtbear/AutoQuantumPipeline.git
cd AutoQuantumPipeline
```

### 2. Create virtual environment / 建立虛擬環境

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get a Google Gemini API Key / 取得 API Key

Visit → **https://aistudio.google.com** (free, no credit card required)

取得免費 Google Gemini API Key，不需要信用卡。

### 4. Set your API Key / 設定 API Key

**Option A (recommended) — Enter directly in the app sidebar:**

直接在 Streamlit 側邊欄的 🔑 API 設定區輸入 Key，不需要任何額外設定。

**Option B — Use a `.env` file:**

```bash
cp .env.example .env
# Edit .env and fill in your key:
# GOOGLE_API_KEY=AIza...your_key...
```

### 5. Run / 啟動應用程式

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📁 Project Structure / 專案結構

```
AutoQuantumPipeline/
├── app.py                  # Streamlit web application (main entry)
├── requirements.txt        # Python dependencies
├── .env.example            # API key template (copy to .env)
│
├── agents/
│   ├── builder_agent.py    # Agent A: NL → GraphProblem (Gemini LLM)
│   └── checker.py          # Agent B: 6-criterion graph validator
│
├── quantum/
│   ├── translator.py       # QUBO → QAOA circuit (Qamomile)
│   ├── simulator.py        # QAOA optimization (Qiskit + COBYLA)
│   └── param_advisor.py    # AI auto parameter selection (p, shots)
│
├── classical/
│   └── brute_force.py      # Exact Max-Cut solver (for comparison)
│
├── models/
│   └── schemas.py          # Pydantic data models (GraphProblem, Edge)
│
├── experiments/            # Reproducible experiment scripts & results
│   ├── run_all.py          # Run all 4 test cases
│   ├── run_stats.py        # Statistical reliability (N=5 runs)
│   ├── run_ablation.py     # QAOA depth p ablation study
│   ├── run_scalability.py  # Circuit scaling analysis (n=4~8)
│   ├── run_robustness.py   # Agent A semantic robustness
│   ├── analyze.py          # Generate all figures
│   └── results/            # CSV results + PNG figures
│
└── report/                 # Academic report (LaTeX + HTML)
    ├── report_en.tex        # English LaTeX (compile with XeLaTeX on Overleaf)
    ├── report_en.html       # English HTML (compile locally with weasyprint)
    ├── report.tex           # Chinese LaTeX
    └── report.html          # Chinese HTML
```

---

## 🧪 Experiments / 實驗重跑

All experiments can be reproduced independently (no LLM calls needed for scalability/ablation):

所有實驗均可獨立重現（規模/消融實驗不需要 LLM）。

```bash
# Set API key first (for Agent A experiments)
export GOOGLE_API_KEY=your_key   # or use .env

# Run all 4 test cases
python experiments/run_all.py

# Statistical reliability (5 independent runs per TC)
python experiments/run_stats.py

# QAOA depth p ablation (p=1,2,3 × N=5 runs)
python experiments/run_ablation.py

# Circuit scalability (n=4~8 random graphs, no LLM)
python experiments/run_scalability.py

# Agent A semantic robustness (3 phrasings × 3 runs)
python experiments/run_robustness.py

# Regenerate all figures
python experiments/analyze.py
```

---

## 📊 Key Results / 主要實驗結果

| Metric | Result |
|--------|--------|
| Pipeline success rate | **100%** (4/4 test cases) |
| Avg. approximation ratio | **r = 0.894** |
| Random baseline | r = 0.568 |
| Agent A robustness | **100%** (9/9 parsing attempts) |
| Circuit depth scaling | Linear O(pm), n=4→8: depth 8→14 |

**p-ablation finding (TC3, N=5):** r(p=1)=0.786±0.004, r(p=2)=0.702±0.012, r(p=3)=0.770±0.009 — non-monotonic due to COBYLA local-minima in higher-dimensional parameter space.

---

## 🔧 Requirements / 環境需求

- **Python** 3.11+
- **Google Gemini API Key** (free at aistudio.google.com)
- See `requirements.txt` for full dependency list

Key packages: `streamlit`, `qiskit-aer`, `qamomile`, `google-genai`, `networkx`, `matplotlib`, `scipy`, `pydantic`, `python-dotenv`

---

## License

MIT License — feel free to use, modify, and distribute.
