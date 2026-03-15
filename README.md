# Traders-Behavior-Analysis-over-Hyperliquid-historical-trading-data
Data science project analyzing market sentiments over the trading patterns with essential criteria.

Visit Application Demo on here -> https://dhanyatha-s-traders-behavior-analysis-over-hyp-dashboard-51xynr.streamlit.app/
### Hyperliquid Historical Trades × Bitcoin Fear & Greed Index

> Exploring how market sentiment affects crypto trader performance — with statistical proof, ML prediction, and an interactive dashboard.

---

## 🧠 What This Project Does

Takes **211,000+ real trades** from Hyperliquid (a crypto derivatives exchange) and asks one simple question:

> *When traders are scared vs greedy — do they make or lose money?*

The answer turns out to be a strong **yes** — and this project proves it with data, charts, statistics, and a machine learning model.

---

## 📁 Project Structure

```
├── TRADER_BEHAVIOR_ANALYSIS.ipynb   # Full analysis notebook 
├── dashboard.py                      # Streamlit interactive dashboard
├── historical_data.csv               # Hyperliquid trade logs 
├── fear_greed_index.csv              # Bitcoin Fear & Greed Index
└── README.md
```

---

## 📦 Datasets

| Dataset | Source | Description |
|---|---|---|
| `historical_data.csv` | [Hyperliquid](https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs) | 211k+ trade records — account, coin, PnL, side, leverage, timestamp |
| `fear_greed_index.csv` | [Alternative.me](https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf) | Daily Bitcoin sentiment score (0–100) + label |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/trader-behavior-insights.git
cd trader-behavior-insights
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn streamlit plotly
```

### 3. Run the Jupyter notebook
```bash
jupyter notebook TRADER_BEHAVIOR_ANNOTATED.ipynb
```

### 4. Or launch the interactive dashboard
```bash
streamlit run dashboard.py
```
Then open `http://localhost:8501` in your browser and upload your CSV files.

---

## 🔍 Analysis Sections

| # | Section | Question Answered |
|---|---|---|
| 1 | Data Loading & Merging | Connect trades to daily sentiment by date |
| 2 | Cleaning & Feature Engineering | Build derived features (leverage proxy, win flag, etc.) |
| 3 | Sentiment Distribution | How fearful/greedy was the market over time? |
| 4 | PnL by Sentiment | Do traders profit more when the market is greedy? |
| 5 | Win Rate Analysis | What % of trades were profitable per regime? |
| 6 | Long/Short Bias | Do traders bet in the right direction based on mood? |
| 7 | Leverage Risk Profile | Does more leverage actually = more profit? |
| 8 | Coin Performance | Which coins do best in which sentiment? |
| 9 | Trader Segmentation | Who are the consistent winners vs losers? |
| 10 | Contrarian Strategy | Can you beat the market by doing the opposite? |
| 11 | Statistical Tests | Are these patterns real or just random noise? |
| 12 | ML Trade Predictor | Can a model predict if a trade will win? |
| 13 | Key Findings | Full auto-generated summary of all conclusions |

---

## 📊 Dashboard Features

The Streamlit dashboard (`dashboard.py`) gives you a fully interactive UI:

- **Upload your own CSVs** via the sidebar — all charts update live
- **10 interactive tabs** — one per analysis section
- **Plotly charts** — hover over any data point for exact values
- **ML tab** — click one button to train a Random Forest on your data
- **Key Findings tab** — auto-generates colour-coded insight cards with your real numbers
- **Works without your files** — runs on sample data so you can explore first

---

## 🔑 Key Findings

These are the conclusions drawn from the real dataset:

**1. Sentiment is the strongest single predictor of PnL**
Greed regimes produce positive average returns. Fear regimes destroy capital. Confirmed by Kruskal-Wallis test (p < 0.001).

**2. Traders systematically fight the market during Fear**
Most traders go long during Fear (trying to catch the bottom). This produces negative expected PnL — a predictable, avoidable loss.

**3. High leverage amplifies loss more than gain**
Trades with 20×+ leverage show high variance but near-zero or negative average outcomes. Leverage is not a shortcut to profit.

**4. The contrarian strategy has a real edge**
Buying during Extreme Fear and selling during Extreme Greed outperforms the average trader baseline by exploiting market overreaction.

**5. Coin selection should match the regime**
High-beta altcoins amplify sentiment — bigger gains in Greed, steeper losses in Fear. BTC/ETH are more stable across all regimes.

---

## 🤖 ML Model

A **Random Forest classifier** is trained to predict whether a trade will be profitable.

**Input features:**
- Sentiment regime (encoded)
- Trade direction (BUY / SELL)
- Leverage proxy
- Hour of day
- Raw Fear/Greed score (0–100)
- Trader's rolling 20-trade win rate

**Output:** Probability that a trade will close with positive PnL

> AUC > 0.55 is considered meaningful in live trading contexts. Perfect prediction (1.0) is impossible in financial markets.

---

## 🎯 Strategic Recommendations

| Rule | Details |
|---|---|
| 🚦 **Sentiment Gate** | Never open a long when F/G index < 30. Wait for it to recover above 35. |
| ⚡ **Leverage Cap** | Hard cap at 10× for directional trades. 20×+ only for short scalps with a stop-loss. |
| 🪙 **Coin Rotation** | F/G < 40 → BTC/ETH only. F/G > 60 → expand into high-beta alts. |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data loading, merging, feature engineering |
| `numpy` | Numerical operations |
| `matplotlib` + `seaborn` | Static charts in the notebook |
| `scipy` | Statistical significance tests |
| `scikit-learn` | Random Forest ML model |
| `streamlit` | Interactive web dashboard |
| `plotly` | Interactive charts in the dashboard |

---

## 📬 Contact

Built as part of a Junior Data Scientist application for **PrimeTrade.ai**

---

*If this helped you, give it a ⭐ on GitHub!*
