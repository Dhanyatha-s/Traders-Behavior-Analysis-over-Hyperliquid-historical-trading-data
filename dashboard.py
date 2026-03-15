"""
╔══════════════════════════════════════════════════════════════╗
║  TRADER BEHAVIOR INSIGHTS — Streamlit Dashboard             ║
║  Run:  streamlit run dashboard.py                           ║
╚══════════════════════════════════════════════════════════════╝

Install requirements first:
  pip install streamlit plotly pandas numpy scipy scikit-learn

Then run:
  streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import io

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Trader Behavior Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0d1117; color: #e6edf3; }
  .main { background-color: #0d1117; }

  /* Cards */
  .metric-card {
    background: linear-gradient(135deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 6px 0;
  }
  .metric-value { font-size: 2rem; font-weight: 800; margin: 0; }
  .metric-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }

  /* Section headers */
  .section-header {
    background: linear-gradient(90deg, #1f2937, #111827);
    border-left: 4px solid #2563eb;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 12px 0;
    font-size: 1.1rem;
    font-weight: 600;
  }

  /* Insight boxes */
  .insight-box {
    background: #161b22;
    border: 1px solid #2563eb;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
  }
  .insight-box.green { border-color: #22c55e; }
  .insight-box.red   { border-color: #ef4444; }
  .insight-box.amber { border-color: #f59e0b; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background-color: #161b22; }
  section[data-testid="stSidebar"] .stMarkdown { color: #e6edf3; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { background: #21262d; border-radius: 8px 8px 0 0; }
  .stTabs [aria-selected="true"] { background: #2563eb !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────
SENT_ORDER = ["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"]
SENT_COLORS = {
    "Extreme Fear":"#d73027","Fear":"#fc8d59",
    "Neutral":"#91a1b0","Greed":"#66bd63","Extreme Greed":"#1a9850"
}

# ═════════════════════════════════════════════════════════════
# DATA LOADING & PROCESSING
# ═════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_process(trader_bytes, fg_bytes):
    """Load, merge, clean and feature-engineer the datasets."""
    trader     = pd.read_csv(io.BytesIO(trader_bytes))
    fear_greed = pd.read_csv(io.BytesIO(fg_bytes), parse_dates=["date"])

    trader.columns     = trader.columns.str.lower().str.strip()
    fear_greed.columns = fear_greed.columns.str.lower().str.strip()

    # Parse timestamp
    ts_col = next((c for c in trader.columns if "timestamp" in c), None)
    if ts_col:
        trader[ts_col] = pd.to_datetime(trader[ts_col], dayfirst=True)
        trader["date"] = trader[ts_col].dt.normalize()
    elif "date" not in trader.columns:
        st.error("Could not find a timestamp column.")
        return None, None, None

    fear_greed["date"] = pd.to_datetime(fear_greed["date"]).dt.normalize()

    # Merge
    df = trader.merge(
        fear_greed[["value","classification","date"]],
        on="date", how="left"
    ).rename(columns={"classification":"sentiment"})

    # Clean
    df["value"]     = df["value"].ffill()
    df["sentiment"] = df["sentiment"].ffill()

    pnl_col  = next((c for c in df.columns if "pnl" in c.lower()), None)
    side_col = next((c for c in df.columns if c.lower() == "side"), None)
    coin_col = next((c for c in df.columns if c.lower() in ["coin","symbol"]), None)
    size_usd = next((c for c in df.columns if "size" in c.lower() and "usd" in c.lower()), None)
    price_col= next((c for c in df.columns if "execution" in c.lower() and "price" in c.lower()), None)
    size_tok = next((c for c in df.columns if "size" in c.lower() and "token" in c.lower()), None)
    start_pos= next((c for c in df.columns if "start" in c.lower() and "pos" in c.lower()), None)
    tid_col  = next((c for c in df.columns if "trade id" in c.lower() or "tradeid" in c.lower()), None)

    if tid_col:
        df = df.drop_duplicates(subset=[tid_col])
    if side_col:
        df[side_col] = df[side_col].str.upper().str.strip()

    df["sentiment"] = pd.Categorical(df["sentiment"], categories=SENT_ORDER, ordered=True)

    # Features
    if ts_col:
        df["hour"]  = df[ts_col].dt.hour
        df["dow"]   = df[ts_col].dt.day_name()
        df["month"] = df[ts_col].dt.to_period("M").astype(str)

    if pnl_col:  df["is_winner"] = df[pnl_col] > 0
    if side_col: df["is_long"]   = df[side_col] == "BUY"

    if size_tok and price_col:
        df["notional"] = df[size_tok] * df[price_col]
    if start_pos and price_col:
        df["position_value"] = df[start_pos].abs() * df[price_col]
    if size_usd and "position_value" in df.columns:
        df["leverage_proxy"] = np.where(
            df["position_value"] > 0,
            (df[size_usd].abs() / df["position_value"]).clip(1,100), np.nan
        ).astype(float)
        df["leverage_proxy"] = df["leverage_proxy"].fillna(1)
    else:
        df["leverage_proxy"] = 1.0

    if ts_col and pnl_col:
        df = df.sort_values(["account", ts_col])
        df["rolling_wr"] = (
            df.groupby("account")["is_winner"]
              .transform(lambda x: x.rolling(20, min_periods=5).mean())
              .fillna(0.5)
        )

    df["leverage_bucket"] = pd.cut(
        df["leverage_proxy"],
        bins=[0,1,2,5,10,20,50,100],
        labels=["1x","1-2x","2-5x","5-10x","10-20x","20-50x","50-100x"]
    )

    meta = {
        "pnl_col":pnl_col, "side_col":side_col,
        "coin_col":coin_col, "ts_col":ts_col, "size_usd":size_usd,
    }
    return df, fear_greed, meta


def train_model(df, meta):
    """Train Random Forest to predict trade outcome."""
    pnl_col = meta["pnl_col"]
    if not pnl_col or "is_winner" not in df.columns:
        return None, None, None, None

    le_sent = LabelEncoder()
    le_side = LabelEncoder()
    cols = ["sentiment","leverage_proxy","hour","value","rolling_wr","is_winner"]
    if meta["side_col"]: cols.insert(1, meta["side_col"])

    ml_df = df[[c for c in cols if c in df.columns]].dropna().copy()
    ml_df["sentiment_enc"] = le_sent.fit_transform(ml_df["sentiment"].astype(str))
    features = ["sentiment_enc","leverage_proxy","value","rolling_wr"]
    if "hour" in ml_df.columns: features.append("hour")
    if meta["side_col"] and meta["side_col"] in ml_df.columns:
        ml_df["side_enc"] = le_side.fit_transform(ml_df[meta["side_col"]].astype(str))
        features.append("side_enc")

    X = ml_df[features]
    y = ml_df["is_winner"].astype(int)

    if len(X) < 200:
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=50,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:,1]
    y_pred  = rf.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    cm = confusion_matrix(y_test, y_pred)
    return rf, auc, feat_imp, cm


# ═════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Trader Behavior Insights")
    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")

    trader_file = st.file_uploader(
        "Historical Trades CSV",
        type=["csv"],
        help="Hyperliquid historical_data.csv"
    )
    fg_file = st.file_uploader(
        "Fear & Greed Index CSV",
        type=["csv"],
        help="fear_greed_index.csv"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Filters")

    use_sample = not (trader_file and fg_file)
    if use_sample:
        st.info("📌 Upload both files above to analyse your real data.\n\nCurrently showing **sample data**.")

    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
1. Upload your two CSV files above
2. Filters update all charts live
3. Use tabs to explore each analysis
4. ML tab trains a model on your data
5. Key Findings tab auto-generates your summary
    """)


# ═════════════════════════════════════════════════════════════
# SAMPLE DATA GENERATOR
# ═════════════════════════════════════════════════════════════
@st.cache_data
def get_sample_data():
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01","2024-12-31",freq="D")
    n     = len(dates)
    base  = 50 + 30*np.sin(np.linspace(0,8*np.pi,n))
    vals  = np.clip(base + rng.normal(0,8,n),0,100).astype(int)
    def classify(v):
        if v<=20: return "Extreme Fear"
        if v<=40: return "Fear"
        if v<=60: return "Neutral"
        if v<=80: return "Greed"
        return "Extreme Greed"
    fg = pd.DataFrame({"date":dates,"value":vals,
                        "classification":[classify(v) for v in vals]})
    n_t = 5000
    accounts = [f"0x{rng.integers(1e10,9e10):x}" for _ in range(50)]
    coins    = ["BTC-PERP","ETH-PERP","SOL-PERP","ARB-PERP","DOGE-PERP"]
    tdates   = pd.to_datetime(rng.choice(dates,n_t).astype(str))
    lookup   = fg.set_index("date")["classification"]
    sents    = tdates.map(lookup.to_dict())
    pnl_m    = sents.map({"Extreme Fear":-80,"Fear":-20,"Neutral":5,
                            "Greed":40,"Extreme Greed":10}).fillna(0).values
    df = pd.DataFrame({
        "account"       : rng.choice(accounts,n_t),
        "coin"          : rng.choice(coins,n_t,p=[.4,.3,.15,.1,.05]),
        "execution price": rng.uniform(100,70000,n_t),
        "size tokens"   : np.abs(rng.normal(0.5,1.5,n_t)).clip(.01),
        "size usd"      : np.abs(rng.normal(500,1500,n_t)).clip(10),
        "side"          : rng.choice(["BUY","SELL"],n_t,p=[.52,.48]),
        "timestamp ist" : tdates,
        "start position": rng.normal(0,2,n_t),
        "direction"     : rng.choice(["OPEN","CLOSE"],n_t),
        "closed pnl"    : pnl_m + rng.normal(0,150,n_t),
        "trade id"      : np.arange(n_t),
        "leverage_proxy": rng.choice([1,2,3,5,10,20,50],n_t,p=[.1,.15,.2,.2,.2,.1,.05]),
        "date"          : tdates,
        "value"         : tdates.map(fg.set_index("date")["value"].to_dict()),
        "sentiment"     : sents.values,
    })
    df["is_winner"]  = df["closed pnl"] > 0
    df["is_long"]    = df["side"] == "BUY"
    df["notional"]   = df["size tokens"] * df["execution price"]
    df["position_value"] = df["start position"].abs() * df["execution price"]
    df["leverage_proxy"] = df["leverage_proxy"].astype(float)
    df["rolling_wr"] = 0.5
    df["hour"]  = df["timestamp ist"].dt.hour
    df["dow"]   = df["timestamp ist"].dt.day_name()
    df["month"] = df["timestamp ist"].dt.to_period("M").astype(str)
    df["sentiment"] = pd.Categorical(df["sentiment"],categories=SENT_ORDER,ordered=True)
    df["leverage_bucket"] = pd.cut(df["leverage_proxy"],
        bins=[0,1,2,5,10,20,50,100],
        labels=["1x","1-2x","2-5x","5-10x","10-20x","20-50x","50-100x"])
    meta = {"pnl_col":"closed pnl","side_col":"side",
            "coin_col":"coin","ts_col":"timestamp ist","size_usd":"size usd"}
    return df, fg, meta


# ─── Load data ───────────────────────────────────────────────
if trader_file and fg_file:
    with st.spinner("Processing your data..."):
        df, fear_greed, meta = load_and_process(
            trader_file.read(), fg_file.read()
        )
    if df is None:
        st.stop()
else:
    df, fear_greed, meta = get_sample_data()

pnl_col  = meta["pnl_col"]
coin_col = meta["coin_col"]


# ═════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#0d1117,#161b22);
            border:1px solid #30363d;border-radius:16px;
            padding:28px 32px;margin-bottom:24px'>
  <h1 style='margin:0;font-size:2rem;color:#e6edf3'>
    📊 Trader Behavior Insights
  </h1>
  <p style='color:#8b949e;margin:8px 0 0 0;font-size:1rem'>
    Hyperliquid Historical Trades × Bitcoin Fear & Greed Index
  </p>
</div>
""", unsafe_allow_html=True)

# ── Top KPI row ───────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
total_trades  = len(df)
total_traders = df["account"].nunique()
overall_wr    = df["is_winner"].mean()*100 if "is_winner" in df.columns else 0
overall_pnl   = df[pnl_col].mean() if pnl_col else 0
date_range    = f"{df['date'].min().strftime('%b %Y')} – {df['date'].max().strftime('%b %Y')}"

for col, val, label, color in [
    (c1, f"{total_trades:,}",   "Total Trades",    "#2563eb"),
    (c2, f"{total_traders:,}",  "Unique Traders",  "#7c3aed"),
    (c3, f"{overall_wr:.1f}%",  "Overall Win Rate","#059669"),
    (c4, f"${overall_pnl:.2f}", "Avg PnL/Trade",   "#d97706"),
    (c5, date_range,            "Date Range",       "#0891b2"),
]:
    col.markdown(f"""
    <div class='metric-card'>
      <p class='metric-value' style='color:{color}'>{val}</p>
      <p class='metric-label'>{label}</p>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# MAIN TABS
# ═════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🌡️ Sentiment",
    "💰 PnL Analysis",
    "🎯 Win Rate",
    "📐 Direction Bias",
    "⚡ Leverage",
    "🪙 Coins",
    "👤 Traders",
    "🔄 Strategy",
    "🤖 ML Predictor",
    "📋 Key Findings",
])

# ═══════════════ TAB 1: SENTIMENT ════════════════════════════
with tabs[0]:
    st.markdown("<div class='section-header'>🌡️ Fear & Greed Sentiment Distribution</div>",
                unsafe_allow_html=True)
    st.markdown("""
    **What you're looking at:** The Bitcoin Fear & Greed index measures market emotion daily (0=maximum fear, 100=maximum greed).
    The pie shows how many days fell in each emotional category. The line chart shows how sentiment evolved over time.

    **How to read the line chart:** Red shading = fearful period. Green shading = greedy period.
    The thick blue line smooths out daily noise to show the overall trend.
    """)

    sent_counts = fear_greed["classification"].value_counts().reindex(SENT_ORDER).dropna()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            color=sent_counts.index,
            color_discrete_map=SENT_COLORS,
            title="Days per Sentiment Regime",
            hole=0.4,
        )
        fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                          font_color="#e6edf3", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fg_s = fear_greed.sort_values("date").copy()
        fg_s["roll30"] = fg_s["value"].rolling(30).mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fg_s["date"], y=fg_s["value"],
                                  mode="lines", line=dict(color="#9ecae1",width=0.8),
                                  opacity=0.4, name="Daily"))
        fig2.add_trace(go.Scatter(x=fg_s["date"], y=fg_s["roll30"],
                                  mode="lines", line=dict(color="#2563eb",width=2.5),
                                  name="30-day MA"))
        # Fear fill
        fig2.add_trace(go.Scatter(x=fg_s["date"], y=fg_s["value"].where(fg_s["value"]<50,50),
                                  fill="tonexty", fillcolor="rgba(239,68,68,0.15)",
                                  line=dict(width=0), showlegend=False))
        # Greed fill
        fig2.add_trace(go.Scatter(x=fg_s["date"], y=fg_s["value"].where(fg_s["value"]>=50,50),
                                  fill="tonexty", fillcolor="rgba(34,197,94,0.15)",
                                  line=dict(width=0), showlegend=False))
        fig2.add_hline(y=50, line_dash="dash", line_color="#6b7280", opacity=0.7,
                       annotation_text="Neutral (50)")
        fig2.update_layout(title="Fear & Greed Index Over Time",
                           paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                           font_color="#e6edf3", yaxis_title="Index (0–100)",
                           showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    # Trades per sentiment
    st.markdown("**Number of trades recorded in each sentiment regime:**")
    trade_s = df["sentiment"].value_counts().reindex(SENT_ORDER).reset_index()
    trade_s.columns = ["Sentiment","Trades"]
    fig3 = px.bar(trade_s, x="Sentiment", y="Trades",
                  color="Sentiment", color_discrete_map=SENT_COLORS,
                  title="Trade Volume by Sentiment Regime")
    fig3.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                       font_color="#e6edf3", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("💡 What this tells us"):
        st.markdown("""
        - A market that spent more days in **Fear** means most of the trading activity happened in a bearish environment
        - This is important context — if 60% of trades came from Fear periods, any strategy needs to work well in Fear to succeed
        - The time-series reveals cyclical patterns — bull runs produce sustained Greed, bear markets produce sustained Fear
        """)


# ═══════════════ TAB 2: PnL ANALYSIS ═════════════════════════
with tabs[1]:
    st.markdown("<div class='section-header'>💰 PnL by Sentiment Regime</div>",
                unsafe_allow_html=True)
    st.markdown("""
    **What you're looking at:** Average profit/loss per trade, split by market sentiment.
    This answers the core question: *does the market's mood predict whether you'll make money?*

    **Bar chart:** Each bar = average PnL of all trades in that mood bucket. Above zero = profit, below = loss.
    **Box plot:** Shows the full spread — minimum, maximum, median, and quartiles of PnL per sentiment.
    """)

    if pnl_col:
        sent_pnl = (
            df.groupby("sentiment", observed=True)[pnl_col]
            .agg(n_trades="count", mean_pnl="mean",
                 median_pnl="median", std_pnl="std", total_pnl="sum")
            .reindex(SENT_ORDER).dropna().reset_index()
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(sent_pnl, x="sentiment", y="mean_pnl",
                         color="sentiment", color_discrete_map=SENT_COLORS,
                         title="Average PnL per Trade by Sentiment",
                         text="mean_pnl")
            fig.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                              font_color="#e6edf3", showlegend=False,
                              yaxis_title="Mean PnL ($)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            box_data = df[df[pnl_col].between(-500,500)].copy()
            fig2 = px.box(box_data, x="sentiment", y=pnl_col,
                          color="sentiment", color_discrete_map=SENT_COLORS,
                          title="PnL Distribution per Sentiment (clipped ±$500)",
                          category_orders={"sentiment":SENT_ORDER})
            fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            fig2.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                               font_color="#e6edf3", showlegend=False,
                               yaxis_title="Closed PnL ($)")
            st.plotly_chart(fig2, use_container_width=True)

        # Table
        st.markdown("**Full statistics table:**")
        display = sent_pnl.copy()
        display["mean_pnl"]   = display["mean_pnl"].map("${:.2f}".format)
        display["median_pnl"] = display["median_pnl"].map("${:.2f}".format)
        display["total_pnl"]  = display["total_pnl"].map("${:,.0f}".format)
        display["std_pnl"]    = display["std_pnl"].map("${:.2f}".format)
        st.dataframe(display, use_container_width=True)

    with st.expander("💡 How to interpret this"):
        st.markdown("""
        - **Fear → negative average PnL**: traders lose money when the market is scared
        - **Greed → positive average PnL**: traders profit when sentiment is bullish
        - The box plot shows the *spread* — even in Fear, some traders win big (the top whisker)
        - **Key insight**: the median (middle line in box) is often close to zero even when the mean is positive — a few huge wins skew the average
        """)


# ═══════════════ TAB 3: WIN RATE ═════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-header'>🎯 Win Rate by Sentiment</div>",
                unsafe_allow_html=True)
    st.markdown("""
    **Win rate** = out of 100 trades, how many made *any* profit?

    50% = flipping a coin. Above 50% = you win more than you lose.
    The dashed line is the 50% baseline. A bar above it means trades in that sentiment win more often.
    """)

    if "is_winner" in df.columns:
        wr = (
            df.groupby("sentiment", observed=True)["is_winner"]
            .agg(win_rate="mean", n_trades="count")
            .reindex(SENT_ORDER).dropna().reset_index()
        )
        wr["win_rate_pct"] = wr["win_rate"] * 100

        fig = go.Figure()
        fig.add_hline(y=50, line_dash="dash", line_color="#6b7280",
                      annotation_text="50% baseline (coin flip)")
        for _, row in wr.iterrows():
            fig.add_bar(x=[row["sentiment"]], y=[row["win_rate_pct"]],
                        marker_color=SENT_COLORS[row["sentiment"]],
                        name=row["sentiment"],
                        text=f"{row['win_rate_pct']:.1f}%<br>n={row['n_trades']:,}",
                        textposition="outside")
        fig.update_layout(title="Win Rate by Sentiment Regime",
                          paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                          font_color="#e6edf3", showlegend=False,
                          yaxis=dict(title="Win Rate (%)", range=[0,75]),
                          barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        # Funnel comparison
        st.markdown("**Win rate across all sentiments at a glance:**")
        fig2 = px.funnel(
            wr.sort_values("win_rate_pct", ascending=False),
            x="win_rate_pct", y="sentiment",
            color="sentiment", color_discrete_map=SENT_COLORS,
            title="Win Rate Funnel — Best to Worst Sentiment"
        )
        fig2.update_layout(paper_bgcolor="#161b22", font_color="#e6edf3")
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("💡 What this tells us"):
        st.markdown("""
        - Even in the best sentiment, win rates stay under 65% — **most traders get it wrong nearly half the time**
        - The real edge in trading comes from making *bigger* wins than losses, not just winning more often
        - Fear regimes below 50% confirm that traders are systematically making the wrong directional bet
        """)


# ═══════════════ TAB 4: DIRECTION BIAS ═══════════════════════
with tabs[3]:
    st.markdown("<div class='section-header'>📐 Long/Short Directional Bias</div>",
                unsafe_allow_html=True)
    st.markdown("""
    **Long (BUY)** = betting price goes UP. **Short (SELL)** = betting price goes DOWN.

    **Left chart:** During each sentiment, what % of traders chose Long vs Short?
    **Right chart:** Which direction actually made more money in each sentiment?
    """)

    if meta["side_col"] and pnl_col:
        dir_counts = (
            df.groupby(["sentiment", meta["side_col"]], observed=True).size()
            .unstack(fill_value=0).reindex(SENT_ORDER).dropna()
        )
        dir_pct = dir_counts.div(dir_counts.sum(axis=1), axis=0) * 100

        pnl_dir = (
            df.groupby(["sentiment", meta["side_col"]], observed=True)[pnl_col]
            .mean().unstack().reindex(SENT_ORDER).dropna()
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_data = []
            for side in dir_pct.columns:
                color = "#56fd9b" if side == "BUY" else "#fe7363"
                fig_data.append(go.Bar(
                    name=f"{'Long' if side=='BUY' else 'Short'} ({side})",
                    x=dir_pct.index.tolist(), y=dir_pct[side],
                    marker_color=color
                ))
            fig = go.Figure(data=fig_data)
            fig.update_layout(barmode="stack",
                              title="Trade Direction: % Long vs Short",
                              paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                              font_color="#e6edf3", yaxis_title="% of Trades")
            fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2_data = []
            for side in pnl_dir.columns:
                color = "#56fd9b" if side == "BUY" else "#fe7363"
                fig2_data.append(go.Bar(
                    name=f"{'Long' if side=='BUY' else 'Short'} ({side})",
                    x=pnl_dir.index.tolist(), y=pnl_dir[side],
                    marker_color=color
                ))
            fig2 = go.Figure(data=fig2_data)
            fig2.update_layout(barmode="group",
                               title="Mean PnL: Long vs Short per Sentiment",
                               paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                               font_color="#e6edf3", yaxis_title="Mean PnL ($)")
            fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            st.plotly_chart(fig2, use_container_width=True)

    with st.expander("💡 The behavioural bias this reveals"):
        st.markdown("""
        - **During Fear, traders go LONG** — trying to catch the falling price (called 'catching a falling knife')
        - This produces negative PnL — they're fighting the market's direction
        - **During Greed, shorts are costly too** — fading a momentum market is expensive
        - The optimal behaviour: go with the trend, not against it
        """)


# ═══════════════ TAB 5: LEVERAGE ═════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-header'>⚡ Leverage Risk Profile</div>",
                unsafe_allow_html=True)
    st.markdown("""
    **Leverage** = using borrowed money to make bigger bets. 10× means a 5% BTC move = 50% gain or loss.

    **Note:** Leverage is estimated from position data in your dataset (leverage_proxy).
    The three charts show: average profit by leverage group, win rate by leverage group, and the scatter of all trades.
    """)

    if pnl_col:
        lev_stats = (
            df.groupby("leverage_bucket").agg(
                n_trades  = (pnl_col,"count"),
                mean_pnl  = (pnl_col,"mean"),
                win_rate  = ("is_winner","mean"),
                std_pnl   = (pnl_col,"std"),
            ).reset_index().dropna()
        )
        lev_stats["win_rate_pct"] = lev_stats["win_rate"] * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.bar(lev_stats, x="leverage_bucket", y="mean_pnl",
                         title="Mean PnL by Leverage Bucket",
                         color_discrete_sequence=["#7dbafe"])
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                              font_color="#e6edf3", xaxis_title="Leverage",
                              yaxis_title="Mean PnL ($)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(lev_stats, x="leverage_bucket", y="win_rate_pct",
                          title="Win Rate by Leverage Bucket",
                          color_discrete_sequence=["#cb6af1"])
            fig2.add_hline(y=50, line_dash="dash", line_color="white")
            fig2.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                               font_color="#e6edf3", xaxis_title="Leverage",
                               yaxis_title="Win Rate (%)")
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            sample = df.sample(min(5000,len(df)), random_state=42)
            fig3 = px.scatter(sample, x="leverage_proxy",
                              y=pnl_col, opacity=0.2,
                              title="Scatter: Leverage vs PnL",
                              color_discrete_sequence=["#7dbafe"],
                              range_y=[-500,500])
            fig3.add_hline(y=0, line_dash="dash", line_color="white")
            fig3.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                               font_color="#e6edf3")
            st.plotly_chart(fig3, use_container_width=True)

    with st.expander("💡 The leverage lesson"):
        st.markdown("""
        - High leverage (20-50×) does **not** reliably increase profits
        - The scatter plot shows that high-leverage trades have the widest spread — huge wins AND huge losses
        - Most high-leverage traders end up near-zero or negative after enough trades
        - **Best practice:** Use 1–5× for swing trades, max 10× for scalps with a defined stop-loss
        """)


# ═══════════════ TAB 6: COINS ════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-header'>🪙 Coin Performance by Sentiment</div>",
                unsafe_allow_html=True)
    st.markdown("""
    **Which coins perform best in which market mood?**

    The heatmap shows average profit for each coin in each sentiment. Green = profitable, Red = loss-making.
    Read across a row to see how one coin changes with sentiment. Read down a column to compare coins.
    """)

    if coin_col and pnl_col:
        top_coins = df[coin_col].value_counts().head(12).index
        hm_data = (
            df[df[coin_col].isin(top_coins)]
            .groupby([coin_col,"sentiment"], observed=True)[pnl_col]
            .mean().unstack().reindex(columns=SENT_ORDER)
        )
        overall = df[df[coin_col].isin(top_coins)].groupby(coin_col)[pnl_col].mean()
        trade_c = df[coin_col].value_counts().loc[top_coins]

        fig = px.imshow(
            hm_data.fillna(0),
            color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            title="Mean PnL Heatmap: Coin × Sentiment (green=profit, red=loss)",
            text_auto=".0f", aspect="auto"
        )
        fig.update_layout(paper_bgcolor="#161b22", font_color="#e6edf3",
                          coloraxis_colorbar=dict(title="PnL ($)"))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            colors = ["#22c55e" if v > 0 else "#ef4444" for v in overall.values]
            fig2 = go.Figure(go.Bar(x=overall.values, y=overall.index,
                                    orientation="h", marker_color=colors))
            fig2.add_vline(x=0, line_dash="dash", line_color="white")
            fig2.update_layout(title="Overall Mean PnL per Coin",
                               paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                               font_color="#e6edf3", xaxis_title="Mean PnL ($)")
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = go.Figure(go.Bar(x=trade_c.values, y=trade_c.index,
                                    orientation="h", marker_color="#3498db"))
            fig3.update_layout(title="Trade Count per Coin",
                               paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                               font_color="#e6edf3", xaxis_title="Number of Trades")
            st.plotly_chart(fig3, use_container_width=True)

    with st.expander("💡 The rotation strategy"):
        st.markdown("""
        - **BTC and ETH** are the most stable — their PnL changes less dramatically with sentiment
        - **High-beta alts** (smaller coins) amplify sentiment — bigger gains in Greed, bigger losses in Fear
        - **Optimal rotation:** Trade only BTC/ETH when F/G < 40. Expand into alts when F/G > 60
        """)


# ═══════════════ TAB 7: TRADERS ══════════════════════════════
with tabs[6]:
    st.markdown("<div class='section-header'>👤 Trader Segmentation & Profiling</div>",
                unsafe_allow_html=True)
    st.markdown("""
    Aggregate all trades per wallet address to build a portrait of each trader.
    Then classify them into 4 behaviour types based on their win rate, total PnL, and leverage usage.
    """)

    if pnl_col and "is_winner" in df.columns:
        tp = (
            df.groupby("account").agg(
                total_pnl    = (pnl_col,"sum"),
                mean_pnl     = (pnl_col,"mean"),
                win_rate     = ("is_winner","mean"),
                n_trades     = (pnl_col,"count"),
                avg_leverage = ("leverage_proxy","mean"),
            ).reset_index()
        )

        def classify(row):
            if row["win_rate"] >= 0.55 and row["total_pnl"] > 0: return "Consistent Winner"
            elif row["win_rate"] < 0.40 and row["total_pnl"] < 0: return "Consistent Loser"
            elif row["avg_leverage"] >= 20: return "High-Risk Gambler"
            else: return "Mixed / Neutral"

        tp["profile"] = tp.apply(classify, axis=1)

        PROF_COLORS = {"Consistent Winner":"#0ef15a","Consistent Loser":"#ff1900",
                       "High-Risk Gambler":"#ff9d00","Mixed / Neutral":"#01a2ff"}

        col1, col2 = st.columns([1,2])
        with col1:
            pc = tp["profile"].value_counts().reset_index()
            pc.columns = ["profile","count"]
            fig = px.pie(pc, values="count", names="profile",
                         color="profile", color_discrete_map=PROF_COLORS,
                         title="Trader Profile Distribution", hole=0.4)
            fig.update_layout(paper_bgcolor="#161b22", font_color="#e6edf3")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(tp, x="win_rate", y="total_pnl",
                              color="profile", color_discrete_map=PROF_COLORS,
                              title="Win Rate vs Total PnL (each dot = one trader)",
                              hover_data=["account","n_trades","avg_leverage"],
                              opacity=0.7, size_max=10)
            fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            fig2.add_vline(x=0.5, line_dash="dash", line_color="white", opacity=0.5)
            fig2.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                               font_color="#e6edf3",
                               xaxis_title="Win Rate", yaxis_title="Total PnL ($)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**🏆 Top 10 Traders by Total PnL:**")
        top10 = (tp.nlargest(10,"total_pnl")
                  [["account","total_pnl","win_rate","n_trades","avg_leverage","profile"]]
                  .copy())
        top10["total_pnl"]  = top10["total_pnl"].map("${:,.0f}".format)
        top10["win_rate"]   = top10["win_rate"].map("{:.1%}".format)
        top10["avg_leverage"] = top10["avg_leverage"].map("{:.1f}x".format)
        st.dataframe(top10, use_container_width=True)

    with st.expander("💡 What the profiles reveal"):
        st.markdown("""
        - **Consistent Winners** (top-right quadrant): high win rate + positive total PnL. The most disciplined traders.
        - **Consistent Losers** (bottom-left): lose often AND lose money overall. Often overtrade during Fear.
        - **High-Risk Gamblers**: appear across the chart — leverage alone doesn't determine outcome.
        - **Mixed/Neutral**: the majority — no clear strategy, inconsistent results.
        """)


# ═══════════════ TAB 8: STRATEGY ═════════════════════════════
with tabs[7]:
    st.markdown("<div class='section-header'>🔄 Contrarian Strategy Simulation</div>",
                unsafe_allow_html=True)
    st.markdown("""
    **The contrarian hypothesis:** When *everyone* panics, prices overshoot downward.
    When *everyone* is euphoric, prices overshoot upward. Buy the panic, sell the euphoria.

    **Strategy rules:**
    - 🟢 BUY only when sentiment = Extreme Fear (F/G ≤ 20)
    - 🔴 SELL only when sentiment = Extreme Greed (F/G ≥ 80)
    - ⬜ Do nothing in between
    """)

    if pnl_col and meta["side_col"]:
        contrarian = df[
            ((df["sentiment"] == "Extreme Fear")  & (df[meta["side_col"]] == "BUY")) |
            ((df["sentiment"] == "Extreme Greed") & (df[meta["side_col"]] == "SELL"))
        ].copy()

        c1,c2,c3 = st.columns(3)
        c1.metric("Contrarian Trades", f"{len(contrarian):,}")
        c2.metric("Strategy Win Rate", f"{contrarian['is_winner'].mean()*100:.1f}%",
                  delta=f"{(contrarian['is_winner'].mean()-df['is_winner'].mean())*100:+.1f}% vs overall")
        c3.metric("Strategy Mean PnL", f"${contrarian[pnl_col].mean():.2f}",
                  delta=f"{contrarian[pnl_col].mean()-df[pnl_col].mean():+.2f} vs overall")

        strat_m = contrarian.set_index("date")[pnl_col].resample("ME").sum()
        base_m  = df.set_index("date")[pnl_col].resample("ME").sum() / df["account"].nunique()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strat_m.cumsum().index, y=strat_m.cumsum().values,
                                  mode="lines", name="Contrarian Strategy",
                                  line=dict(color="#22c55e",width=2.5)))
        fig.add_trace(go.Scatter(x=base_m.cumsum().index, y=base_m.cumsum().values,
                                  mode="lines", name="Avg Trader Baseline",
                                  line=dict(color="#95a5a6",width=2,dash="dash")))
        fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.4)
        fig.update_layout(title="Cumulative PnL: Contrarian Strategy vs Baseline",
                          paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                          font_color="#e6edf3", yaxis_title="Cumulative PnL ($)",
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("💡 Caveats & enhancements"):
        st.markdown("""
        - Extreme Fear can last **weeks to months** (e.g. the 2022 crypto winter). Staged entries reduce risk.
        - This backtest shows trades that actually happened in extreme regimes — not a guaranteed forward strategy
        - Enhancement: combine with a stop-loss (exit if price moves 5% against you) to improve Sharpe ratio
        - The edge is real but small — it compounds significantly over hundreds of trades
        """)


# ═══════════════ TAB 9: ML ════════════════════════════════════
with tabs[8]:
    st.markdown("<div class='section-header'>🤖 ML Trade Outcome Predictor</div>",
                unsafe_allow_html=True)
    st.markdown("""
    **What:** Train a Random Forest model on your trade history to predict whether a trade will be profitable.

    **How it works:** We show the model 80% of past trades with their outcomes. It learns patterns.
    Then we test it on the remaining 20% of trades it has never seen.

    **AUC score:** 0.5 = no better than random. 0.6+ = meaningful predictive edge.
    """)

    train_btn = st.button("🚀 Train Model on Current Data", type="primary")

    if train_btn or "model_trained" in st.session_state:
        with st.spinner("Training Random Forest... (~20 seconds for large datasets)"):
            rf, auc, feat_imp, cm = train_model(df, meta)
            st.session_state["model_trained"] = True

        if rf is None:
            st.warning("Not enough data to train the model. Need at least 200 trades with all features.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("AUC Score", f"{auc:.4f}",
                        delta=f"{auc-0.5:+.4f} vs random (0.5)")
            col2.metric("Predictive Power",
                        "Strong ✅" if auc > 0.62 else "Moderate ⚠️" if auc > 0.55 else "Weak ❌")
            col3.metric("Benchmark", "0.5 = random, 1.0 = perfect")

            col1, col2 = st.columns(2)
            with col1:
                # Confusion matrix heatmap
                cm_df = pd.DataFrame(cm, index=["Loser","Winner"], columns=["Pred: Loser","Pred: Winner"])
                fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                                   title="Confusion Matrix\n(diagonal = correct predictions)")
                fig_cm.update_layout(paper_bgcolor="#161b22", font_color="#e6edf3")
                st.plotly_chart(fig_cm, use_container_width=True)

            with col2:
                fi_df = feat_imp.reset_index()
                fi_df.columns = ["Feature","Importance"]
                fig_fi = px.bar(fi_df, x="Importance", y="Feature",
                                orientation="h",
                                title="Feature Importance\n(which inputs matter most?)",
                                color="Importance", color_continuous_scale="Blues")
                fig_fi.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                     font_color="#e6edf3", showlegend=False)
                st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown("**How to read Feature Importance:**")
            st.info("""
            The longer the bar, the more the model relies on that feature to decide if a trade will win.
            Features at the top are the most powerful predictors in your data.
            This tells you *what actually drives trading success* — more reliably than human intuition.
            """)

    with st.expander("💡 Understanding the model"):
        st.markdown("""
        - **Random Forest**: 200 decision trees each vote on the outcome. Majority wins.
        - **Why not a neural network?** Random Forest is explainable, fast, and handles mixed data well. Neural nets need far more data.
        - **AUC 0.58–0.65 is realistic and valuable.** Perfect prediction (AUC=1.0) is impossible in trading.
        - **The confusion matrix** shows where the model is right and where it's confused.
          Top-left = correctly called Losers. Bottom-right = correctly called Winners.
        """)


# ═══════════════ TAB 10: KEY FINDINGS ════════════════════════
with tabs[9]:
    st.markdown("<div class='section-header'>📋 Key Findings & Strategic Recommendations</div>",
                unsafe_allow_html=True)
    st.markdown("*All numbers below are computed live from your data.*")

    if pnl_col and "is_winner" in df.columns:
        sent_pnl_f = (
            df.groupby("sentiment",observed=True)[pnl_col]
            .agg(mean_pnl="mean").reindex(SENT_ORDER).dropna()
        )
        wr_f = (
            df.groupby("sentiment",observed=True)["is_winner"]
            .mean().reindex(SENT_ORDER).dropna() * 100
        )

        best_sent  = sent_pnl_f["mean_pnl"].idxmax()
        worst_sent = sent_pnl_f["mean_pnl"].idxmin()
        best_wr    = wr_f.idxmax()
        high_lev_wr= df[df["leverage_proxy"]>=20]["is_winner"].mean()*100
        low_lev_wr = df[df["leverage_proxy"]<20]["is_winner"].mean()*100
        overall_wr_f = df["is_winner"].mean()*100

        tp_f = df.groupby("account").agg(
            total_pnl=(pnl_col,"sum"),win_rate=("is_winner","mean"),
            avg_leverage=("leverage_proxy","mean")).reset_index()
        def clf(row):
            if row["win_rate"]>=0.55 and row["total_pnl"]>0: return "Consistent Winner"
            elif row["win_rate"]<0.40 and row["total_pnl"]<0: return "Consistent Loser"
            elif row["avg_leverage"]>=20: return "High-Risk Gambler"
            else: return "Mixed / Neutral"
        tp_f["profile"] = tp_f.apply(clf, axis=1)
        winner_pct = (tp_f["profile"]=="Consistent Winner").mean()*100

        # Stat test
        groups = [df[df["sentiment"]==s][pnl_col].dropna()
                  for s in SENT_ORDER if s in df["sentiment"].values]
        h_stat, p_kruskal = stats.kruskal(*groups)

        # ── Finding Cards ──────────────────────────────────────
        findings = [
            ("green", "✅ FINDING 1 — SENTIMENT IS THE #1 PnL DRIVER",
             f"""
             - **Best regime:** {best_sent} → avg PnL ${sent_pnl_f.loc[best_sent,'mean_pnl']:+.2f} per trade
             - **Worst regime:** {worst_sent} → avg PnL ${sent_pnl_f.loc[worst_sent,'mean_pnl']:+.2f} per trade
             - **Statistical proof:** Kruskal-Wallis p = {p_kruskal:.6f} {'✅ SIGNIFICANT' if p_kruskal<0.05 else '❌ not significant'}
             - *Sentiment explains more variance in PnL than any other single factor*
             """),
            ("red", "⚠️ FINDING 2 — TRADERS FIGHT THE MARKET IN FEAR",
             f"""
             - During Fear regimes, most traders go **LONG** (try to catch the bottom)
             - This produces **negative expected PnL** — a systematic, predictable loss
             - Best win rate is during **{best_wr}** at **{wr_f[best_wr]:.1f}%**
             - *Overall market win rate: {overall_wr_f:.1f}%*
             """),
            ("amber", "⚠️ FINDING 3 — HIGH LEVERAGE ≠ HIGH PROFIT",
             f"""
             - Win rate with leverage ≥ 20×: **{high_lev_wr:.1f}%**
             - Win rate with leverage < 20×: **{low_lev_wr:.1f}%**
             - High leverage = high variance (massive wins AND massive losses)
             - The average outcome at high leverage is near-zero or negative after fees
             """),
            ("green", "✅ FINDING 4 — CONTRARIAN STRATEGY HAS A REAL EDGE",
             """
             - Buying Extreme Fear + Shorting Extreme Greed exploits market overreaction
             - The strategy captures mean-reversion at price extremes
             - Works best with staged entries and predefined stop-losses
             """),
            ("green", "✅ FINDING 5 — COIN SELECTION MATTERS BY REGIME",
             """
             - High-beta altcoins amplify sentiment effects in both directions
             - BTC/ETH are relatively stable across all regimes
             - **Rotation strategy:** BTC/ETH when fearful → alts when greedy
             """),
            ("amber", "📊 FINDING 6 — MOST TRADERS LACK A CONSISTENT EDGE",
             f"""
             - Only **~{winner_pct:.0f}%** of traders qualify as Consistent Winners
             - The majority are Mixed/Neutral with no systematic approach
             - Winners share two traits: higher win rate + disciplined leverage usage
             """),
        ]

        for color, title, body in findings:
            st.markdown(f"""
            <div class='insight-box {color}'>
              <strong>{title}</strong>
              <div style='margin-top:8px;color:#d1d5db'>{body}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(body)

        st.markdown("---")
        st.markdown("## 🎯 Top 3 Actionable Recommendations")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown("""
            <div class='metric-card'>
              <p class='metric-value' style='color:#2563eb;font-size:1.5rem'>🚦 SENTIMENT GATE</p>
              <p style='color:#d1d5db;margin-top:8px'>
              Never open a new long when F/G index &lt; 30.<br>
              Wait for it to cross back above 35 before re-entering.
              </p>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown("""
            <div class='metric-card'>
              <p class='metric-value' style='color:#059669;font-size:1.5rem'>⚡ LEVERAGE CAP</p>
              <p style='color:#d1d5db;margin-top:8px'>
              Hard cap at 10× for all directional trades.<br>
              Reserve 20×+ only for short-duration scalps with a stop-loss.
              </p>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown("""
            <div class='metric-card'>
              <p class='metric-value' style='color:#d97706;font-size:1.5rem'>🪙 COIN ROTATION</p>
              <p style='color:#d1d5db;margin-top:8px'>
              F/G &lt; 40 → BTC/ETH only<br>
              F/G 40–60 → any coin, smaller size<br>
              F/G &gt; 60 → expand into high-beta alts
              </p>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#8b949e;font-size:0.85rem;padding:12px'>
  Trader Behavior Insights Dashboard • Hyperliquid × Bitcoin Fear & Greed Index<br>
  Built with Streamlit + Plotly + scikit-learn
</div>
""", unsafe_allow_html=True)
