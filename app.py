
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from marketlens import analytics, config

# Page configuration
st.set_page_config(
    page_title="MarketLens",
    page_icon="📈",
    layout="wide"
)

# Visual theme. One design system: tokens first, then components.
# Data calculations below are unaffected by anything in this block.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
  color-scheme: dark;
  --bg:#08090F; --surface:#10121B; --surface-2:#151826; --line:#23273A; --line-strong:#353A52;
  --text:#EEEDF7; --muted:#A6A9C0; --subtle:#8084A0;
  --violet:#9B8AFB; --violet-deep:#5B4BC4; --teal:#48D5C0; --amber:#E9B976; --rose:#E68D9B;
  --radius:16px; --ui:'Inter',system-ui,sans-serif; --mono:'JetBrains Mono',ui-monospace,monospace;
}

/* Canvas */
.stApp {
  background:
    radial-gradient(1100px 520px at 78% -12%, rgba(155,138,251,.16), transparent 62%),
    radial-gradient(800px 420px at 8% 4%, rgba(72,213,192,.08), transparent 60%),
    var(--bg);
  color:var(--text); font-family:var(--ui);
}
.stApp, .stApp p, .stApp label, .stApp li, .stApp span, .stApp input, .stApp button {font-family:var(--ui);}
[data-testid="stIconMaterial"], .material-symbols-rounded {font-family:'Material Symbols Rounded'!important;}
[data-testid="stHeader"] {background:transparent; height:2.25rem;}
[data-testid="stDecoration"], [data-testid="stHeaderActionElements"] {display:none;}
.block-container {max-width:1480px; padding:2.2rem 2.8rem 4rem;}
::selection {background:rgba(155,138,251,.35);}

/* Sidebar */
[data-testid="stSidebar"] {background:rgba(14,16,25,.92); border-right:1px solid var(--line); min-width:232px!important; max-width:232px!important; backdrop-filter:blur(14px);}
[data-testid="stSidebarUserContent"] {padding:1.6rem .9rem!important;}
.brand {display:flex; align-items:center; gap:.65rem; padding:0 .35rem 1.2rem; border-bottom:1px solid var(--line); margin-bottom:1.2rem;}
.brand-mark {width:34px; height:34px; border-radius:10px; display:grid; place-items:center; flex-shrink:0;
  background:conic-gradient(from 210deg, var(--violet), var(--teal), var(--violet)); box-shadow:0 0 24px rgba(155,138,251,.35);}
.brand-mark::after {content:""; width:12px; height:12px; border-radius:50%; background:var(--bg); box-shadow:0 0 0 3px rgba(8,9,15,.4);}
.brand-name {font:800 1.02rem var(--ui); letter-spacing:-.02em; color:var(--text); line-height:1.1;}
.brand-sub {font:500 .62rem var(--mono); letter-spacing:.08em; color:var(--subtle); margin-top:.2rem;}
.nav-title {font:600 .62rem var(--mono); letter-spacing:.16em; color:var(--subtle); margin:0 .4rem .55rem;}
[data-testid="stSidebar"] .st-key-nav_panel [data-testid="stVerticalBlock"] {gap:.2rem!important;}
[data-testid="stSidebar"] .st-key-nav_panel [data-testid="stElementContainer"] {margin:0!important;}
[data-testid="stSidebar"] .st-key-nav_panel .stButton button {
  position:relative; width:100%; justify-content:flex-start; min-height:40px; padding:.4rem .8rem!important;
  border:1px solid transparent!important; border-radius:11px!important; background:transparent!important;
  color:var(--muted)!important; font-size:.86rem!important; font-weight:550; transition:background .18s, color .18s, transform .18s;
}
[data-testid="stSidebar"] .st-key-nav_panel .stButton button p {white-space:nowrap!important; font-weight:550;}
[data-testid="stSidebar"] .st-key-nav_panel .stButton button > div {width:100%; justify-content:flex-start!important; gap:.6rem;}
[data-testid="stSidebar"] .st-key-nav_panel .stButton button:hover {background:var(--surface-2)!important; color:var(--text)!important; transform:translateX(2px);}
[data-testid="stSidebar"] .st-key-nav_panel .stButton button[kind="primary"] {
  background:linear-gradient(90deg, rgba(155,138,251,.20), rgba(155,138,251,.04))!important;
  border-color:rgba(155,138,251,.35)!important; color:#F1EDFF!important;
}
[data-testid="stSidebar"] .st-key-nav_panel .stButton button[kind="primary"]::before {
  content:""; position:absolute; left:-1px; top:22%; bottom:22%; width:3px; border-radius:3px;
  background:linear-gradient(var(--violet), var(--teal)); box-shadow:0 0 12px var(--violet);
}
[data-testid="stSidebar"] .st-key-nav_panel .stButton button[kind="primary"] [data-testid="stIconMaterial"] {color:var(--violet);}
.side-foot {border-top:1px solid var(--line); padding:1rem .4rem 0; margin-top:1.2rem; font:500 .6rem/1.8 var(--mono); color:var(--subtle); letter-spacing:.06em;}
.side-foot .live {display:inline-flex; align-items:center; gap:.45rem; color:var(--teal); margin-bottom:.35rem;}
.side-foot .live::before {content:""; width:6px; height:6px; border-radius:50%; background:var(--teal); box-shadow:0 0 10px var(--teal); animation:pulse 2.4s ease-in-out infinite;}

/* Page hero */
.hero {animation:rise .55s cubic-bezier(.2,.7,.2,1) both; margin-bottom:1.3rem;}
.kicker {display:inline-flex; align-items:center; gap:.55rem; font:600 .66rem var(--mono); letter-spacing:.16em; color:var(--violet);
  padding:.32rem .7rem; border:1px solid rgba(155,138,251,.28); border-radius:999px; background:rgba(155,138,251,.07);}
.kicker::before {content:""; width:6px; height:6px; border-radius:50%; background:var(--violet); box-shadow:0 0 10px var(--violet);}
.hero-title {font:800 clamp(2.2rem,1.4rem + 2.4vw,3.4rem)/1.02 var(--ui)!important; letter-spacing:-.045em!important; margin:.9rem 0 .55rem!important; padding:0!important;
  background:linear-gradient(100deg,#FFFFFF 10%,#D9D2FF 55%,#8FE9DA 100%); -webkit-background-clip:text; background-clip:text; color:transparent!important;}
.hero-sub {color:var(--muted); font-size:1rem; max-width:64ch; margin:0;}

/* Section labels */
.section-label {display:flex; align-items:center; gap:.75rem; font:600 .68rem var(--mono); letter-spacing:.16em; color:#C9C5E4; margin:2.1rem 0 .9rem;}
.section-label::after {content:""; flex:1; height:1px; background:linear-gradient(90deg, var(--line-strong), transparent);}

/* Universe strip */
.strip {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; margin:1.6rem 0 .4rem; border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; background:var(--line); animation:rise .6s .08s cubic-bezier(.2,.7,.2,1) both;}
.strip > div {background:linear-gradient(180deg, var(--surface-2), var(--surface)); padding:1rem 1.2rem;}
.strip .k {font:600 .6rem var(--mono); letter-spacing:.14em; color:var(--subtle);}
.strip .v {font:700 1.35rem var(--ui); letter-spacing:-.03em; color:var(--text); margin-top:.35rem; font-variant-numeric:tabular-nums;}
.strip .v small {font:500 .78rem var(--ui); color:var(--muted); letter-spacing:0; margin-left:.35rem;}
.strip .up {color:var(--teal);}
@media(max-width:900px) {.strip {grid-template-columns:repeat(2,minmax(0,1fr));}}

/* Metrics */
[data-testid="stMetric"] {
  position:relative; overflow:hidden; background:linear-gradient(180deg, var(--surface-2), var(--surface));
  border:1px solid var(--line); border-radius:var(--radius); padding:16px 18px; min-height:96px;
  transition:border-color .2s, transform .2s, box-shadow .2s;
}
[data-testid="stMetric"]::before {content:""; position:absolute; inset:0 0 auto 0; height:1px; background:linear-gradient(90deg, transparent, rgba(155,138,251,.7), transparent);}
[data-testid="stMetric"]:hover {border-color:var(--line-strong); transform:translateY(-2px); box-shadow:0 14px 34px -18px rgba(155,138,251,.45);}
[data-testid="stMetricLabel"] p {color:var(--muted)!important; font-size:.8rem!important; font-weight:550;}
[data-testid="stMetricValue"] {color:var(--text); font:700 clamp(1.4rem,1.6vw,1.85rem) var(--ui); letter-spacing:-.035em; font-variant-numeric:tabular-nums;}
[data-testid="stMetricDelta"] {color:var(--subtle)!important;}

/* Charts, tables, inputs */
[data-testid="stPlotlyChart"] {border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; background:linear-gradient(180deg, var(--surface-2), var(--surface)); padding:4px;}
[data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:12px; overflow:hidden;}
[data-testid="stExpander"] details {border:1px solid var(--line)!important; border-radius:12px!important; background:var(--surface);}
div[data-baseweb="select"] > div {background:var(--surface-2)!important; border:1px solid var(--line-strong)!important; border-radius:11px!important; min-height:44px; transition:border-color .2s;}
div[data-baseweb="select"] > div:hover {border-color:var(--violet)!important;}
div[data-baseweb="select"] * {color:var(--text);}
span[data-baseweb="tag"] {background:rgba(155,138,251,.18)!important; border:1px solid rgba(155,138,251,.35); border-radius:8px!important;}
[data-testid="stSegmentedControl"] button {background:var(--surface-2)!important; border-color:var(--line-strong)!important; color:var(--muted)!important; font-family:var(--mono)!important; font-size:.75rem!important;}
[data-testid="stSegmentedControl"] button[aria-pressed="true"], [data-testid="stSegmentedControl"] button[data-selected="true"] {
  background:linear-gradient(180deg, rgba(155,138,251,.35), rgba(155,138,251,.18))!important; border-color:var(--violet)!important; color:#fff!important;}
[data-testid="stSlider"] [role="slider"] {background:var(--violet)!important; border-color:var(--violet)!important; box-shadow:0 0 0 5px rgba(155,138,251,.18);}
[data-testid="stCaptionContainer"] {color:var(--subtle); font-size:.84rem;}
[data-testid="stAlert"] {border-radius:12px; border:1px solid var(--line-strong); background:var(--surface-2);}
hr {border-color:var(--line);}

@keyframes rise {from {opacity:0; transform:translateY(12px);} to {opacity:1; transform:none;}}
@keyframes pulse {0%,100% {opacity:1;} 50% {opacity:.35;}}
@media (prefers-reduced-motion: reduce) {*, *::before {animation:none!important; transition:none!important;}}
@media(max-width:900px) {.block-container {padding:1.2rem 1rem 3rem;}}
</style>
""", unsafe_allow_html=True)

CHART_COLORS = ["#9B8AFB", "#48D5C0", "#E9B976", "#79A9F1", "#E68D9B", "#92D1A7"]
GRID = "rgba(255,255,255,0.06)"


def style_chart(fig, height=410):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#A6A9C0", size=12),
        title_font=dict(family="Inter, system-ui, sans-serif", color="#EEEDF7", size=15),
        title_x=0.015,
        height=height,
        margin=dict(l=50, r=24, t=56, b=40),
        colorway=CHART_COLORS,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        legend=dict(title_text="", bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#1B1E2E", bordercolor="#353A52",
                        font=dict(family="Inter, system-ui, sans-serif", color="#EEEDF7")),
    )
    return fig


def gradient_area(fig, x, y, color="#9B8AFB", name="", hover=None, width=2.6):
    """Line with a soft vertical gradient fill that fades out above the series minimum."""
    floor = float(min(y)) * 0.97 if min(y) > 0 else float(min(y)) * 1.05
    fig.add_trace(go.Scatter(x=x, y=[floor] * len(x), mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name, fill="tonexty",
        line=dict(color=color, width=width, shape="spline", smoothing=0.3),
        fillgradient=dict(type="vertical", colorscale=[[0, f"rgba({r},{g},{b},0)"],
                                                         [1, f"rgba({r},{g},{b},0.28)"]]),
        hovertemplate=hover,
    ))
    return fig


def hero(kicker, title, subtitle):
    st.markdown(
        f'<div class="hero"><div class="kicker">{kicker}</div>'
        f'<h1 class="hero-title">{title}</h1><p class="hero-sub">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


# Data loads once per server and is shared across visitors and reruns.
# Rebuild it with `python -m marketlens all`.
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(config.DATASET_FILE, parse_dates=["Date"])


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    return pd.read_csv(config.PRICES_FILE, parse_dates=["Date"])


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(config.ROOT / path)


@st.cache_data(show_spinner=False)
def load_holdout() -> dict | None:
    path = config.RESULTS_DIR / "holdout" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


data = load_dataset()
prices_all = load_prices()
results = load_csv("results/final_model_comparison.csv")

# Button navigation with Material icons (no radio circles).
page_names = ["Overview", "ETF Explorer", "Portfolio Simulator", "Diversification", "ML Research",
              "Test Results", "Risk Analysis"]
page_icons = [":material/space_dashboard:", ":material/stacked_line_chart:",
              ":material/donut_large:", ":material/hub:", ":material/model_training:",
              ":material/verified:", ":material/shield:"]
nav_labels = ["Overview", "ETF Explorer", "Portfolio", "Diversification", "ML Research",
              "2026 Test", "Risk Analysis"]
if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark"></div><div>'
        '<div class="brand-name">MarketLens</div>'
        '<div class="brand-sub">ETF RISK RESEARCH</div></div></div>'
        '<div class="nav-title">EXPLORE</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="nav_panel"):
        for icon, name, label in zip(page_icons, page_names, nav_labels, strict=True):
            if st.button(
                label,
                icon=icon,
                key=f"nav_{name}",
                type="primary" if st.session_state.current_page == name else "secondary",
                width="stretch",
            ):
                st.session_state.current_page = name
                st.rerun()
    last_day = prices_all["Date"].max()
    st.markdown(
        f'<div class="side-foot"><span class="live">DATA TO {last_day:%b %d, %Y}'.replace(
            last_day.strftime("%b"), last_day.strftime("%b").upper())
        + '</span><br>COMP 4112 · DATA SCIENCE<br>ACADEMIC RESEARCH PROJECT<br>'
        'NOT INVESTMENT ADVICE</div>',
        unsafe_allow_html=True,
    )

page = st.session_state.current_page

# Overview
if page == "Overview":
    tickers_all = sorted(data["Ticker"].unique())
    title_col, picker_col = st.columns([2, 1], gap="large", vertical_alignment="bottom")
    with title_col:
        hero("ETF INTELLIGENCE", "Market Overview",
             "Historical performance and downside risk across 13 Canadian-listed ETFs.")
    with picker_col:
        selected = st.selectbox("ETF", tickers_all, index=tickers_all.index("XEQT.TO"))

    # Universe strip: facts about the whole dataset, computed from the saved prices.
    universe = analytics.summary_table(prices_all)
    best = universe.sort_values("Total_Return_%", ascending=False).iloc[0]
    first_day = prices_all["Date"].min()
    st.markdown(
        '<div class="strip">'
        f'<div><div class="k">ETFS TRACKED</div><div class="v">{prices_all["Ticker"].nunique()}'
        '<small>Canadian-listed</small></div></div>'
        f'<div><div class="k">PRICE HISTORY</div><div class="v">{first_day:%Y}–{last_day:%Y}'
        f'<small>{prices_all["Date"].nunique():,} trading days</small></div></div>'
        f'<div><div class="k">MODELLING ROWS</div><div class="v">{len(data):,}'
        '<small>ETF-days</small></div></div>'
        f'<div><div class="k">BEST TOTAL RETURN</div><div class="v up">{best["Total_Return_%"]:+.0f}%'
        f'<small>{best["ETF"]}</small></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    etf_data = prices_all[prices_all["Ticker"] == selected].sort_values("Date").copy()
    risk_data = data[data["Ticker"] == selected]
    close = etf_data.set_index("Date")["Close"]
    total_return = analytics.total_return(close)
    volatility = analytics.annualized_volatility(close)
    underwater = analytics.drawdown(close)
    max_drawdown = float(underwater.min())

    section("PRICE PERFORMANCE")
    chart_col, stats_col = st.columns([7, 3], gap="medium")
    with chart_col:
        period = st.segmented_control(
            "Chart period", ["6M", "1Y", "3Y", "5Y", "ALL"], default="ALL",
            label_visibility="collapsed",
        ) or "ALL"
        end_date = etf_data["Date"].max()
        offsets = {"6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1),
                   "3Y": pd.DateOffset(years=3), "5Y": pd.DateOffset(years=5)}
        start_date = end_date - offsets[period] if period in offsets else etf_data["Date"].min()
        chart_data = etf_data[etf_data["Date"] >= start_date]
        period_change = analytics.total_return(chart_data.set_index("Date")["Close"])

        fig = gradient_area(go.Figure(), chart_data["Date"], chart_data["Close"].tolist(),
                            hover="%{x|%b %d, %Y}<br><b>$%{y:.2f}</b><extra></extra>")
        fig = style_chart(fig, height=440)
        fig.update_layout(
            title=dict(text=f"{selected}  <span style='color:{'#48D5C0' if period_change >= 0 else '#E68D9B'};"
                            f"font-size:13px'>{period_change:+.1f}% · {period}</span>"),
            xaxis_title="", yaxis_title="Adjusted price (CAD)", showlegend=False,
            hovermode="x unified", xaxis=dict(showgrid=False),
            yaxis=dict(tickprefix="$"),
        )
        st.plotly_chart(fig, width="stretch")

    with stats_col:
        st.metric("Total return", f"{total_return:+.1f}%")
        st.metric("Annualized volatility", f"{volatility:.1f}%")
        st.metric("Maximum drawdown", f"{max_drawdown:.1f}%")
        st.metric("Below peak today", f"{float(underwater.iloc[-1]):.1f}%")
        st.caption(f"Full history, {close.index[0]:%b %Y} to {end_date:%b %d, %Y}")

    section("DRAWDOWN · DISTANCE FROM PREVIOUS PEAK")
    dd = go.Figure(go.Scatter(
        x=underwater.index, y=underwater.values, mode="lines", fill="tozeroy",
        line=dict(color="#E68D9B", width=1.6),
        fillgradient=dict(type="vertical", colorscale=[[0, "rgba(230,141,155,0.38)"],
                                                         [1, "rgba(230,141,155,0.02)"]]),
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f}% below peak<extra></extra>",
    ))
    dd = style_chart(dd, height=230)
    dd.update_layout(margin=dict(l=50, r=24, t=16, b=32), yaxis=dict(ticksuffix="%"),
                     xaxis=dict(showgrid=False))
    st.plotly_chart(dd, width="stretch")

    section("DOWNSIDE EVENTS · HOW OFTEN THE NEXT 5 DAYS FELL")
    risk_left, risk_right = st.columns(2, gap="medium")
    risk_left.metric("Fixed: fell 3% or more", f"{risk_data['Fixed_Risk'].mean() * 100:.2f}% of days")
    risk_right.metric("Volatility-adjusted: fell beyond 1.5σ",
                      f"{risk_data['Adjusted_Risk'].mean() * 100:.2f}% of days")
    st.caption("Observed historical frequencies, not predicted probabilities.")

    with st.expander("About the dataset"):
        st.write(f"{len(data):,} prepared observations across {data['Ticker'].nunique()} ETFs.")
        st.write(f"{results['Model'].nunique()} model types are included in the evaluation.")

# ETF Explorer
elif page == "ETF Explorer":
    hero("COMPARATIVE ANALYTICS", "ETF Explorer",
         "Compare Canadian-listed ETFs over the same historical trading dates.")

    selected = st.multiselect(
        "ETFs to compare", sorted(data["Ticker"].unique()),
        default=["VFV.TO", "XEQT.TO", "ZNQ.TO"], max_selections=6,
    )
    if not selected:
        st.info("Select at least one ETF to view the comparison.")
    else:
        table = analytics.price_table(prices_all, selected)
        if len(table) < 2:
            st.warning("The selected ETFs do not have enough overlapping trading dates.")
        else:
            growth = table.div(table.iloc[0]).mul(100)
            long_growth = growth.rename_axis("Date").reset_index().melt(
                id_vars="Date", var_name="ETF", value_name="Growth")
            chart_col, info_col = st.columns([7, 3], gap="medium")
            with chart_col:
                section("GROWTH OF $100 · COMMON PERIOD")
                fig = px.line(long_growth, x="Date", y="Growth", color="ETF",
                              color_discrete_sequence=CHART_COLORS)
                fig.update_traces(line=dict(width=2.5),
                                  hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra>%{fullData.name}</extra>")
                fig = style_chart(fig, height=455)
                fig.update_layout(title="Historical relative performance", xaxis_title="",
                                  yaxis_title="Value of $100 (CAD)", hovermode="x unified")
                st.plotly_chart(fig, width="stretch")
            with info_col:
                section("COMPARISON SNAPSHOT")
                st.metric("ETFs compared", len(selected))
                st.metric("Shared trading days", f"{len(table):,}")
                st.metric("Comparison span", f"{(table.index[-1]-table.index[0]).days:,} days")
                st.caption(f"{table.index[0]:%b %d, %Y} – {table.index[-1]:%b %d, %Y}")

            comparison = []
            for ticker in table.columns:
                series = table[ticker]
                comparison.append({
                    "ETF": ticker,
                    "Total Return (%)": round(analytics.total_return(series), 2),
                    "Annualized Volatility (%)": round(analytics.annualized_volatility(series), 2),
                    "Maximum Drawdown (%)": round(analytics.max_drawdown(series), 2),
                })
            section("PERFORMANCE COMPARISON")
            st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
            st.caption("All ETFs are normalized to $100 on their first shared date. "
                       "Results are historical and exclude fees, taxes, and trading costs.")


# Portfolio Simulator
elif page == "Portfolio Simulator":
    hero("PORTFOLIO RESEARCH", "Portfolio Simulator",
         "Explore a hypothetical buy-and-hold ETF allocation using historical adjusted prices.")

    left, right = st.columns([7, 3], gap="large")
    with left:
        holdings = st.multiselect("Portfolio ETFs", sorted(data["Ticker"].unique()),
                                  default=["XEQT.TO", "VFV.TO", "CGL.TO"], max_selections=5)
    with right:
        benchmark = st.selectbox("Benchmark", sorted(data["Ticker"].unique()),
                                 index=sorted(data["Ticker"].unique()).index("XEQT.TO"))

    if not holdings:
        st.info("Choose at least one ETF to simulate a portfolio.")
    else:
        section("PORTFOLIO ALLOCATION")
        st.caption("Set relative weights below. They are automatically normalized to total 100%.")
        allocation_cols = st.columns(len(holdings))
        raw_weights = {}
        for col, ticker in zip(allocation_cols, holdings, strict=True):
            with col:
                raw_weights[ticker] = st.slider(ticker, min_value=0, max_value=100,
                                                value=50, step=5, key=f"weight_{ticker}")
        total_weight = sum(raw_weights.values())
        if total_weight == 0:
            st.warning("At least one ETF needs a nonzero weight.")
        else:
            weights = pd.Series({k: v / total_weight for k, v in raw_weights.items()})
            table = analytics.price_table(prices_all, sorted(set(holdings + [benchmark])))
            if len(table) < 2:
                st.warning("Not enough overlapping dates for the selected holdings and benchmark.")
            else:
                # Buy-and-hold: allocate $100 at the start and let each position drift.
                portfolio = analytics.buy_and_hold(table, weights.to_dict())
                benchmark_growth = table[benchmark].div(table[benchmark].iloc[0]) * 100
                port_return = analytics.total_return(portfolio)
                port_vol = analytics.annualized_volatility(portfolio)
                port_dd = analytics.max_drawdown(portfolio)

                section("HISTORICAL PORTFOLIO PERFORMANCE")
                chart_col, stats_col = st.columns([7, 3], gap="medium")
                with chart_col:
                    chart = pd.DataFrame({"Portfolio": portfolio,
                                          f"Benchmark · {benchmark}": benchmark_growth})
                    long_chart = chart.rename_axis("Date").reset_index().melt(
                        id_vars="Date", var_name="Series", value_name="Value")
                    fig = px.line(long_chart, x="Date", y="Value", color="Series",
                                  color_discrete_sequence=["#9B8AFB", "#48D5C0"])
                    fig.update_traces(line=dict(width=2.7))
                    fig = style_chart(fig, height=445)
                    fig.update_layout(title="Growth of an initial $100", xaxis_title="",
                                      yaxis_title="Portfolio value (CAD)", hovermode="x unified")
                    st.plotly_chart(fig, width="stretch")
                with stats_col:
                    section("PORTFOLIO SUMMARY")
                    st.metric("Historical Return", f"{port_return:+.2f}%")
                    st.metric("Annualized Volatility", f"{port_vol:.2f}%")
                    st.metric("Maximum Drawdown", f"{port_dd:.2f}%")

                details_col, allocation_col = st.columns([7, 3], gap="medium")
                with details_col:
                    section("PORTFOLIO VS BENCHMARK")
                    comparison = pd.DataFrame([
                        {"Series": "Portfolio", "Total Return (%)": port_return,
                         "Annualized Volatility (%)": port_vol, "Maximum Drawdown (%)": port_dd},
                        {"Series": benchmark,
                         "Total Return (%)": analytics.total_return(benchmark_growth),
                         "Annualized Volatility (%)": analytics.annualized_volatility(benchmark_growth),
                         "Maximum Drawdown (%)": analytics.max_drawdown(benchmark_growth)},
                    ])
                    st.dataframe(comparison.round(2), hide_index=True, width="stretch")
                with allocation_col:
                    section("ALLOCATION")
                    pie = px.pie(names=weights.index, values=weights.values,
                                 color_discrete_sequence=CHART_COLORS, hole=.6)
                    pie = style_chart(pie, height=265)
                    pie.update_layout(showlegend=True, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(pie, width="stretch")
                st.caption(f"Shared history: {table.index[0]:%Y-%m-%d} to {table.index[-1]:%Y-%m-%d}. "
                           "Weights represent initial allocations without rebalancing. "
                           "This is a historical illustration, not a forecast; fees, taxes, "
                           "and trading costs are excluded.")


# Diversification
elif page == "Diversification":
    hero("PORTFOLIO INTELLIGENCE", "Diversification",
         "Explore how ETF daily returns moved together over the available historical period.")

    # Correlations use overlapping observed daily returns, not future risk labels.
    returns = (data.pivot(index="Date", columns="Ticker", values="Return_1D")
               .sort_index())
    correlation = returns.corr()
    tickers = sorted(correlation.columns)

    chart_col, insight_col = st.columns([7, 3], gap="medium")
    with chart_col:
        section("ETF RETURN CORRELATION")
        fig = px.imshow(
            correlation.loc[tickers, tickers],
            text_auto=".2f",
            color_continuous_scale=[
                [0.0, "#48D5C0"], [0.5, "#1A1D2B"], [1.0, "#9B8AFB"]
            ],
            zmin=-1, zmax=1,
            aspect="auto",
            labels=dict(x="ETF", y="ETF", color="Correlation"),
        )
        fig = style_chart(fig, height=555)
        fig.update_layout(
            title="Daily return correlation matrix",
            margin=dict(l=45, r=25, t=55, b=40),
            xaxis=dict(side="bottom", tickangle=-40, showgrid=False),
            yaxis=dict(showgrid=False, autorange="reversed"),
            coloraxis_colorbar=dict(title="r", thickness=12, len=0.75),
        )
        fig.update_traces(
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>"
        )
        st.plotly_chart(fig, width="stretch")

    with insight_col:
        section("EXPLORE AN ETF PAIR")
        first = st.selectbox(
            "First ETF", tickers,
            index=tickers.index("XEQT.TO") if "XEQT.TO" in tickers else 0,
            key="div_first",
        )
        second = st.selectbox(
            "Second ETF", tickers,
            index=tickers.index("VFV.TO") if "VFV.TO" in tickers else min(1, len(tickers)-1),
            key="div_second",
        )
        pair_corr = correlation.loc[first, second]
        overlap = returns[[first, second]].dropna().shape[0]
        st.metric("Daily Return Correlation", f"{pair_corr:+.2f}" if pd.notna(pair_corr) else "N/A")
        st.metric("Shared Trading Days", f"{overlap:,}")
        if first == second:
            st.info("Select two different ETFs to compare their co-movement.")
        elif pd.isna(pair_corr):
            st.info("Not enough shared daily returns to calculate correlation.")
        elif pair_corr >= 0.7:
            st.caption("Strong positive co-movement: these ETFs often move in the same direction.")
        elif pair_corr >= 0.3:
            st.caption("Moderate positive co-movement: some shared movement, but not identical.")
        elif pair_corr > -0.3:
            st.caption("Weak co-movement: their daily returns have limited linear association.")
        else:
            st.caption("Negative co-movement: their daily returns have tended to move in opposite directions.")
        section("HOW TO READ THE MATRIX")
        st.caption("+1 means returns move together; 0 means little linear relationship; "
                   "−1 means they move in opposite directions. Correlation is historical "
                   "and can change over time.")

    section("CORRELATION DETAILS")
    with st.expander("View correlation table"):
        st.dataframe(correlation.loc[tickers, tickers].round(2), width="stretch")
    st.caption("Calculated from overlapping historical daily ETF returns. "
               "Correlation alone does not establish diversification benefits or predict future performance.")


# ML Research
elif page == "ML Research":
    hero("MODEL EVALUATION", "ML Research",
         "Downside-risk classification · 2025 validation data · interactive decision thresholds")

    # All three controls sit together, rather than stacking vertically.
    risk_col, model_col, threshold_col = st.columns([3, 3, 4], gap="medium")
    with risk_col:
        target = st.selectbox("Risk definition", ["Fixed_Risk", "Adjusted_Risk"])
    with model_col:
        model_name = st.selectbox(
            "Classification model", ["Logistic Regression", "Random Forest", "Neural Network"]
        )
    with threshold_col:
        threshold = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.05)

    model_id = target + "_" + model_name.replace(" ", "_")
    predictions = load_csv(f"results/ml_research/{model_id}_predictions.csv")
    curve = load_csv(f"results/ml_research/{model_id}_curve.csv")

    actual = predictions["Actual"].astype(int)
    predicted = (predictions["Probability"] >= threshold).astype(int)
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)
    cm = confusion_matrix(actual, predicted, labels=[0, 1])

    section("VALIDATION PERFORMANCE")
    precision_col, recall_col, f1_col = st.columns(3, gap="medium")
    precision_col.metric("Precision", f"{precision:.3f}")
    recall_col.metric("Recall", f"{recall:.3f}")
    f1_col.metric("F1 Score", f"{f1:.3f}")

    section("CLASSIFICATION DIAGNOSTICS")
    matrix_col, curve_col = st.columns(2, gap="medium")
    with matrix_col:
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=["Predicted low", "Predicted high"],
            y=["Actual low", "Actual high"],
            color_continuous_scale=[[0, "#151826"], [0.5, "#4A3E8C"], [1, "#9B8AFB"]],
            title="Confusion matrix",
            aspect="auto",
        )
        fig_cm = style_chart(fig_cm, height=385)
        fig_cm.update_layout(coloraxis_showscale=False, margin=dict(l=35, r=20, t=55, b=45))
        fig_cm.update_xaxes(side="bottom", title=None)
        fig_cm.update_yaxes(title=None)
        st.plotly_chart(fig_cm, width="stretch")

    with curve_col:
        fig_pr = px.line(curve, x="Recall", y="Precision", title="Precision–recall curve")
        fig_pr.update_traces(line_color="#9B8AFB", line_width=3, fill="tozeroy",
                             fillcolor="rgba(155,138,251,0.12)")
        fig_pr.add_hline(y=actual.mean(), line_dash="dash", line_color="#7F9BAF",
                         annotation_text="Random guessing", annotation_position="bottom left")
        fig_pr = style_chart(fig_pr, height=385)
        fig_pr.update_layout(
            xaxis=dict(range=[0, 1], gridcolor="#2A3042"),
            yaxis=dict(range=[0, 1], gridcolor="#2A3042"),
            margin=dict(l=45, r=25, t=55, b=45),
        )
        st.plotly_chart(fig_pr, width="stretch")

    section("MODEL INTERPRETATION")
    positive_count = int(actual.sum())
    positive_rate = actual.mean() * 100
    predicted_positive = int(predicted.sum())
    st.write(
        f"The 2025 validation set contains **{len(actual):,} observations**, "
        f"including **{positive_count:,} downside events ({positive_rate:.1f}%)**. "
        f"At the **{threshold:.2f}** threshold, **{model_name}** flags "
        f"**{predicted_positive:,} observations** as high risk."
    )
    if predicted_positive == 0:
        st.info(
            "No observations are classified as high risk at this threshold. "
            "Precision, recall, and F1 are therefore reported as zero. "
            "Try lowering the threshold to explore the precision–recall trade-off."
        )
    elif recall < 0.2:
        st.info(
            "This threshold misses most observed downside events (low recall). "
            "Higher overall accuracy can still be misleading when downside events are uncommon."
        )
    else:
        st.info(
            "The threshold changes how many events are flagged. "
            "Increasing recall can also increase false positives; compare both metrics before drawing conclusions."
        )
    st.caption(
        "Metrics are recalculated from saved 2025 validation probabilities; this page does not retrain models. "
        "The precision–recall curve shows performance across thresholds; the dashed line is the "
        "event rate, which is what random guessing would score. The untouched 2026 test is on the "
        "Test Results page."
    )

# Test Results: the single, untouched 2026 evaluation
elif page == "Test Results":
    hero("FINAL EVALUATION", "2026 Test Results",
         "One evaluation on data the models never saw, with every choice fixed in advance on 2025.")

    holdout = load_holdout()
    if holdout is None:
        st.info("No test results yet. Run `python -m marketlens all` to produce them.")
    else:
        section("HOW THE TEST WAS RUN")
        train_period, test_period = holdout["periods"]["train"], holdout["periods"]["test"]
        st.write(
            f"1. Every model was trained on **{train_period[0]} to {train_period[1]}**.  \n"
            "2. On **2025**, the model with the best average precision was chosen, then the "
            "decision threshold with the best F1.  \n"
            f"3. Both choices were frozen and scored once on **{test_period[0]} to {test_period[1]}**."
        )

        rules = {r["risk_definition"]: r for r in holdout.get("volatility_rule", [])}
        for result in holdout["results"]:
            rule = rules.get(result["risk_definition"])
            label = ("Fixed 3% drop" if result["risk_definition"] == "Fixed_Risk"
                     else "Volatility-adjusted drop")
            section(f'{label.upper()} · {result["model"].upper()}')
            c1, c2, c3, c4 = st.columns(4, gap="medium")
            c1.metric("Average precision", f"{result['test_average_precision']:.3f}",
                      f"{result['test_lift']:.2f}× random guessing", delta_color="off", delta_arrow="off")
            c2.metric("Random guessing", f"{result['test_base_rate']:.3f}",
                      f"{result['test_events']:,} of {result['test_observations']:,} were drops",
                      delta_color="off", delta_arrow="off")
            c3.metric(f"Recall @ {result['threshold']:.2f}", f"{result['test_recall']:.1%}",
                      "share of drops flagged", delta_color="off", delta_arrow="off")
            c4.metric(f"Precision @ {result['threshold']:.2f}", f"{result['test_precision']:.1%}",
                      "share of flags that were drops", delta_color="off", delta_arrow="off")

            curve_col, side_col = st.columns([7, 3], gap="medium")
            with curve_col:
                curve = load_csv(f"results/holdout/{result['risk_definition']}_test_curve.csv")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=curve["Recall"], y=curve["Precision"], mode="lines",
                                         name=result["model"], line=dict(color="#9B8AFB", width=3),
                                         fill="tozeroy", fillcolor="rgba(155,138,251,0.12)"))
                if rule:
                    rule_curve = load_csv(
                        f"results/holdout/{result['risk_definition']}_volatility_rule_curve.csv")
                    fig.add_trace(go.Scatter(x=rule_curve["Recall"], y=rule_curve["Precision"],
                                             mode="lines", name="Volatility rule (no model)",
                                             line=dict(color="#E9B976", width=2, dash="dot")))
                fig.add_hline(y=result["test_base_rate"], line_dash="dash", line_color="#7F9BAF",
                              annotation_text="Random guessing", annotation_position="bottom left")
                fig = style_chart(fig, height=360)
                fig.update_layout(title="Precision–recall on 2026", xaxis_title="Recall",
                                  yaxis_title="Precision", xaxis=dict(range=[0, 1]),
                                  yaxis=dict(range=[0, 1]), showlegend=bool(rule))
                st.plotly_chart(fig, width="stretch")
            with side_col:
                importance_path = (config.RESULTS_DIR / "holdout"
                                   / f"{result['risk_definition']}_feature_importance.csv")
                if importance_path.exists():
                    importance = load_csv(str(importance_path.relative_to(config.ROOT)))
                    fig = px.bar(importance.sort_values("Importance"), x="Importance", y="Feature",
                                 orientation="h", color_discrete_sequence=["#48D5C0"])
                    fig = style_chart(fig, height=360)
                    fig.update_layout(title="What the forest relies on", xaxis_title="",
                                      yaxis_title="")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.caption(f"Validation AP {result['validation_average_precision']:.3f} "
                               f"vs random {result['validation_base_rate']:.3f} on 2025.")

        if rules:
            section("HOW MUCH OF THIS IS JUST VOLATILITY?")
            st.write(
                "A model-free check, added **after** the 2026 test had been seen: rank every day by "
                "its 20-day volatility and nothing else. If the models were finding something "
                "beyond \"volatile ETFs fall more often\", they should beat this rule."
            )
            rows = []
            for result in holdout["results"]:
                rule = rules.get(result["risk_definition"])
                if not rule:
                    continue
                name = "Fixed 3%" if result["risk_definition"] == "Fixed_Risk" else "Vol-adjusted"
                for period, base, model_ap, rule_ap in [
                    ("2025", result["validation_base_rate"], result["validation_average_precision"],
                     rule["validation_average_precision"]),
                    ("2026", result["test_base_rate"], result["test_average_precision"],
                     rule["test_average_precision"]),
                ]:
                    group = f"{name} · {period}"
                    rows += [{"Group": group, "Ranker": "Random guessing", "AP": base},
                             {"Group": group, "Ranker": "Volatility rule (no model)", "AP": rule_ap},
                             {"Group": group, "Ranker": f"Chosen model ({result['model']})", "AP": model_ap}]
            ap_table = pd.DataFrame(rows)
            ap_table["Ranker"] = ap_table["Ranker"].str.replace(r"\(.*\)", "", regex=True).str.strip()
            fig = px.bar(ap_table, x="Group", y="AP", color="Ranker", barmode="group", text_auto=".3f",
                         color_discrete_map={"Random guessing": "#5A5F78", "Volatility rule": "#E9B976",
                                             "Chosen model": "#9B8AFB"})
            fig = style_chart(fig, height=380)
            fig.update_traces(marker_cornerradius=4, textposition="outside", textfont_size=11)
            fig.update_layout(title="Average precision: model vs volatility alone", xaxis_title="",
                              yaxis_title="Average precision", bargap=0.3)
            st.plotly_chart(fig, width="stretch")
            st.info(
                "For the fixed 3% label, ranking by volatility alone does about as well as the chosen "
                "model: a little worse on 2025, a little better on 2026. Most of what the models learn "
                "is that volatile ETFs fall 3% more often. Once the bar scales with each ETF's own "
                "volatility (the adjusted label), neither the models nor the rule beat chance by much."
            )

        section("WHAT THIS DOES AND DOES NOT SHOW")
        st.caption(
            "Average precision summarizes ranking quality across every threshold; random guessing "
            "scores the event rate. A result above that line means the model ranks risky days "
            "higher than chance did on this period. It is one test period, one market regime and "
            "13 ETFs, so it is evidence of a modest historical signal, not a trading strategy. "
            "Not investment advice."
        )

# Risk Analysis
elif page == "Risk Analysis":
    hero("DOWNSIDE RISK", "Risk Analysis",
         "How often each ETF was followed by a sharp five-day drop, under both risk definitions.")

    risk_summary = (data.groupby("Ticker")[["Fixed_Risk", "Adjusted_Risk"]].mean() * 100).reset_index()
    risk_summary = risk_summary.sort_values("Fixed_Risk")
    riskiest = risk_summary.iloc[-1]
    calmest = risk_summary.iloc[0]

    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("All ETFs · fixed 3% events", f"{data['Fixed_Risk'].mean() * 100:.2f}% of days")
    c2.metric("Most frequent", riskiest["Ticker"], f"{riskiest['Fixed_Risk']:.1f}% of days",
              delta_color="off", delta_arrow="off")
    c3.metric("Least frequent", calmest["Ticker"], f"{calmest['Fixed_Risk']:.1f}% of days",
              delta_color="off", delta_arrow="off")

    section("EVENT FREQUENCY BY ETF")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=risk_summary["Ticker"], x=risk_summary["Fixed_Risk"], orientation="h",
                         name="Fixed: fell 3% or more", marker=dict(color="#9B8AFB", cornerradius=4),
                         hovertemplate="%{y}<br>%{x:.2f}% of days<extra>Fixed 3%</extra>"))
    fig.add_trace(go.Bar(y=risk_summary["Ticker"], x=risk_summary["Adjusted_Risk"], orientation="h",
                         name="Volatility-adjusted: beyond 1.5σ",
                         marker=dict(color="#48D5C0", cornerradius=4),
                         hovertemplate="%{y}<br>%{x:.2f}% of days<extra>Adjusted</extra>"))
    fig = style_chart(fig, height=560)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08,
                      xaxis=dict(ticksuffix="%", title="Share of trading days"),
                      yaxis=dict(showgrid=False, title=""))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2, gap="medium")
    with left:
        section("FIXED DEFINITION")
        st.caption("1 if the ETF's return over the next five trading days is −3% or worse. "
                   "Simple and comparable across ETFs, but volatile funds such as gold miners "
                   "hit it far more often.")
    with right:
        section("VOLATILITY-ADJUSTED DEFINITION")
        st.caption("1 if the next five-day return is below −1.5 × 20-day volatility × √5. "
                   "Scales the bar to each ETF's own recent behaviour, so event rates are "
                   "more even across funds.")
