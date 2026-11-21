
import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="MarketLens",
    page_icon="📈",
    layout="wide"
)

# Institutional terminal visual system — presentation only
st.markdown("""
<style>
/* Simple terminal theme: all changes are visual. */
:root {color-scheme: dark;}
.stApp {background:#090D12; color:#E8EDF1;}
[data-testid="stHeader"] {background:#090D12; height:2.6rem;}
[data-testid="stDecoration"] {display:none;}
.block-container {max-width:1450px; padding:1.1rem 2.1rem 3rem;}
/* Narrow sidebar with a single-line logo. */
[data-testid="stSidebar"] {background:#111820; border-right:1px solid #27323A; min-width:215px !important; max-width:215px !important;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding:1.3rem .8rem;}
[data-testid="stSidebar"] h2 {font-size:1.05rem !important; letter-spacing:.06em; white-space:nowrap;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {font-size:.65rem; color:#8495A3;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.2rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {width:100%; padding:.55rem .55rem; background:transparent; border-left:2px solid transparent; border-radius:3px;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#1D2932;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#1D3034; border-left-color:#4ED7C1;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color:#70E4CF;}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-size:.82rem; color:#D2DBE1; white-space:nowrap;}
/* Hide radio dots but keep the native Streamlit selection working. */
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {display:none !important;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child:has(input[type="radio"]) {display:none !important;}
h1,h2,h3 {color:#F2F5F7 !important; letter-spacing:-.02em;}
h1 {font-size:1.85rem !important; font-weight:700 !important;}
h2 {font-size:1.13rem !important; font-weight:650 !important;}
h3 {font-size:.98rem !important; font-weight:650 !important;}
[data-testid="stCaptionContainer"] {color:#91A2AF;}
/* Compact KPI cards rather than oversized blocks. */
[data-testid="stMetric"] {background:#151D25; border:1px solid #2C3942; border-left:3px solid #55C9B6; border-radius:5px; padding:12px 15px; min-height:82px;}
[data-testid="stMetricLabel"] p {font-size:.75rem; color:#AAB8C1;}
[data-testid="stMetricValue"] {color:#6CE0CB; font-family:Consolas,monospace; font-size:clamp(1.25rem,1.6vw,1.65rem); font-weight:600;}
[data-testid="stPlotlyChart"] {border:1px solid #2C3942; border-radius:5px; overflow:hidden;}
[data-testid="stDataFrame"] {border:1px solid #2C3942;}
div[data-baseweb="select"] > div {background:#17212A !important; border:1px solid #3A5058 !important; border-radius:4px !important; color:#EAF0F3 !important;}
div[data-baseweb="select"] * {color:#EAF0F3;}
[data-testid="stSegmentedControl"] button {background:#17212A !important; border-color:#35444D !important; color:#C1CCD3 !important;}
[data-testid="stSegmentedControl"] button[aria-pressed="true"] {background:#25544D !important; border-color:#59DCC7 !important; color:#B9FFF0 !important;}
[data-testid="stSlider"] [role="slider"] {background:#6CE0CB !important;}
hr {border-color:#2C3942;}
.terminal-eyebrow {font:500 .67rem Consolas,monospace; letter-spacing:.12em; color:#68CDBB; margin:0 0 .45rem;}
.terminal-strip {display:flex; justify-content:space-between; flex-wrap:wrap; gap:.6rem; border:1px solid #2B3941; background:#121A22; padding:.52rem .8rem; margin:0 0 1rem; border-radius:4px; font: .7rem Consolas,monospace; color:#A2B3BE;}
.terminal-strip strong {color:#70DDC9;}
.terminal-strip .right {color:#D9B57B;}
.terminal-rule {border-bottom:1px solid #2C3942; margin:.4rem 0 1rem;}
@media(max-width:900px) {.block-container {padding-left:1rem;padding-right:1rem;} [data-testid="stSidebar"] {min-width:200px !important;max-width:200px !important;}}
</style>
""", unsafe_allow_html=True)

CHART_COLORS = ["#64D7C1", "#E2AE61", "#78A6D8", "#B2A0DD", "#D98080", "#7EBC91"]

def style_chart(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131920",
        plot_bgcolor="#131920",
        font=dict(family="Inter, Arial, sans-serif", color="#BBC8D2", size=12),
        title_font=dict(color="#E9EFF4", size=15),
        height=height,
        margin=dict(l=38, r=25, t=52, b=35),
        colorway=CHART_COLORS,
        xaxis=dict(gridcolor="#28333D", zerolinecolor="#28333D", linecolor="#3B4853"),
        yaxis=dict(gridcolor="#28333D", zerolinecolor="#28333D", linecolor="#3B4853"),
        legend=dict(title_text="", bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#25303A", font_color="#F1F5F8"),
    )
    return fig

def terminal_header(section, subtitle):
    st.markdown(
        '<div class="terminal-strip"><span>MARKETLENS <strong>/ '+section.upper()+
        '</strong></span><span class="right">HISTORICAL DATA &nbsp; • &nbsp; RESEARCH TERMINAL</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="terminal-eyebrow">ETF ANALYTICS / '+section.upper()+'</div>', unsafe_allow_html=True)
    st.title(section)
    st.caption(subtitle)
    st.markdown('<div class="terminal-rule"></div>', unsafe_allow_html=True)

# Load datasets
data = pd.read_csv(
    "data/processed/final_ml_dataset.csv"
)

results = pd.read_csv(
    "results/final_model_comparison.csv"
)

data["Date"] = pd.to_datetime(data["Date"])

# Sidebar navigation
st.sidebar.markdown("### ◈ MARKETLENS")
st.sidebar.caption("ETF RESEARCH / TERMINAL")
st.sidebar.divider()
pages = ["Overview", "ETF Explorer", "Diversification", "ML Research", "Risk Analysis"]
nav_labels = {
    "Overview": "⌂  Overview",
    "ETF Explorer": "▥  ETF Explorer",
    "Diversification": "◇  Diversification",
    "ML Research": "◈  ML Research",
    "Risk Analysis": "⬡  Risk Analysis",
}
page = st.sidebar.radio("Navigation", pages, format_func=lambda p: nav_labels[p], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("COMP 4112 / DATA SCIENCE")
st.sidebar.caption("RESEARCH USE ONLY · NOT INVESTMENT ADVICE")

# Overview

# Overview
if page == "Overview":

    terminal_header("Market Overview", "ETF performance, market indicators, and downside-risk observations")

    st.markdown("### DATASET SNAPSHOT")

    col1, col2, col3 = st.columns(3)

    col1.metric("Historical Observations", f"{len(data):,}")
    col2.metric("ETFs Analyzed", data["Ticker"].nunique())
    col3.metric("ML Model Types", results["Model"].nunique())

    st.markdown("### HISTORICAL PRICE SERIES")

    selected = st.selectbox(
        "Select an ETF",
        sorted(data["Ticker"].unique()),
        index=list(sorted(data["Ticker"].unique())).index("XEQT.TO")
    )

    etf_data = data[data["Ticker"] == selected].copy()
    etf_data = etf_data.sort_values("Date")

    # Historical return
    first_price = etf_data["Close"].iloc[0]
    last_price = etf_data["Close"].iloc[-1]

    total_return = (last_price / first_price - 1) * 100

    # Annualized historical volatility
    volatility = (
        etf_data["Return_1D"].std() * (252 ** 0.5) * 100
    )

    # Maximum historical drawdown
    running_peak = etf_data["Close"].cummax()
    drawdown = (etf_data["Close"] / running_peak - 1) * 100
    max_drawdown = drawdown.min()

    col1, col2, col3 = st.columns(3)

    col1.metric("Historical Return", f"{total_return:+.2f}%")
    col2.metric("Annualized Volatility", f"{volatility:.2f}%")
    col3.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")

    # Select the historical period to display
    period = st.segmented_control(
    "Chart Period",
    options=["6M", "1Y", "3Y", "5Y", "ALL"],
    default="ALL"
)
    end_date = etf_data["Date"].max()

    if period == "6M":
        start_date = end_date - pd.DateOffset(months=6)
    elif period == "1Y":
        start_date = end_date - pd.DateOffset(years=1)
    elif period == "3Y":
        start_date = end_date - pd.DateOffset(years=3)
    elif period == "5Y":
        start_date = end_date - pd.DateOffset(years=5)
    else:
        start_date = etf_data["Date"].min()

    chart_data = etf_data[
        etf_data["Date"] >= start_date
    ].copy()

    # Interactive closing-price chart
    fig = px.line(
        chart_data,
        x="Date",
        y="Close",
        title=f"{selected} — Historical Adjusted Price"
    )

    fig.update_traces(
        line_color="#64D7C1",
        line_width=3,
        hovertemplate=(
            "Date: %{x|%b %d, %Y}"
            "<br>Price: $%{y:.2f}<extra></extra>"
        )
    )

    fig.update_layout(
       template="plotly_dark",
    paper_bgcolor="#131920",
    plot_bgcolor="#131920",
    font_color="#D8E8F2",
    height=450,
    xaxis_title="",
    yaxis_title="Adjusted price (CAD)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="#28333D"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        title_text=""
    )
    )
    st.plotly_chart(style_chart(fig), use_container_width=True)

    # Compare risk-event frequencies
    st.markdown("### Historical Downside Risk")

    fixed_risk = etf_data["Fixed_Risk"].mean() * 100
    adjusted_risk = etf_data["Adjusted_Risk"].mean() * 100

    col1, col2 = st.columns(2)

    col1.metric("Fixed 3% Risk Frequency", f"{fixed_risk:.2f}%")
    col2.metric(
        "Volatility-Adjusted Risk Frequency",
        f"{adjusted_risk:.2f}%"
    )

    st.caption(
        "Historical statistics are calculated from observed ETF "
        "prices. Risk frequencies describe past events, not "
        "predicted probabilities."
    )


# ETF Explorer
elif page == "ETF Explorer":

    terminal_header("ETF Explorer", "Compare historical price performance across selected ETFs")

    tickers = sorted(data["Ticker"].unique())

    selected = st.multiselect(
        "Select ETFs to compare",
        tickers,
        default=["VFV.TO", "XEQT.TO", "ZNQ.TO"]
    )

    if not selected:
        st.warning("Please select at least one ETF.")

    else:
        # Load complete historical price data
        prices = pd.read_csv(
            "data/processed/all_etfs.csv"
        )
        prices["Date"] = pd.to_datetime(prices["Date"])

        prices = prices[
            prices["Ticker"].isin(selected)
        ].copy()

        # Use only dates available for every selected ETF
        price_table = prices.pivot(
            index="Date",
            columns="Ticker",
            values="Close"
        )

        price_table = price_table.dropna()
        price_table = price_table.sort_index()

        if len(price_table) < 2:
            st.warning(
                "Not enough overlapping dates to compare "
                "these ETFs."
            )
        else:
            # Normalize every ETF on the same starting date
            growth = 100 * price_table / price_table.iloc[0]

            growth_long = (
                growth.reset_index()
                .melt(
                    id_vars="Date",
                    var_name="ETF",
                    value_name="Growth"
                )
            )

            st.markdown("### Historical Growth of $100")

            fig = px.line(
                growth_long,
                x="Date",
                y="Growth",
                color="ETF",
                color_discrete_sequence=CHART_COLORS
            )

            fig.update_traces(
    hovertemplate="$%{y:.2f}<extra>%{fullData.name}</extra>",
    line=dict(width=2.5)
)

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#131920",
                plot_bgcolor="#131920",
                font_color="#D8E8F2",
                height=450,
                xaxis_title="",
                yaxis_title="Price (CAD)",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=30, b=20)
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

            # Calculate comparable performance statistics
            comparison = []

            for ticker in price_table.columns:

                series = price_table[ticker]

                daily_returns = series.pct_change().dropna()

                total_return = (
                    series.iloc[-1] / series.iloc[0] - 1
                ) * 100

                volatility = (
                    daily_returns.std() * (252 ** 0.5) * 100
                )

                running_peak = series.cummax()
                drawdown = (
                    series / running_peak - 1
                ) * 100

                comparison.append({
                    "ETF": ticker,
                    "Total Return (%)": round(
                        total_return, 2
                    ),
                    "Annualized Volatility (%)": round(
                        volatility, 2
                    ),
                    "Maximum Drawdown (%)": round(
                        drawdown.min(), 2
                    )
                })

            comparison_df = pd.DataFrame(comparison)

            st.markdown("### Performance Comparison")

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"Common comparison period: "
                f"{price_table.index.min():%Y-%m-%d} to "
                f"{price_table.index.max():%Y-%m-%d}. "
                "Returns are based on the collected "
                "adjusted-price series and exclude "
                "personal trading costs and taxes."
            )


# Diversification
elif page == "Diversification":

    terminal_header("Diversification", "Correlation structure of historical ETF daily returns")

    returns = data.pivot(
        index="Date",
        columns="Ticker",
        values="Return_1D"
    )

    correlation = returns.corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="ETF Daily Return Correlation"
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(style_chart(fig), use_container_width=True)


# ML Research
elif page == "ML Research":

    terminal_header("ML Research", "Downside-risk classification · 2025 validation period")

    target = st.selectbox(
        "Risk definition",
        ["Fixed_Risk", "Adjusted_Risk"]
    )

    model_name = st.selectbox(
        "Classification model",
        ["Logistic Regression", "Random Forest"]
    )

    model_id = (
        target + "_" + model_name.replace(" ", "_")
    )

    file_path = (
        f"results/ml_research/{model_id}_predictions.csv"
    )

    predictions = pd.read_csv(file_path)

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    threshold = st.slider(
        "Classification threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05
    )

    actual = predictions["Actual"]
    predicted = (
        predictions["Probability"] >= threshold
    ).astype(int)

    precision = precision_score(
        actual, predicted, zero_division=0
    )

    recall = recall_score(
        actual, predicted, zero_division=0
    )

    f1 = f1_score(
        actual, predicted, zero_division=0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Precision", f"{precision:.3f}")
    col2.metric("Recall", f"{recall:.3f}")
    col3.metric("F1 Score", f"{f1:.3f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        actual, predicted, labels=[0, 1]
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted Low", "Predicted High"],
        y=["Actual Low", "Actual High"],
        color_continuous_scale="Teal",
        title="Classification Results"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131920",
        font_color="#D8E8F2",
        height=400
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.subheader("Precision–Recall Curve")

    curve = pd.read_csv(
        f"results/ml_research/{model_id}_curve.csv"
    )

    fig_pr = px.line(
        curve,
        x="Recall",
        y="Precision",
        title="Precision vs Recall"
    )

    fig_pr.update_traces(
        line_color="#64D7C1",
        line_width=3
    )

    fig_pr.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131920",
        plot_bgcolor="#131920",
        font_color="#D8E8F2",
        height=400,
        xaxis_range=[0, 1],
        yaxis_range=[0, 1]
    )

    st.plotly_chart(
        style_chart(fig_pr, height=400),
        use_container_width=True
    )

    st.caption(
        "These are 2025 validation results. "
        "The classification threshold is adjustable "
        "for exploratory analysis. "
        "The 2026 period is not represented as an "
        "independent untouched test set."
    )


# Risk Analysis
elif page == "Risk Analysis":

    terminal_header("Risk Analysis", "Observed five-day downside event frequencies by ETF")

    risk_summary = data.groupby("Ticker")[
        ["Fixed_Risk", "Adjusted_Risk"]
    ].mean() * 100

    risk_summary = risk_summary.reset_index()

    fig = px.bar(
        risk_summary,
        x="Ticker",
        y=["Fixed_Risk", "Adjusted_Risk"],
        barmode="group",
        color_discrete_sequence=CHART_COLORS,
        title="Risk Event Frequency by ETF"
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.write(
        "Fixed risk identifies a decline of at least "
        "3% over five trading days. Adjusted risk "
        "uses each ETF's recent volatility."
    )
