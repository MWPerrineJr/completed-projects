"""
Stock analysis dashboard
------------------------
Streamlit port of stock_analysis.ipynb: technical-indicator comparisons
across AAPL, MSFT, GOOGL, NVDA, and TSLA, plus the inverse-volatility
book and its Monte Carlo stress test.

Run with:
    uv run streamlit run app.py
from this folder (or the workspace root with the path to this file).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis import (
    HORIZON,
    INITIAL,
    N_PATHS,
    N_PORTFOLIOS,
    START_DATE,
    TICKER_COLORS,
    TICKERS,
    PortfolioBook,
    build_portfolio,
    load_market_data,
    load_tbill,
    macd_crosses,
    portfolio_performance,
    sma_crosses,
)

st.set_page_config(
    page_title="Stock analysis",
    page_icon=":material/candlestick_chart:",
    layout="wide",
)

CHART_HEIGHT = 400
MACD_HEIGHT = 500
MC_HEIGHT = 460
# Title sits at the top of the figure; the legend sits in the top margin below it
# (paper y=1 is the top of the plot area, so y=1.02 is just above the traces).
PLOT_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=108, b=40, l=52, r=24),
    title=dict(y=0.98, yanchor="top", yref="container", pad=dict(t=4, b=10)),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        yref="paper",
        x=0,
        xanchor="left",
        bgcolor="rgba(0,0,0,0)",
    ),
)

TAB_LABELS = (
    ":material/show_chart: Price",
    ":material/sync_alt: Moving averages",
    ":material/stacked_line_chart: Bollinger bands",
    ":material/ssid_chart: MACD",
    ":material/pie_chart: Weights",
    ":material/analytics: Monte Carlo",
)


@st.cache_data(ttl="1h", show_spinner="Downloading prices and computing indicators...")
def cached_market(tickers: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    return load_market_data(tickers, start=START_DATE)


@st.cache_data(ttl="1h", show_spinner="Fetching 3-month Treasury yield from FRED...")
def cached_tbill() -> tuple[pd.Series | None, float, float]:
    return load_tbill(start=START_DATE)


@st.cache_data(ttl="30m", show_spinner="Running portfolio math and Monte Carlo...")
def cached_portfolio(close: pd.DataFrame, risk_free: float) -> PortfolioBook:
    return build_portfolio(close, risk_free)


def apply_layout(fig: go.Figure, height: int = CHART_HEIGHT) -> go.Figure:
    fig.update_layout(height=height, **PLOT_BG)
    return fig


def color_map(tickers: list[str]) -> dict[str, str]:
    return {t: TICKER_COLORS.get(t, "#94A3B8") for t in tickers}


def price_figure(close: pd.DataFrame, title: str, y_title: str, percent: bool = False) -> go.Figure:
    fig = px.line(
        close,
        title=title,
        color_discrete_map=color_map(list(close.columns)),
    )
    fig.update_layout(xaxis_title="Date", yaxis_title=y_title, hovermode="x unified")
    if percent:
        fig.update_yaxes(tickformat=".0%")
    return apply_layout(fig)


def sma_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    golden, death = sma_crosses(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#94A3B8", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["FastSMA"], name="10-day SMA", line=dict(color="#FB923C", width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SlowSMA"], name="50-day SMA", line=dict(color="#A78BFA", width=2)))
    fig.add_trace(
        go.Scatter(
            x=df.index[golden],
            y=df.loc[golden, "FastSMA"],
            mode="markers",
            name=f"Golden cross ({int(golden.sum())})",
            marker=dict(symbol="triangle-up", size=9, color="#34D399"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index[death],
            y=df.loc[death, "FastSMA"],
            mode="markers",
            name=f"Death cross ({int(death.sum())})",
            marker=dict(symbol="triangle-down", size=9, color="#F87171"),
        )
    )
    fig.update_layout(title=f"{ticker} — 10-day vs 50-day SMA", xaxis_title="Date", yaxis_title="Price ($)")
    return apply_layout(fig)


def bollinger_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["UpperBand"],
            name="Upper band (+2σ)",
            line=dict(color="#FB923C", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["LowerBand"],
            name="Lower band (−2σ)",
            line=dict(color="#A78BFA", width=1),
            fill="tonexty",
            fillcolor="rgba(96, 165, 250, 0.12)",
        )
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MiddleBand"], name="Middle band (20-day)", line=dict(color="#60A5FA", width=1.5))
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#F1F5F9", width=1.5)))
    fig.update_layout(title=f"{ticker} — Bollinger bands (20-day, 2σ)", xaxis_title="Date", yaxis_title="Price ($)")
    return apply_layout(fig)


def macd_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    hist, bullish, bearish = macd_crosses(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06)
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#F1F5F9", width=1.4)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index[bullish],
            y=df.loc[bullish, "Close"],
            mode="markers",
            name=f"Bullish ({int(bullish.sum())})",
            marker=dict(symbol="triangle-up", size=8, color="#34D399"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index[bearish],
            y=df.loc[bearish, "Close"],
            mode="markers",
            name=f"Bearish ({int(bearish.sum())})",
            marker=dict(symbol="triangle-down", size=8, color="#F87171"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=hist,
            name="Histogram",
            marker_color=np.where(hist >= 0, "#34D399", "#F87171"),
            opacity=0.35,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#60A5FA", width=1.5)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Signal"], name="Signal", line=dict(color="#FB923C", width=1.5)),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#94A3B8", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="EMA gap ($)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(title=f"{ticker} — MACD (12, 26, 9)", barmode="overlay")
    return apply_layout(fig, height=MACD_HEIGHT)


def grouped_bar(frame: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    long = frame.reset_index().melt(id_vars=frame.index.name or "index", var_name="Strategy", value_name="Share")
    x_col = long.columns[0]
    fig = px.bar(long, x=x_col, y="Share", color="Strategy", barmode="group", title=title)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(xaxis_title="Ticker", yaxis_title=y_title)
    return apply_layout(fig)


def frontier_figure(book: PortfolioBook) -> go.Figure:
    rng = np.random.default_rng(0)
    n_show = min(4_000, len(book.mc_vol))
    sample = rng.choice(len(book.mc_vol), size=n_show, replace=False)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=book.mc_vol[sample],
            y=book.mc_ret[sample],
            mode="markers",
            name="Random portfolios",
            marker=dict(
                size=4,
                color=book.mc_sharpe[sample],
                colorscale="Viridis",
                colorbar=dict(title="Sharpe", thickness=12, len=0.7),
                opacity=0.45,
            ),
            hovertemplate="Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
        )
    )
    markers = {
        "Equal weight": "circle",
        "Inverse volatility": "diamond",
        "Inverse variance": "square",
        "Max Sharpe (MC)": "star",
        "Min volatility (MC)": "x",
    }
    for name, weights in book.weight_schemes.items():
        ret, vol, sharpe = portfolio_performance(weights, book.mean_ann, book.cov_ann, book.risk_free)
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[ret],
                mode="markers",
                name=name,
                marker=dict(size=12, symbol=markers.get(name, "circle")),
                hovertemplate=(
                    f"{name}<br>Vol: %{{x:.1%}}<br>Return: %{{y:.1%}}"
                    f"<br>Sharpe: {sharpe:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=dict(
            text=f"Feasible set ({N_PORTFOLIOS:,} long-only books)",
            y=0.98,
            yanchor="top",
            yref="container",
            pad=dict(t=4, b=10),
        ),
        xaxis_title="Annual volatility",
        yaxis_title="Annual expected return",
        height=MC_HEIGHT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=72, b=40, l=52, r=140),
        legend=dict(orientation="v", y=1, yanchor="top", x=1.18, xanchor="left", tracegroupgap=10),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


def growth_figure(growth: pd.DataFrame) -> go.Figure:
    fig = px.line(growth, title="Growth of $1 by strategy (in-sample)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Growth of $1")
    return apply_layout(fig, height=MC_HEIGHT)


def fan_figure(paths: np.ndarray, title: str) -> go.Figure:
    days = np.arange(1, HORIZON + 1)
    p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([p95, p5[::-1]]),
            fill="toself",
            fillcolor="rgba(96, 165, 250, 0.15)",
            line=dict(width=0),
            name="5–95% band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill="toself",
            fillcolor="rgba(96, 165, 250, 0.30)",
            line=dict(width=0),
            name="25–75% band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(x=days, y=p50, mode="lines", name="Median path", line=dict(color="#60A5FA", width=2))
    )
    fig.add_hline(y=INITIAL, line_dash="dot", line_color="#94A3B8", annotation_text="Start")
    fig.update_layout(title=title, xaxis_title="Trading day", yaxis_title="Portfolio value ($)")
    return apply_layout(fig, height=MC_HEIGHT)


def terminal_figure(chosen: np.ndarray, compare: np.ndarray) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Inverse volatility": chosen[-1],
            "Max Sharpe (MC)": compare[-1],
        }
    )
    fig = px.histogram(
        frame.melt(var_name="Strategy", value_name="Ending wealth"),
        x="Ending wealth",
        color="Strategy",
        barmode="overlay",
        opacity=0.65,
        nbins=50,
        title="Ending wealth after one year",
    )
    fig.add_vline(x=INITIAL, line_dash="dot", line_color="#F1F5F9")
    fig.update_layout(yaxis_title="Paths")
    return apply_layout(fig, height=MC_HEIGHT)


def corr_figure(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Daily return correlations",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="ρ"))
    return apply_layout(fig, height=MC_HEIGHT)


def format_path_table(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy()
    rate_rows = ["P(end below start)", "Median return"]
    for col in formatted.columns:
        out = []
        for idx, value in formatted[col].items():
            if idx in rate_rows:
                out.append(f"{value:.1%}")
            else:
                out.append(f"{value:,.0f}")
        formatted[col] = out
    return formatted


def comparison_note(title: str, body: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(body)


def stock_grid(tickers: list[str], frames: dict[str, pd.DataFrame], builder) -> None:
    for i in range(0, len(tickers), 2):
        pair = tickers[i : i + 2]
        cols = st.columns(2, border=True)
        for col, ticker in zip(cols, pair):
            with col:
                st.plotly_chart(builder(frames[ticker], ticker), width="stretch")
        if len(pair) == 1:
            with cols[1]:
                st.caption("Select a second ticker in the sidebar to compare side by side.")


def fmt_pct(value: float) -> str:
    return f"{value:.1%}"


st.title("Stock analysis")
st.caption(
    "Technicals and a five-name tech book for AAPL, MSFT, GOOGL, NVDA, and TSLA. "
    "Every number is in-sample — it describes this window, not a forecast."
)

with st.sidebar:
    st.header("Filters")
    selected = st.pills(
        "Tickers to compare",
        options=list(TICKERS),
        selection_mode="multi",
        default=list(TICKERS),
        help="Technical tabs plot these names side by side. Portfolio tabs rebuild the book from the same set.",
    )
    rf_mode = st.segmented_control(
        "Risk-free rate",
        options=["Current T-bill", "Window average"],
        default="Current T-bill",
        required=True,
        help="Current uses the latest FRED DGS3MO yield. Window average is the mean over the return period.",
    )
    st.caption(f"Prices from Yahoo Finance starting {START_DATE}. Yield from FRED series DGS3MO.")

if not selected:
    st.info("Select at least one ticker in the sidebar.")
    st.stop()

selected_tickers = tuple(t for t in TICKERS if t in selected)
close_all, frames_all = cached_market(TICKERS)
_tbill, rf_current, rf_window = cached_tbill()
risk_free = rf_current if rf_mode == "Current T-bill" else rf_window
close = close_all[list(selected_tickers)]
frames = {t: frames_all[t] for t in selected_tickers}

with st.container(horizontal=True):
    last = close.iloc[-1]
    first = close.iloc[0]
    st.metric("Window", f"{close.index.min():%b %Y} – {close.index.max():%b %Y}", border=True)
    st.metric("Risk-free", fmt_pct(risk_free), border=True)
    leader = (last / first).idxmax()
    st.metric(f"{leader} total return", fmt_pct(float(last[leader] / first[leader] - 1)), border=True)
    quietest = close.pct_change().std().idxmin()
    st.metric(
        f"Quietest ({quietest})",
        fmt_pct(float(close.pct_change().std()[quietest] * np.sqrt(252))),
        border=True,
    )

price_tab, sma_tab, bb_tab, macd_tab, weights_tab, mc_tab = st.tabs(
    TAB_LABELS,
    on_change="rerun",
    key="main_tabs",
)

with price_tab:
    if price_tab.open:
        rebased = close.div(close.iloc[0]).sub(1)
        left, right = st.columns(2, border=True)
        with left:
            st.plotly_chart(price_figure(close, "Adjusted close", "Price ($)"), width="stretch")
        with right:
            st.plotly_chart(
                price_figure(rebased, "Growth of $1 at the start of the window", "Return", percent=True),
                width="stretch",
            )
        comparison_note(
            "What these two views show",
            "Dollar prices hide relative performance because these names trade at very different levels — "
            "NVDA's split-adjusted price sits far below MSFT even after a much larger run. The right-hand "
            "chart puts every name on the same starting dollar so you can compare who compounded, not who "
            "is expensive per share. NVDA and TSLA will usually look like the wildest lines on both sides.",
        )

with sma_tab:
    if sma_tab.open:
        comparison_note(
            "How to read the pair",
            "Each panel is one stock. A green triangle is a golden cross (10-day SMA rising through the "
            "50-day); a red triangle is a death cross. Compare how often names whip around: a choppy "
            "stock prints far more crosses than a trending one, which is why this is a confirmation tool, "
            "not a forecast. The fast line hugging price more tightly is the 10-day average.",
        )
        stock_grid(list(selected_tickers), frames, sma_figure)

with bb_tab:
    if bb_tab.open:
        comparison_note(
            "How to read the pair",
            "Band width is recent volatility. Compare a quiet name (bands pinched in calm stretches) "
            "with NVDA or TSLA, whose envelope expands whenever the stock is running. A squeeze often "
            "precedes a larger move, but a touch of the upper band is not automatically a sell — trending "
            "names ride the band for weeks.",
        )
        stock_grid(list(selected_tickers), frames, bollinger_figure)

with macd_tab:
    if macd_tab.open:
        comparison_note(
            "How to read the pair",
            "Price on top, MACD below. MACD uses exponential averages, so it fires roughly three times as "
            "often as the SMA crosses on the previous tab — compare the marker counts. Markers on price let "
            "you judge whether a crossover actually led the next move. The zero line on the lower panel is "
            "the slower signal: MACD above zero means the 12-day EMA still sits above the 26-day.",
        )
        stock_grid(list(selected_tickers), frames, macd_figure)

with weights_tab:
    if weights_tab.open:
        if len(selected_tickers) < 2:
            st.info("Select at least two tickers to build portfolio weights.")
        else:
            book = cached_portfolio(close, risk_free)
            off_diag = book.corr.where(~np.eye(len(book.assets), dtype=bool))
            left, right = st.columns(2, border=True)
            with left:
                weights_frame = pd.DataFrame(book.weight_schemes, index=book.assets)
                heuristic = weights_frame[["Equal weight", "Inverse volatility", "Inverse variance"]]
                st.plotly_chart(grouped_bar(heuristic, "Capital weights", "Weight"), width="stretch")
            with right:
                risk_heuristic = book.risk_shares[["Equal weight", "Inverse volatility", "Inverse variance"]]
                st.plotly_chart(grouped_bar(risk_heuristic, "Risk contribution", "Risk share"), width="stretch")
            comparison_note(
                "Capital weight is not risk weight",
                "Equal dollars (left) still dump most of the book's risk (right) into the wildest names — "
                "typically NVDA and TSLA. Inverse volatility thins those two so each holding's risk share "
                f"sits closer to 1/{len(book.assets)}. Average pairwise correlation here is "
                f"{off_diag.stack().mean():.2f}, which is why reweighting these five is not the same as "
                "adding a different asset class.",
            )
            st.dataframe(
                book.strategy_table.style.format(
                    {
                        "Expected return": "{:.2%}",
                        "Volatility": "{:.2%}",
                        "Sharpe": "{:.2f}",
                        **{asset: "{:.1%}" for asset in book.assets},
                    }
                ),
                width="stretch",
            )
            st.caption(
                f"Sharpe uses a {book.risk_free:.2%} risk-free rate ({rf_mode.lower()}). "
                "Max Sharpe and min volatility are added on the Monte Carlo tab."
            )

with mc_tab:
    if mc_tab.open:
        if len(selected_tickers) < 2:
            st.info("Select at least two tickers to run the Monte Carlo.")
        else:
            book = cached_portfolio(close, risk_free)
            row1_left, row1_right = st.columns(2, border=True)
            with row1_left:
                st.plotly_chart(frontier_figure(book), width="stretch")
            with row1_right:
                st.plotly_chart(growth_figure(book.growth), width="stretch")
            comparison_note(
                "Feasible set vs realized path",
                "Left is every random long-only mix of these names; the upper-left edge is the efficient "
                "frontier. Max Sharpe usually piles into whichever name ran hardest in this sample (often NVDA). "
                "Inverse volatility sits inside the cloud on purpose — it is a diversification rule, not an "
                "in-sample optimum. Right compounds $1 under daily rebalancing with no costs, so treat it as "
                "a ranking of these rules, not a live track record.",
            )

            row2_left, row2_right = st.columns(2, border=True)
            with row2_left:
                st.plotly_chart(
                    fan_figure(book.chosen_paths, f"Inverse volatility — {N_PATHS:,} bootstrap paths"),
                    width="stretch",
                )
            with row2_right:
                st.plotly_chart(terminal_figure(book.chosen_paths, book.compare_paths), width="stretch")
            comparison_note(
                "Forward paths vs ending-wealth spread",
                "Left bootstraps the inverse-volatility book's own daily returns for one trading year. "
                "The fan is the cross-section of those paths, not a single scenario. Right compares ending "
                "wealth with max Sharpe: the higher median comes with a wider left tail. Independent daily "
                "draws understate sustained drawdowns because they destroy volatility clustering.",
            )

            metrics = st.columns(3, border=True)
            with metrics[0]:
                st.metric("Inverse-vol max drawdown", fmt_pct(float(book.drawdown["Inverse volatility"])))
            with metrics[1]:
                st.metric("Max-Sharpe max drawdown", fmt_pct(float(book.drawdown["Max Sharpe (MC)"])))
            with metrics[2]:
                p_loss = float(book.path_table.loc["P(end below start)", "Inverse volatility"])
                st.metric("P(end below $10k), inverse vol", fmt_pct(p_loss))

            corr_col, table_col = st.columns(2, border=True)
            with corr_col:
                st.plotly_chart(corr_figure(book.corr), width="stretch")
            with table_col:
                st.markdown("**One-year bootstrap from $10,000**")
                st.dataframe(format_path_table(book.path_table), width="stretch")
                st.caption(
                    "Wealth rows are dollars; P(end below start) and median return are rates. "
                    "CVaR averages only the worst 5% of paths."
                )
