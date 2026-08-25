"""Shared market-data and portfolio math for the notebook and Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ("AAPL", "MSFT", "GOOGL", "NVDA", "TSLA")
START_DATE = "2020-01-01"
TRADING_DAYS = 252
FRED_SERIES = "DGS3MO"
FALLBACK_RISK_FREE = 0.04
N_PORTFOLIOS = 25_000
N_PATHS = 5_000
HORIZON = 252
INITIAL = 10_000
RNG_SEED = 42

TICKER_COLORS = {
    "AAPL": "#60A5FA",
    "MSFT": "#34D399",
    "GOOGL": "#A78BFA",
    "NVDA": "#F87171",
    "TSLA": "#FBBF24",
}


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, std_dev: float = 2) -> pd.DataFrame:
    df = df.copy()
    df["MiddleBand"] = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()
    df["UpperBand"] = df["MiddleBand"] + (std_dev * rolling_std)
    df["LowerBand"] = df["MiddleBand"] - (std_dev * rolling_std)
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    df = df.copy()
    df["EMA_fast"] = df["Close"].ewm(span=fast_period, adjust=False).mean()
    df["EMA_slow"] = df["Close"].ewm(span=slow_period, adjust=False).mean()
    df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
    df["Signal"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
    return df


def add_fast_sma(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["FastSMA"] = df["Close"].rolling(window=window).mean()
    return df


def add_slow_sma(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    df = df.copy()
    df["SlowSMA"] = df["Close"].rolling(window=window).mean()
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = true_range.rolling(window=window).mean()
    return df


INDICATOR_PIPELINE = (
    add_bollinger_bands,
    add_rsi,
    add_macd,
    add_fast_sma,
    add_slow_sma,
    add_atr,
)


def load_market_data(
    tickers: tuple[str, ...] = TICKERS,
    start: str = START_DATE,
    end: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """One yfinance download: close panel plus per-ticker OHLCV with indicators."""
    panel = yf.download(
        list(tickers),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    close = panel.xs("Close", axis=1, level="Price")[list(tickers)].dropna(how="all")

    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = panel.xs(ticker, axis=1, level="Ticker")
        for add_indicator in INDICATOR_PIPELINE:
            frame = add_indicator(frame)
        frames[ticker] = frame
    return close, frames


def fetch_fred_series(series_id: str, start: str, end: str | None = None) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    if end is not None:
        url += f"&coed={end}"
    frame = pd.read_csv(url, parse_dates=["observation_date"])
    return frame.set_index("observation_date")[series_id].dropna() / 100


def load_tbill(start: str = START_DATE, end: str | None = None) -> tuple[pd.Series | None, float, float]:
    """Return (series or None, current rate, window-average rate)."""
    try:
        tbill = fetch_fred_series(FRED_SERIES, start, end=None)
        current = float(tbill.iloc[-1])
        window = tbill if end is None else tbill.loc[:end]
        window_avg = float(window.mean())
        return tbill, current, window_avg
    except Exception:
        return None, FALLBACK_RISK_FREE, FALLBACK_RISK_FREE


def sma_crosses(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    prev_diff = df["FastSMA"].shift(1) - df["SlowSMA"].shift(1)
    curr_diff = df["FastSMA"] - df["SlowSMA"]
    golden = ((prev_diff < 0) & (curr_diff >= 0)).fillna(False).astype(bool)
    death = ((prev_diff > 0) & (curr_diff <= 0)).fillna(False).astype(bool)
    return golden, death


def macd_crosses(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    hist = df["MACD"] - df["Signal"]
    prev_hist = hist.shift(1)
    bullish = ((prev_hist < 0) & (hist >= 0)).fillna(False).astype(bool)
    bearish = ((prev_hist > 0) & (hist <= 0)).fillna(False).astype(bool)
    return hist, bullish, bearish


def normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum()


def equal_weights(n_assets: int) -> np.ndarray:
    return np.ones(n_assets) / n_assets


def inverse_vol_weights(return_frame: pd.DataFrame) -> np.ndarray:
    return normalize(1.0 / return_frame.std())


def inverse_var_weights(return_frame: pd.DataFrame) -> np.ndarray:
    return normalize(1.0 / return_frame.var())


def portfolio_performance(
    weights: np.ndarray,
    mean_ann: np.ndarray | pd.Series,
    cov_ann: np.ndarray | pd.DataFrame,
    rf: float,
) -> tuple[float, float, float]:
    weights = np.asarray(weights, dtype=float)
    mean_ann = np.asarray(mean_ann, dtype=float)
    cov_ann = np.asarray(cov_ann, dtype=float)
    ret = float(weights @ mean_ann)
    vol = float(np.sqrt(weights @ cov_ann @ weights))
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe


def risk_contributions(weights: np.ndarray, cov_ann: np.ndarray | pd.DataFrame) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    cov_ann = np.asarray(cov_ann, dtype=float)
    port_vol = np.sqrt(weights @ cov_ann @ weights)
    marginal = cov_ann @ weights / port_vol
    contrib = weights * marginal
    return contrib / contrib.sum()


def bootstrap_paths(
    daily_returns: np.ndarray,
    n_paths: int,
    horizon: int,
    initial: float,
    rng: np.random.Generator,
) -> np.ndarray:
    draws = rng.choice(daily_returns, size=(horizon, n_paths), replace=True)
    return initial * np.cumprod(1 + draws, axis=0)


def path_summary(paths: np.ndarray, initial: float, name: str) -> pd.Series:
    terminal = paths[-1]
    var_5 = np.percentile(terminal, 5)
    tail = terminal[terminal <= var_5]
    return pd.Series(
        {
            "Median ending wealth": np.median(terminal),
            "5th percentile (VaR)": var_5,
            "95th percentile": np.percentile(terminal, 95),
            "Expected shortfall (CVaR 5%)": tail.mean() if len(tail) else np.nan,
            "P(end below start)": float((terminal < initial).mean()),
            "Median return": np.median(terminal) / initial - 1,
        },
        name=name,
    )


@dataclass
class PortfolioBook:
    assets: list[str]
    returns: pd.DataFrame
    mean_ann: pd.Series
    vol_ann: pd.Series
    cov_ann: pd.DataFrame
    corr: pd.DataFrame
    cagr: pd.Series
    years: float
    weight_schemes: dict[str, np.ndarray]
    asset_stats: pd.DataFrame
    strategy_table: pd.DataFrame
    risk_shares: pd.DataFrame
    growth: pd.DataFrame
    drawdown: pd.Series
    mc_vol: np.ndarray
    mc_ret: np.ndarray
    mc_sharpe: np.ndarray
    chosen_paths: np.ndarray
    compare_paths: np.ndarray
    path_table: pd.DataFrame
    risk_free: float


def build_portfolio(close: pd.DataFrame, risk_free: float) -> PortfolioBook:
    assets = list(close.columns)
    returns = close.pct_change().dropna()
    n_assets = len(assets)

    mean_ann = returns.mean() * TRADING_DAYS
    vol_ann = returns.std() * np.sqrt(TRADING_DAYS)
    cov_ann = returns.cov() * TRADING_DAYS
    corr = returns.corr()
    years = len(returns) / TRADING_DAYS
    cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1

    asset_stats = pd.DataFrame(
        {
            "Annual return": mean_ann,
            "Realized CAGR": cagr,
            "Annual volatility": vol_ann,
            "Sharpe": (mean_ann - risk_free) / vol_ann,
        }
    )

    schemes: dict[str, np.ndarray] = {
        "Equal weight": equal_weights(n_assets),
        "Inverse volatility": inverse_vol_weights(returns),
        "Inverse variance": inverse_var_weights(returns),
    }

    rng = np.random.default_rng(RNG_SEED)
    mc_weights = rng.dirichlet(np.ones(n_assets), size=N_PORTFOLIOS)
    mc_ret = mc_weights @ mean_ann.values
    mc_vol = np.sqrt(np.einsum("ij,jk,ik->i", mc_weights, cov_ann.values, mc_weights))
    mc_sharpe = (mc_ret - risk_free) / mc_vol

    schemes["Max Sharpe (MC)"] = mc_weights[int(np.argmax(mc_sharpe))]
    schemes["Min volatility (MC)"] = mc_weights[int(np.argmin(mc_vol))]

    rows = []
    risk_share_cols = {}
    for name, weights in schemes.items():
        ret, vol, sharpe = portfolio_performance(weights, mean_ann, cov_ann, risk_free)
        rows.append(
            {
                "Strategy": name,
                "Expected return": ret,
                "Volatility": vol,
                "Sharpe": sharpe,
                **{asset: weight for asset, weight in zip(assets, weights)},
            }
        )
        risk_share_cols[name] = risk_contributions(weights, cov_ann)
    strategy_table = pd.DataFrame(rows).set_index("Strategy")
    risk_shares = pd.DataFrame(risk_share_cols, index=assets)

    growth = (1 + returns @ pd.DataFrame(schemes, index=assets)).cumprod()
    drawdown = (growth / growth.cummax() - 1).min()

    chosen_w = pd.Series(schemes["Inverse volatility"], index=assets)
    compare_w = pd.Series(schemes["Max Sharpe (MC)"], index=assets)
    boot_rng = np.random.default_rng(RNG_SEED + 1)
    chosen_paths = bootstrap_paths(
        (returns @ chosen_w).values, N_PATHS, HORIZON, INITIAL, boot_rng
    )
    compare_paths = bootstrap_paths(
        (returns @ compare_w).values, N_PATHS, HORIZON, INITIAL, boot_rng
    )
    path_table = pd.concat(
        [
            path_summary(chosen_paths, INITIAL, "Inverse volatility"),
            path_summary(compare_paths, INITIAL, "Max Sharpe (MC)"),
        ],
        axis=1,
    )

    return PortfolioBook(
        assets=assets,
        returns=returns,
        mean_ann=mean_ann,
        vol_ann=vol_ann,
        cov_ann=cov_ann,
        corr=corr,
        cagr=cagr,
        years=years,
        weight_schemes=schemes,
        asset_stats=asset_stats,
        strategy_table=strategy_table,
        risk_shares=risk_shares,
        growth=growth,
        drawdown=drawdown,
        mc_vol=mc_vol,
        mc_ret=mc_ret,
        mc_sharpe=mc_sharpe,
        chosen_paths=chosen_paths,
        compare_paths=compare_paths,
        path_table=path_table,
        risk_free=risk_free,
    )
