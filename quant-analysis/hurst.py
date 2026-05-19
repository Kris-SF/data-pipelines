"""
Hurst exponent estimation via classic R/S analysis.

For a series of log returns, R/S grows like T^H over window size T.
H is the slope of log₂(R/S) regressed on log₂(T):

    H = 0.5   random walk (independent steps)
    H > 0.5   trending — wandering grows faster than √T
    H < 0.5   mean-reverting — wandering grows slower than √T

The trader's punchline: H ≠ 0.5 means √T vol annualization is wrong
for that asset — long-horizon vol scales as σ_daily · T^H, not √T.

Usage (Jupyter / VS Code notebook):

    from hurst import run_hurst_analysis
    result = run_hurst_analysis(
        tickers=["SPY", "QQQ", "TLT"],
        start="2020-01-01",
        end="2025-01-01",
        compare_methods=True,    # runs A/B/C side-by-side at 252d and 126d,
                                 # with iid-Gaussian calibration + verdict
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

from data import fetch_daily_returns

# Moontower palette — self-contained so this module has no cross-folder deps.
INDIGO_600 = "#4F46E5"
BLUE_500 = "#3B82F6"
GREEN_500 = "#22C55E"
RED_500 = "#EF4444"
ORANGE_600 = "#EA580C"
VIOLET_600 = "#7C3AED"
GRAY_900 = "#111827"
GRAY_700 = "#374151"
GRAY_500 = "#6B7280"
GRAY_300 = "#D1D5DB"
GRAY_200 = "#E5E7EB"
GRAY_100 = "#F3F4F6"
GRAY_50 = "#F9FAFB"

_LINE_PALETTE = [INDIGO_600, BLUE_500, GREEN_500, ORANGE_600, VIOLET_600, GRAY_700]
DEFAULT_T_CANDIDATES = [5, 10, 20, 40, 80, 160, 320]
DEFAULT_ROLLING_WINDOW = 252

# Method-specific T schedules. "A" is the historical default (wide spread).
# "B" is the narrow-spread variant; "C" reuses A's T's but applies inverse-
# variance-style weighting (sqrt of chunk count) in the regression.
METHOD_T_VALUES: dict[str, list[int]] = {
    "A": [5, 10, 20, 40, 80],
    "B": [5, 10, 15, 20, 25],
    "C": [5, 10, 20, 40, 80],
}
METHOD_LABELS: dict[str, str] = {
    "A": "A · wide T, unweighted",
    "B": "B · narrow T, unweighted",
    "C": "C · wide T, chunk-weighted",
}
METHOD_COLORS: dict[str, str] = {
    "A": INDIGO_600,
    "B": ORANGE_600,
    "C": GREEN_500,
}


# -------------------------------------------------------------------
# Math
# -------------------------------------------------------------------

def rs_at_T(returns: np.ndarray, T: int) -> float:
    """
    Average R/S across non-overlapping T-sized chunks of a return series.

    For each chunk: de-mean, cumulative sum, R = max - min of that path,
    S = std dev of the original returns, R/S = R / S. Skips chunks with
    zero variance (constant returns) since R/S is undefined there.
    """
    rs, _ = _rs_at_T_with_count(returns, T)
    return rs


def _rs_at_T_with_count(returns: np.ndarray, T: int) -> tuple[float, int]:
    """Like rs_at_T, but also returns the number of usable chunks."""
    n = len(returns)
    n_chunks = n // T
    if n_chunks == 0:
        return float("nan"), 0

    # Reshape into (n_chunks, T) and compute R/S per chunk in one shot.
    chunks = np.asarray(returns[: n_chunks * T], dtype=float).reshape(n_chunks, T)
    s = chunks.std(axis=1, ddof=0)
    cum = np.cumsum(chunks - chunks.mean(axis=1, keepdims=True), axis=1)
    r = cum.max(axis=1) - cum.min(axis=1)
    valid = s > 0
    if not valid.any():
        return float("nan"), 0
    rs_vals = r[valid] / s[valid]
    return float(rs_vals.mean()), int(valid.sum())


def _resolve_T_values(n: int, T_values: Sequence[int] | None) -> list[int]:
    """Cap T values so each has at least 4 non-overlapping chunks."""
    if T_values is None:
        T_values = DEFAULT_T_CANDIDATES
    return [t for t in T_values if t >= 2 and n // t >= 2]


@dataclass
class HurstFit:
    H: float
    intercept: float
    rs_table: pd.DataFrame  # columns: T, R/S, sqrt(T), log2(T), log2(R/S), n_chunks


def _resolve_method_T_values(
    n: int,
    method: str,
    T_values: Sequence[int] | None,
) -> list[int]:
    """Pick T's for a method; explicit T_values override the method default."""
    if T_values is not None:
        return _resolve_T_values(n, T_values)
    if method not in METHOD_T_VALUES:
        raise ValueError(f"unknown method {method!r}; expected one of A, B, C")
    return _resolve_T_values(n, METHOD_T_VALUES[method])


def hurst_exponent(
    returns: pd.Series | np.ndarray,
    T_values: Sequence[int] | None = None,
    *,
    method: str = "A",
) -> HurstFit:
    """
    Estimate H by fitting log₂(R/S) ~ H · log₂(T).

    method:
      "A" — wide T = [5, 10, 20, 40, 80], unweighted regression (default).
      "B" — narrow T = [5, 10, 15, 20, 25], unweighted regression.
      "C" — same T as A, weighted regression with weight = sqrt(n_chunks(T))
            (inverse-variance flavor — long-T points carry fewer samples).

    Explicit T_values overrides the method's default T schedule.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    Ts = _resolve_method_T_values(len(arr), method, T_values)
    rows: list[tuple[int, float, int]] = []
    for T in Ts:
        rs, n_chunks = _rs_at_T_with_count(arr, T)
        if np.isfinite(rs) and rs > 0 and n_chunks > 0:
            rows.append((T, rs, n_chunks))

    empty_cols = ["T", "R/S", "sqrt(T)", "log2(T)", "log2(R/S)", "n_chunks"]
    if len(rows) < 2:
        return HurstFit(
            H=float("nan"),
            intercept=float("nan"),
            rs_table=pd.DataFrame(columns=empty_cols),
        )

    T_arr = np.array([t for t, _, _ in rows])
    rs_arr = np.array([rs for _, rs, _ in rows])
    n_arr = np.array([n for _, _, n in rows], dtype=float)
    log_T = np.log2(T_arr)
    log_RS = np.log2(rs_arr)

    if method == "C":
        # sqrt(n_chunks) weighting — np.polyfit's `w` is interpreted as 1/sigma.
        H, intercept = np.polyfit(log_T, log_RS, 1, w=np.sqrt(n_arr))
    else:
        H, intercept = np.polyfit(log_T, log_RS, 1)

    table = pd.DataFrame(
        {
            "T": T_arr,
            "R/S": rs_arr,
            "sqrt(T)": np.sqrt(T_arr),
            "log2(T)": log_T,
            "log2(R/S)": log_RS,
            "n_chunks": n_arr.astype(int),
        }
    )
    return HurstFit(H=float(H), intercept=float(intercept), rs_table=table)


def rolling_hurst(
    returns: pd.Series,
    *,
    window: int = DEFAULT_ROLLING_WINDOW,
    T_values: Sequence[int] | None = None,
    method: str = "A",
) -> pd.Series:
    """Rolling H on a trailing window of daily returns."""
    Hs: list[float] = []
    dates: list[pd.Timestamp] = []
    for end_i in range(window, len(returns) + 1):
        sub = returns.iloc[end_i - window : end_i].to_numpy()
        Hs.append(hurst_exponent(sub, T_values=T_values, method=method).H)
        dates.append(returns.index[end_i - 1])
    return pd.Series(Hs, index=pd.DatetimeIndex(dates), name="H")


# -------------------------------------------------------------------
# Result container + driver
# -------------------------------------------------------------------

@dataclass
class TickerHurst:
    ticker: str
    H: float
    fit: HurstFit


@dataclass
class HurstAnalysisResult:
    returns: pd.DataFrame
    per_ticker: dict[str, TickerHurst]
    T_values: list[int]
    rolling_window: int
    rolling: pd.DataFrame  # columns = tickers, index = dates

    def H_series(self) -> pd.Series:
        return pd.Series(
            {t: v.H for t, v in self.per_ticker.items()}, name="H"
        ).sort_values()


DEFAULT_COMPARISON_WINDOWS: tuple[int, ...] = (252, 126)


def run_hurst_analysis(
    tickers: Iterable[str],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    T_values: Sequence[int] | None = None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    show: bool = True,
    compare_methods: bool = False,
    comparison_windows: Sequence[int] = DEFAULT_COMPARISON_WINDOWS,
    calibration_sims: int = 500,
) -> HurstAnalysisResult:
    """Fetch returns, compute Hurst per ticker, render moontower-styled plots."""
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers list is empty")

    returns = fetch_daily_returns(tickers, start=start, end=end)
    fetched_tickers = list(returns.columns)

    Ts = _resolve_T_values(len(returns), T_values)
    if len(Ts) < 2:
        raise ValueError(
            f"sample of {len(returns)} days is too short for the requested "
            f"T values; got resolved Ts = {Ts}"
        )

    per_ticker: dict[str, TickerHurst] = {}
    for t in fetched_tickers:
        fit = hurst_exponent(returns[t].to_numpy(), T_values=Ts)
        per_ticker[t] = TickerHurst(ticker=t, H=fit.H, fit=fit)

    rolling_df = pd.DataFrame(index=pd.DatetimeIndex([]))
    if len(returns) >= rolling_window + 2:
        cols = {}
        for t in fetched_tickers:
            cols[t] = rolling_hurst(
                returns[t], window=rolling_window, T_values=Ts
            )
        rolling_df = pd.DataFrame(cols)

    result = HurstAnalysisResult(
        returns=returns,
        per_ticker=per_ticker,
        T_values=Ts,
        rolling_window=rolling_window,
        rolling=rolling_df,
    )

    if show:
        _apply_mpl_style()
        _render_summary_card(result)
        _render_rs_detail_table(result)
        _render_loglog_fit(result)
        _render_H_bar(result)
        _render_rolling_H(result)
        if compare_methods:
            _render_method_comparison_multi_window(
                returns=returns,
                tickers=fetched_tickers,
                windows=comparison_windows,
                calibration_sims=calibration_sims,
            )

    return result


# -------------------------------------------------------------------
# Style
# -------------------------------------------------------------------

def _apply_mpl_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRAY_300,
            "axes.linewidth": 0.8,
            "axes.labelcolor": GRAY_700,
            "axes.labelsize": 10,
            "axes.titlecolor": GRAY_900,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlepad": 14,
            "axes.grid": True,
            "grid.color": GRAY_100,
            "grid.linewidth": 1,
            "xtick.color": GRAY_500,
            "ytick.color": GRAY_500,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "sans-serif",
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def _regime(H: float) -> tuple[str, str]:
    """Return (label, color) for a Hurst value."""
    if np.isnan(H):
        return ("Insufficient data", GRAY_500)
    if H >= 0.55:
        return ("Trending", GREEN_500)
    if H <= 0.45:
        return ("Mean-Reverting", BLUE_500)
    return ("Random", GRAY_700)


# -------------------------------------------------------------------
# HTML summary + detail table
# -------------------------------------------------------------------

def _tile(label: str, value: str, sublabel: str | None = None,
          value_color: str = INDIGO_600) -> str:
    sub = (
        f'<div style="font-size: 11px; color: {GRAY_500}; '
        f'font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin-top: 4px;">'
        f"{sublabel}</div>"
        if sublabel
        else ""
    )
    return f"""
    <div style="background: white; border: 1px solid {GRAY_200}; border-radius: 8px;
                padding: 14px 16px; text-align: center;">
      <div style="font-size: 22px; font-weight: 700; color: {value_color};
                  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;">
        {value}
      </div>
      <div style="font-size: 10.5px; color: {GRAY_500}; text-transform: uppercase;
                  letter-spacing: 0.06em; margin-top: 6px;">
        {label}
      </div>
      {sub}
    </div>
    """


def _render_summary_card(r: HurstAnalysisResult) -> None:
    sample_range = (
        f"{r.returns.index.min().date()} &rarr; {r.returns.index.max().date()}"
    )
    n_tickers = len(r.per_ticker)
    cols = min(n_tickers, 4)
    tiles = []
    for t, info in r.per_ticker.items():
        regime, color = _regime(info.H)
        value = "—" if np.isnan(info.H) else f"{info.H:.3f}"
        tiles.append(_tile(t, value, sublabel=regime, value_color=color))

    grid = (
        f'<div style="display: grid; grid-template-columns: repeat({cols}, 1fr); '
        f'gap: 16px;">{"".join(tiles)}</div>'
    )
    header = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                padding: 12px 0;">
      <div style="font-size: 11px; text-transform: uppercase;
                  letter-spacing: 0.08em; color: {INDIGO_600}; font-weight: 600;">
        Hurst Exponent &mdash; R/S Analysis
      </div>
      <div style="font-size: 22px; font-weight: 700; color: {GRAY_900};
                  margin: 4px 0 6px 0;">
        {n_tickers} tickers &middot; {len(r.returns)} days &middot; T = {r.T_values}
      </div>
      <div style="font-size: 11px; color: {GRAY_500};
                  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                  margin-bottom: 16px;">
        Sample: {sample_range} &middot; rolling window: {r.rolling_window}d
      </div>
      {grid}
    </div>
    """
    display(HTML(header))


def _render_rs_detail_table(r: HurstAnalysisResult) -> None:
    """Show the worked-example R/S table for the first ticker as a sanity-check."""
    first_t, info = next(iter(r.per_ticker.items()))
    if info.fit.rs_table.empty:
        return

    df = info.fit.rs_table.copy()
    df["sqrt(T) [random benchmark]"] = df["sqrt(T)"]
    df = df.drop(columns=["sqrt(T)"])
    df = df[["T", "R/S", "sqrt(T) [random benchmark]", "log2(T)", "log2(R/S)"]]

    styled = (
        df.style.format(
            {
                "T": "{:d}",
                "R/S": "{:.3f}",
                "sqrt(T) [random benchmark]": "{:.3f}",
                "log2(T)": "{:.3f}",
                "log2(R/S)": "{:.3f}",
            }
        )
        .hide(axis="index")
        .set_caption(f"R/S table — {first_t}  ·  H ≈ {info.H:.3f}")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": (
                        f"caption-side: top; text-align: left; font-weight: 700; "
                        f"text-transform: uppercase; letter-spacing: 0.06em; "
                        f"color: {GRAY_900}; padding: 12px 0 8px 0; font-size: 13px;"
                    ),
                },
                {
                    "selector": "th.col_heading",
                    "props": (
                        f"background: {GRAY_100}; color: {GRAY_700}; "
                        f"border-bottom: 1px solid {GRAY_300}; padding: 8px 12px; "
                        "text-align: right; text-transform: uppercase; "
                        "letter-spacing: 0.04em; font-size: 11px;"
                    ),
                },
                {
                    "selector": "td",
                    "props": (
                        f"border-bottom: 1px solid {GRAY_100}; padding: 8px 12px; "
                        "font-family: ui-monospace, SFMono-Regular, Menlo, monospace; "
                        "text-align: right;"
                    ),
                },
                {"selector": "", "props": "border-collapse: collapse;"},
            ]
        )
    )
    display(styled)


# -------------------------------------------------------------------
# Matplotlib figures
# -------------------------------------------------------------------

def _render_loglog_fit(r: HurstAnalysisResult) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))

    log_T_grid = np.log2(np.array(r.T_values, dtype=float))

    # Random-walk reference: slope 0.5 through the centroid of all data points.
    all_x: list[float] = []
    all_y: list[float] = []
    for info in r.per_ticker.values():
        if info.fit.rs_table.empty:
            continue
        all_x.extend(info.fit.rs_table["log2(T)"].tolist())
        all_y.extend(info.fit.rs_table["log2(R/S)"].tolist())
    if all_x:
        cx, cy = float(np.mean(all_x)), float(np.mean(all_y))
        y_ref = cy + 0.5 * (log_T_grid - cx)
        ax.plot(
            log_T_grid,
            y_ref,
            color=GRAY_500,
            linestyle="--",
            linewidth=1,
            label="H = 0.5 (random walk)",
        )

    for (t, info), color in zip(r.per_ticker.items(), cycle(_LINE_PALETTE)):
        if info.fit.rs_table.empty:
            continue
        x = info.fit.rs_table["log2(T)"].to_numpy()
        y = info.fit.rs_table["log2(R/S)"].to_numpy()
        ax.scatter(x, y, color=color, s=40, zorder=3, edgecolor="white", linewidth=1)
        x_line = np.array([x.min(), x.max()])
        y_line = info.fit.intercept + info.H * x_line
        ax.plot(
            x_line,
            y_line,
            color=color,
            linewidth=2,
            label=f"{t}  ·  H = {info.H:.3f}",
        )

    ax.set_title("R/S vs T  (log–log)  ·  slope = H")
    ax.set_xlabel("log₂(T)")
    ax.set_ylabel("log₂(R/S)")
    ax.legend(loc="best")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()


def _render_H_bar(r: HurstAnalysisResult) -> None:
    H_series = r.H_series()
    if H_series.dropna().empty:
        return
    height = max(2.5, 0.42 * len(H_series) + 1.0)
    fig, ax = plt.subplots(figsize=(10, height))

    colors = [_regime(h)[1] for h in H_series.values]
    ax.barh(H_series.index, H_series.values, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0.5, color=GRAY_500, linewidth=1, linestyle="--",
               label="H = 0.5 (random walk)")

    for i, v in enumerate(H_series.values):
        if np.isnan(v):
            continue
        ax.text(
            v + 0.012,
            i,
            f"{v:.3f}",
            va="center",
            color=GRAY_700,
            fontsize=9,
            family="monospace",
        )

    ax.set_title("Hurst Exponent by Ticker")
    ax.set_xlabel("H")
    ax.set_xlim(0, max(1.0, float(H_series.max(skipna=True)) * 1.15 if H_series.max() else 1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()


_COMPARISON_TABLE_STYLES = [
    {
        "selector": "caption",
        "props": (
            f"caption-side: top; text-align: left; font-weight: 700; "
            f"text-transform: uppercase; letter-spacing: 0.06em; "
            f"color: {GRAY_900}; padding: 12px 0 8px 0; font-size: 13px;"
        ),
    },
    {
        "selector": "th.col_heading, th.row_heading, th.index_name",
        "props": (
            f"background: {GRAY_100}; color: {GRAY_700}; "
            f"border-bottom: 1px solid {GRAY_300}; padding: 8px 12px; "
            "text-align: right; text-transform: uppercase; "
            "letter-spacing: 0.04em; font-size: 11px;"
        ),
    },
    {
        "selector": "td",
        "props": (
            f"border-bottom: 1px solid {GRAY_100}; padding: 8px 12px; "
            "font-family: ui-monospace, SFMono-Regular, Menlo, monospace; "
            "text-align: right;"
        ),
    },
    {"selector": "", "props": "border-collapse: collapse;"},
]


def _render_method_comparison_table(
    static_df: pd.DataFrame, methods: Sequence[str]
) -> None:
    if static_df.empty:
        return
    fmt = {col: "{:.3f}" for col in static_df.columns}
    styled = (
        static_df.style.format(fmt, na_rep="—")
        .set_caption(
            "Full-sample H by method  ·  spread = max-min H across methods"
        )
        .set_table_styles(_COMPARISON_TABLE_STYLES)
    )
    display(styled)


def _render_rolling_disagreement_table(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fmt = {
        "mean spread": "{:.3f}",
        "max spread": "{:.3f}",
        "frac > 0.05": "{:.0%}",
    }
    styled = (
        df.style.format(fmt, na_rep="—")
        .set_caption(
            "Rolling H disagreement across methods  ·  spread = max-min over the window"
        )
        .set_table_styles(_COMPARISON_TABLE_STYLES)
    )
    display(styled)


def _render_method_comparison_multi_window(
    *,
    returns: pd.DataFrame,
    tickers: Sequence[str],
    windows: Sequence[int],
    calibration_sims: int,
) -> None:
    """
    Drive render_method_comparison for each window, then print a combined
    verdict line that names the best method at each window.
    """
    per_window_calib: list[tuple[int, pd.DataFrame]] = []
    for win in windows:
        _render_window_section_header(win)
        calib = calibrate_methods(window=win, n_sims=calibration_sims)
        render_method_comparison(
            returns=returns,
            tickers=tickers,
            window=win,
            calibration_sims=calibration_sims,
            _precomputed_calibration=calib,
        )
        per_window_calib.append((win, calib))

    # Combined verdict across all windows at the bottom for quick scanning.
    if len(windows) > 1:
        display(HTML(
            f'<div style="font-size: 12px; text-transform: uppercase; '
            f'letter-spacing: 0.08em; color: {GRAY_900}; font-weight: 700; '
            f'padding-top: 18px; border-top: 1px solid {GRAY_200}; '
            f'margin-top: 18px;">Combined verdict</div>'
        ))
        _render_verdict(per_window_calib)


def calibrate_methods(
    *,
    window: int = DEFAULT_ROLLING_WINDOW,
    n_sims: int = 500,
    seed: int | None = 0,
    methods: Sequence[str] = ("A", "B", "C"),
) -> pd.DataFrame:
    """
    Calibrate the three methods on iid Gaussian data (true H = 0.5).

    Generates `n_sims` independent return series of length `window` from
    N(0, 1), runs each method, then summarizes the empirical distribution of
    Ĥ around the true value 0.5:

      bias  = mean(Ĥ) − 0.5
      stdev = standard deviation of Ĥ across sims
      RMSE  = sqrt(bias² + stdev²)   ← single ranking number

    Returns a DataFrame indexed by method.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((n_sims, window))

    h_by_method: dict[str, np.ndarray] = {}
    for m in methods:
        Hs = np.empty(n_sims, dtype=float)
        for i in range(n_sims):
            Hs[i] = hurst_exponent(samples[i], method=m).H
        h_by_method[m] = Hs

    rows = []
    for m in methods:
        Hs = h_by_method[m]
        valid = Hs[np.isfinite(Hs)]
        if valid.size == 0:
            rows.append(
                {"method": m, "mean Ĥ": np.nan, "bias": np.nan,
                 "stdev": np.nan, "RMSE": np.nan}
            )
            continue
        mean_h = float(valid.mean())
        bias = mean_h - 0.5
        stdev = float(valid.std(ddof=1)) if valid.size > 1 else float("nan")
        rmse = float(np.sqrt(bias ** 2 + stdev ** 2))
        rows.append(
            {"method": m, "mean Ĥ": mean_h, "bias": bias,
             "stdev": stdev, "RMSE": rmse}
        )

    return pd.DataFrame(rows).set_index("method")


def _render_calibration_table(df: pd.DataFrame, window: int, n_sims: int) -> None:
    if df.empty:
        return
    fmt = {"mean Ĥ": "{:+.4f}", "bias": "{:+.4f}", "stdev": "{:.4f}", "RMSE": "{:.4f}"}
    # Bold the lowest-RMSE row.
    rmse = df["RMSE"]
    winner = rmse.idxmin() if rmse.notna().any() else None

    def _highlight(row: pd.Series) -> list[str]:
        if winner is not None and row.name == winner:
            return [f"background: {GRAY_100}; font-weight: 700;"] * len(row)
        return [""] * len(row)

    styled = (
        df.style.format(fmt, na_rep="—")
        .apply(_highlight, axis=1)
        .set_caption(
            f"Calibration vs iid Gaussian (true H = 0.5)  ·  "
            f"window {window}d  ·  {n_sims} sims  ·  lowest RMSE wins"
        )
        .set_table_styles(_COMPARISON_TABLE_STYLES)
    )
    display(styled)


def _render_verdict(per_window_results: list[tuple[int, pd.DataFrame]]) -> None:
    """One-line summary naming the lowest-RMSE method per window."""
    bits: list[str] = []
    for win, df in per_window_results:
        if df.empty or df["RMSE"].isna().all():
            continue
        best = df["RMSE"].idxmin()
        rmse = df.loc[best, "RMSE"]
        bias = df.loc[best, "bias"]
        bits.append(
            f"<b>{win}d:</b> Method {best} wins "
            f"(RMSE = {rmse:.4f}, bias = {bias:+.4f})"
        )
    if not bits:
        return
    html = (
        f'<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; '
        f'padding: 8px 0 16px 0; color: {GRAY_900}; font-size: 13px;">'
        f'<span style="display: inline-block; font-size: 11px; '
        f'text-transform: uppercase; letter-spacing: 0.08em; '
        f'color: {INDIGO_600}; font-weight: 700; margin-right: 12px;">Verdict</span>'
        f"{' &nbsp;·&nbsp; '.join(bits)}"
        f"</div>"
    )
    display(HTML(html))


def _render_window_section_header(window: int) -> None:
    html = (
        f'<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; '
        f'padding: 18px 0 4px 0; border-top: 1px solid {GRAY_200}; '
        f'margin-top: 12px;">'
        f'<div style="font-size: 11px; text-transform: uppercase; '
        f'letter-spacing: 0.08em; color: {INDIGO_600}; font-weight: 700;">'
        f'Window comparison</div>'
        f'<div style="font-size: 18px; font-weight: 700; color: {GRAY_900}; '
        f'margin-top: 4px;">{window}-day rolling window</div></div>'
    )
    display(HTML(html))


def render_method_comparison(
    returns: pd.DataFrame,
    tickers: Sequence[str],
    *,
    window: int = DEFAULT_ROLLING_WINDOW,
    methods: Sequence[str] = ("A", "B", "C"),
    calibration_sims: int = 500,
    _precomputed_calibration: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compare the three estimator methods for each ticker at one window.

    Renders, in order:
      1. Calibration table — iid-Gaussian bias / stdev / RMSE at this window.
      2. Full-sample H by method, plus per-ticker spread (max-min H).
      3. Rolling H disagreement summary (mean / max spread, frac > 0.05).
      4. Stacked rolling-H plot — one subplot per ticker, one line per method.

    Returns a dict {ticker -> DataFrame of rolling H, one column per method}.
    """
    _apply_mpl_style()
    tickers = [t for t in tickers if t in returns.columns]
    if not tickers:
        return {}

    calib = (
        _precomputed_calibration
        if _precomputed_calibration is not None
        else calibrate_methods(
            window=window, n_sims=calibration_sims, methods=methods
        )
    )
    _render_calibration_table(calib, window=window, n_sims=calibration_sims)
    _render_verdict([(window, calib)])

    # Static (full-sample) fit per (ticker, method) — gives the comparison table.
    static_rows: list[dict] = []
    for t in tickers:
        arr = returns[t].to_numpy()
        row: dict = {"ticker": t}
        h_values: list[float] = []
        for m in methods:
            fit = hurst_exponent(arr, method=m)
            row[f"H_{m}"] = fit.H
            if np.isfinite(fit.H):
                h_values.append(fit.H)
        row["spread (max-min H)"] = (
            float(max(h_values) - min(h_values)) if len(h_values) >= 2 else float("nan")
        )
        static_rows.append(row)
    static_df = pd.DataFrame(static_rows).set_index("ticker")
    _render_method_comparison_table(static_df, methods)

    per_ticker: dict[str, pd.DataFrame] = {}
    for t in tickers:
        cols = {
            m: rolling_hurst(returns[t], window=window, method=m) for m in methods
        }
        per_ticker[t] = pd.DataFrame(cols)

    # Rolling disagreement summary — how much daylight between methods over time?
    rolling_rows: list[dict] = []
    for t in tickers:
        df = per_ticker[t]
        spread = df.max(axis=1) - df.min(axis=1)
        rolling_rows.append(
            {
                "ticker": t,
                "mean spread": float(spread.mean()) if len(spread) else float("nan"),
                "max spread": float(spread.max()) if len(spread) else float("nan"),
                "frac > 0.05": (
                    float((spread > 0.05).mean()) if len(spread) else float("nan")
                ),
            }
        )
    _render_rolling_disagreement_table(pd.DataFrame(rolling_rows).set_index("ticker"))

    n = len(tickers)
    fig, axes = plt.subplots(
        n, 1, figsize=(10, max(2.4, 2.2 * n)), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    for ax, t in zip(axes, tickers):
        df = per_ticker[t]
        for m in methods:
            if m not in df.columns:
                continue
            ax.plot(
                df.index,
                df[m].values,
                color=METHOD_COLORS.get(m, GRAY_700),
                linewidth=1.5,
                label=METHOD_LABELS.get(m, m),
            )
        ax.axhline(0.5, color=GRAY_500, linewidth=1, linestyle="--")
        ax.set_title(t, loc="left")
        ax.set_ylabel("H")
        vals = df.to_numpy().ravel()
        if np.isfinite(vals).any():
            lo = min(0.25, float(np.nanmin(vals)) - 0.05)
            hi = max(0.75, float(np.nanmax(vals)) + 0.05)
            ax.set_ylim(lo, hi)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    fig.suptitle(
        f"Rolling Hurst — method comparison  ·  {window}d window",
        y=1.04,
        fontsize=12,
        fontweight="bold",
        color=GRAY_900,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
    return per_ticker


def _render_rolling_H(r: HurstAnalysisResult) -> None:
    if r.rolling.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for (t, _), color in zip(r.per_ticker.items(), cycle(_LINE_PALETTE)):
        if t not in r.rolling.columns:
            continue
        series = r.rolling[t]
        ax.plot(series.index, series.values, color=color, linewidth=1.6, label=t)

    ax.axhline(0.5, color=GRAY_500, linewidth=1, linestyle="--",
               label="H = 0.5 (random walk)")

    ax.set_title(f"Rolling Hurst  ·  {r.rolling_window}d window")
    ax.set_ylabel("H")
    all_vals = r.rolling.to_numpy().ravel()
    lo = min(0.25, float(np.nanmin(all_vals)) - 0.05) if np.isfinite(all_vals).any() else 0.2
    hi = max(0.75, float(np.nanmax(all_vals)) + 0.05) if np.isfinite(all_vals).any() else 0.8
    ax.set_ylim(lo, hi)
    ax.legend(loc="best", ncol=min(4, len(r.per_ticker) + 1))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0)
    plt.tight_layout()
    plt.show()
