"""
Companion module for the Moontower post "What Does Your Diversifier Do?"

Two questions, one monthly panel:

1. Does a candidate ETF clear the Sharpe hurdle for a small addition to
   SPY?  The hurdle is rho * S_SPY; the ETF table reports standalone
   Sharpe, correlation with SPY, and the Sharpe of an 80/20 mix.
2. Do commodities (DBC, USO) help a 60/40 in months when inflation
   breakevens rise?  Conditional Sharpes in rising / falling months, a
   stability table, and a circular moving-block bootstrap of the
   conditional Sharpe difference (60/30/10 minus 60/40).

Data rules (all enforced here, none of them tunable from the notebook):

* ETFs: Yahoo Finance daily adjusted closes (splits + distributions), sampled
  on the last available trading day of each calendar month.  Monthly return
  = P_t / P_{t-1} - 1.  No filling, no splicing; a missing month inside the
  window raises DataError.
* Cash: RF from the Kenneth French research-factors file, monthly block only,
  divided by 100.
* Breakevens: FRED T10YIE / T5YIE, last non-missing daily value in each
  month; monthly change in bp = 100 * (v_t - v_{t-1}).  Positive = rising
  month, negative = falling, zero = excluded from both.

Usage (see diversifier_study.ipynb):

    from diversifier import StudyConfig, run_study, compare_to_reported
    cfg = StudyConfig.for_mode("reproduce")     # or "refresh"
    result = run_study(cfg)                     # fetch, compute, display, write outputs
    diff = compare_to_reported(result, REPORTED)

The statistics functions at the top of the module are pure numpy/pandas and
are what tests/test_diversifier.py exercises offline.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import re
import time
import zipfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = HERE / "data_cache"
DEFAULT_OUTPUT_DIR = HERE / "output"

ETF_TICKERS: tuple[str, ...] = ("SPY", "GLD", "IEF", "TLT", "AGG", "DBC", "USO")
BENCHMARK = "SPY"
BREAKEVEN_SERIES: tuple[str, ...] = ("T10YIE", "T5YIE")

FF_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
YAHOO_ENDPOINT = (
    "https://query2.finance.yahoo.com/v8/finance/chart/{ticker} "
    "via yfinance.download(auto_adjust=True)"
)

# Portfolios, in the order the post discusses them.
PORTFOLIOS: dict[str, dict[str, float]] = {
    "60/40": {"SPY": 0.60, "AGG": 0.40},
    "60/30/10 DBC": {"SPY": 0.60, "AGG": 0.30, "DBC": 0.10},
    "60/30/10 USO": {"SPY": 0.60, "AGG": 0.30, "USO": 0.10},
}
PORTFOLIO_NAMES: tuple[str, ...] = tuple(PORTFOLIOS)
COMMODITY_PORTFOLIOS: dict[str, str] = {"DBC": "60/30/10 DBC", "USO": "60/30/10 USO"}

# Bootstrap configs, in the order the post reports them: (breakeven series, block length).
DEFAULT_BOOTSTRAP_CONFIGS: tuple[tuple[str, int], ...] = (
    ("T10YIE", 3), ("T10YIE", 6), ("T10YIE", 12), ("T5YIE", 6),
)

ANNUALIZE = math.sqrt(12.0)


class DataError(RuntimeError):
    """A required series is missing, incomplete, or malformed."""


# -------------------------------------------------------------------
# Pure math (offline-testable)
# -------------------------------------------------------------------

def sharpe(excess: Sequence[float] | np.ndarray | pd.Series, ddof: int = 1) -> float:
    """Mean excess return over sample standard deviation (ddof=1). Not annualized."""
    x = np.asarray(excess, dtype=float)
    if x.size < 2:
        return float("nan")
    s = x.std(ddof=ddof)
    return float(x.mean() / s) if s > 0 else float("nan")


def mix_sharpe(
    w_b: float, mu_a: float, sig_a: float, mu_b: float, sig_b: float,
    rf: float, rho: float,
) -> float:
    """
    Sharpe of a two-asset portfolio holding weight w_b in B and 1 - w_b in A.

    Inputs are arithmetic means and vols in the same units (e.g. 0.12 and 0.20).
    """
    w_a = 1.0 - w_b
    mu = w_a * mu_a + w_b * mu_b - rf
    var = (w_a * sig_a) ** 2 + (w_b * sig_b) ** 2 + 2.0 * rho * w_a * w_b * sig_a * sig_b
    return mu / math.sqrt(var)


def sharpe_derivative_at_zero(s_a: float, s_b: float, sig_a: float, sig_b: float, rho: float) -> float:
    """
    d S_P / d w at w = 0 for a portfolio that starts 100% in A and adds B:

        S'_P(0) = (sig_b / sig_a) * (S_b - rho * S_a)

    A small addition of B raises the Sharpe iff S_b > rho * S_a.
    """
    return (sig_b / sig_a) * (s_b - rho * s_a)


def small_addition_hurdle(s_a: float, rho: float) -> float:
    """The Sharpe a candidate must beat for a *small* addition to help: rho * S_A."""
    return rho * s_a


def finite_weight_hurdle(s_a: float, w: float, k: float, rho: float) -> float:
    """
    Sharpe B must beat for an addition of weight w to raise the portfolio Sharpe:

        S_B > S_A * (sqrt(1 + t^2 + 2 rho t) - 1) / t,   t = w k / (1 - w),  k = sig_B / sig_A

    As w -> 0 this collapses to rho * S_A.
    """
    t = w * k / (1.0 - w)
    return s_a * (math.sqrt(1.0 + t * t + 2.0 * rho * t) - 1.0) / t


def expected_heads_times_tails(n: int, p: float = 0.5) -> float:
    """
    E[H * T] for n coin flips with P(heads) = p, H heads and T = n - H tails,
    computed as an exact binomial sum (100 fair flips -> 2475).
    """
    pf = Fraction(p).limit_denominator(10**9)
    qf = 1 - pf
    total = Fraction(0)
    for h in range(n + 1):
        total += math.comb(n, h) * pf**h * qf ** (n - h) * h * (n - h)
    return float(total)


def block_bootstrap_indices(n: int, block_length: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """
    Circular moving-block bootstrap row indices, shape (reps, n).

    For each replicate: draw ceil(n / L) start indices uniformly from 0..n-1,
    take L consecutive rows from each start wrapping modulo n, concatenate,
    truncate to n rows.  Draws come from `rng` so a fixed seed is reproducible.
    """
    if block_length < 1 or block_length > n:
        raise ValueError(f"block_length must be in 1..{n}; got {block_length}")
    n_blocks = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=(reps, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(reps, n_blocks * block_length)[:, :n]


def conditional_sharpe_rows(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Row-wise mean / std (ddof=1) of x over the columns where mask is True.
    x and mask have shape (reps, n).  Rows with fewer than 2 selected columns
    return nan.
    """
    m = mask.astype(float)
    k = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (x * m).sum(axis=1) / k
        dev = (x - mean[:, None]) ** 2
        var = (dev * m).sum(axis=1) / (k - 1.0)
        out = mean / np.sqrt(var)
    out[k < 2] = np.nan
    return out


# -------------------------------------------------------------------
# Portfolio math
# -------------------------------------------------------------------

def portfolio_returns(returns: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Monthly return of a portfolio rebalanced to `weights` every month."""
    missing = [t for t in weights if t not in returns.columns]
    if missing:
        raise KeyError(f"returns panel lacks {missing}")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"weights sum to {total}, not 1")
    out = sum(returns[t] * w for t, w in weights.items())
    return pd.Series(out, index=returns.index)


def compound_calendar_years(monthly: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compound monthly returns into calendar-year returns; only full 12-month years are kept."""
    years = monthly.index.year
    grouped = (1.0 + monthly).groupby(years)
    counts = grouped.size()
    full = counts.index[counts == 12]
    annual = grouped.prod() - 1.0
    return annual.loc[full]


def conditional_stats(excess: pd.Series, mask: pd.Series) -> dict[str, float]:
    """Count, mean excess (%), excess vol (%), and monthly Sharpe over the masked months."""
    sel = excess[mask.reindex(excess.index).fillna(False).astype(bool)]
    return {
        "n": int(sel.size),
        "mean_excess_pct": float(sel.mean() * 100.0),
        "vol_pct": float(sel.std(ddof=1) * 100.0),
        "sharpe": sharpe(sel.to_numpy()),
    }


# -------------------------------------------------------------------
# Retrieval
# -------------------------------------------------------------------

def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _cached_download(
    name: str, source: str, download: Callable[[], str],
    *, cache_dir: Path, use_cache: bool,
) -> tuple[str, dict[str, str]]:
    """
    Return (text, meta) for a raw download, reusing a same-day cache file.

    The cache lives in `cache_dir` (git-ignored).  Readers always download the
    data when they run the notebook; the cache only prevents re-downloading
    within one day, so reproduce and refresh runs on the same day share bytes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"{name}.csv"
    meta_path = cache_dir / f"{name}.meta.json"
    today = dt.date.today().isoformat()
    if use_cache and data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("retrieved_at", "").startswith(today):
            return data_path.read_text(), meta
    text = download()
    meta = {"source": source, "retrieved_at": _utcnow_iso()}
    data_path.write_text(text)
    meta_path.write_text(json.dumps(meta, indent=2))
    return text, meta


def _http_get(url: str, timeout: int = 60, max_retries: int = 3, retry_pause: float = 2.0) -> bytes:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:  # transient network / proxy hiccups
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_pause * attempt)
    raise DataError(f"download failed after {max_retries} attempts: {url}: {last_err}")


def fetch_yahoo_adjusted_closes(
    tickers: Sequence[str], start: str, *,
    cache_dir: Path = DEFAULT_CACHE_DIR, use_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Daily adjusted closes (splits + distributions) for each ticker from `start`
    through the latest available day, one column per ticker.
    """
    from data import _download  # yfinance wrapper with retries; auto_adjust=True

    def download() -> str:
        close = _download(list(tickers), start=start, end=None, max_retries=3, retry_pause=2.0)
        return close.to_csv()

    text, meta = _cached_download(
        f"yahoo_adj_close_{start}", YAHOO_ENDPOINT.format(ticker="{ticker}"),
        download, cache_dir=cache_dir, use_cache=use_cache,
    )
    close = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True)
    close.index = pd.DatetimeIndex(close.index, name="date")
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise DataError(f"Yahoo returned no columns for {missing}")
    return close[list(tickers)].astype(float), meta


def parse_french_monthly(text: str) -> pd.DataFrame:
    """
    Parse the monthly block of a Kenneth French factors CSV into a DataFrame
    indexed by monthly Period, values divided by 100 (decimal returns).

    The file has a monthly block (YYYYMM rows) followed by a blank line and an
    annual block (YYYY rows); only the first YYYYMM run is used.
    """
    lines = text.splitlines()
    header: list[str] | None = None
    rows: list[list[str]] = []
    started = False
    for i, line in enumerate(lines):
        if re.match(r"^\s*\d{6}\s*,", line):
            if not started:
                started = True
                header = [h.strip() for h in lines[i - 1].split(",")]
            rows.append([c.strip() for c in line.split(",")])
        elif started:
            break  # end of the monthly block; the annual block follows
    if not rows or header is None:
        raise DataError("French factors file: no monthly YYYYMM block found")
    cols = header[1:]
    df = pd.DataFrame([r[1:] for r in rows], columns=cols, dtype=float) / 100.0
    df.index = pd.PeriodIndex([pd.Period(r[0], freq="M") for r in rows], name="month")
    if "RF" not in df.columns:
        raise DataError(f"French factors file: no RF column; got {list(df.columns)}")
    return df


def fetch_french_factors(
    *, cache_dir: Path = DEFAULT_CACHE_DIR, use_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    def download() -> str:
        z = zipfile.ZipFile(io.BytesIO(_http_get(FF_URL)))
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise DataError(f"French zip has no CSV member: {z.namelist()}")
        return z.read(names[0]).decode("latin-1")

    text, meta = _cached_download("french_factors", FF_URL, download, cache_dir=cache_dir, use_cache=use_cache)
    return parse_french_monthly(text), meta


def parse_fred_csv(text: str, series: str) -> pd.Series:
    """Daily FRED series as a float Series (missing values as NaN) indexed by date."""
    df = pd.read_csv(io.StringIO(text), na_values=["."], keep_default_na=True)
    date_col = df.columns[0]
    if series not in df.columns:
        raise DataError(f"FRED CSV for {series} has columns {list(df.columns)}")
    s = pd.Series(df[series].astype(float).to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(df[date_col]), name="date"), name=series)
    return s


def fetch_fred_series(
    series: str, *, cache_dir: Path = DEFAULT_CACHE_DIR, use_cache: bool = True,
) -> tuple[pd.Series, dict[str, str]]:
    url = FRED_URL.format(series=series)
    text, meta = _cached_download(
        f"fred_{series}", url, lambda: _http_get(url).decode("utf-8"),
        cache_dir=cache_dir, use_cache=use_cache,
    )
    return parse_fred_csv(text, series), meta


# -------------------------------------------------------------------
# Monthly sampling and the panel
# -------------------------------------------------------------------

def month_end_observations(daily: pd.DataFrame | pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Last non-missing observation in each calendar month, per column.

    Returns (values, dates): both indexed by monthly Period, one column per
    input column; `dates` holds the actual observation date used.
    """
    if isinstance(daily, pd.Series):
        daily = daily.to_frame()
    values: dict[str, pd.Series] = {}
    dates: dict[str, pd.Series] = {}
    for col in daily.columns:
        s = daily[col].dropna()
        months = s.index.to_period("M")
        last_pos = pd.Series(np.arange(len(s)), index=s.index).groupby(months).max()
        picked = s.iloc[last_pos.to_numpy()]
        values[col] = pd.Series(picked.to_numpy(), index=last_pos.index)
        dates[col] = pd.Series(picked.index, index=last_pos.index)
    vals = pd.DataFrame(values)
    dts = pd.DataFrame(dates)
    vals.index.name = dts.index.name = "month"
    return vals, dts


def _require_months(monthly: pd.Series, first: pd.Period, last: pd.Period, name: str) -> None:
    expected = pd.period_range(first, last, freq="M")
    present = monthly.dropna().index
    missing = expected.difference(present)
    if len(missing):
        raise DataError(
            f"{name} is missing {len(missing)} month(s) inside {first}..{last}: "
            f"{[str(m) for m in missing[:12]]}{' ...' if len(missing) > 12 else ''}"
        )


def _last_complete_month(monthly_index: pd.PeriodIndex, as_of: dt.date) -> pd.Period:
    """Latest month present in the index whose calendar month has fully elapsed as of `as_of`."""
    current = pd.Period(as_of, freq="M")
    complete = [m for m in monthly_index if m < current]
    if not complete:
        raise DataError("no complete months in series")
    return max(complete)


@dataclass
class Panel:
    """The monthly panel every statistic is computed from."""

    returns: pd.DataFrame          # ETF monthly returns, Period index
    rf: pd.Series                  # French RF, decimal, same index
    breakeven_levels: pd.DataFrame  # month-end T10YIE / T5YIE levels (percent)
    breakeven_changes_bp: pd.DataFrame  # monthly change in bp, same index as returns
    price_dates: pd.DataFrame      # long: ticker, month, price_date, adj_close, retrieved_at, source
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def months(self) -> pd.PeriodIndex:
        return self.returns.index

    @property
    def n(self) -> int:
        return len(self.returns)

    def excess(self, weights: Mapping[str, float]) -> pd.Series:
        return portfolio_returns(self.returns, weights) - self.rf

    def frame(self) -> pd.DataFrame:
        """One row per month: ETF returns, RF, and both breakeven changes (bp)."""
        chg = self.breakeven_changes_bp.add_prefix("d").add_suffix("_bp")
        return pd.concat([self.returns, self.rf.rename("RF"), chg], axis=1)


@dataclass
class StudyConfig:
    mode: str = "reproduce"
    start: str = "2006-08"              # first return month
    end: str | None = "2026-07"         # last return month; None = latest complete month (refresh)
    tickers: tuple[str, ...] = ETF_TICKERS
    benchmark: str = BENCHMARK
    breakeven_series: tuple[str, ...] = BREAKEVEN_SERIES
    n_boot: int = 20_000
    seed_base: int = 20260921
    bootstrap_configs: tuple[tuple[str, int], ...] = DEFAULT_BOOTSTRAP_CONFIGS
    cache_dir: Path = DEFAULT_CACHE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    use_cache: bool = True

    @classmethod
    def for_mode(cls, mode: str, **overrides: Any) -> "StudyConfig":
        if mode == "reproduce":
            return cls(mode="reproduce", end="2026-07", **overrides)
        if mode == "refresh":
            return cls(mode="refresh", end=None, **overrides)
        raise ValueError(f"MODE must be 'reproduce' or 'refresh'; got {mode!r}")

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.mode


def load_panel(cfg: StudyConfig, *, verbose: bool = True) -> Panel:
    """Download every series, sample month-ends, validate the window, build the panel."""
    first = pd.Period(cfg.start, freq="M")
    base_month = first - 1                      # starting price observation (Jul 2006)
    fetch_start = (base_month.to_timestamp(how="start") - pd.Timedelta(days=45)).strftime("%Y-%m-%d")

    close, yahoo_meta = fetch_yahoo_adjusted_closes(cfg.tickers, fetch_start, cache_dir=cfg.cache_dir, use_cache=cfg.use_cache)
    ff, ff_meta = fetch_french_factors(cache_dir=cfg.cache_dir, use_cache=cfg.use_cache)
    fred: dict[str, pd.Series] = {}
    fred_meta: dict[str, dict[str, str]] = {}
    for s in cfg.breakeven_series:
        fred[s], fred_meta[s] = fetch_fred_series(s, cache_dir=cfg.cache_dir, use_cache=cfg.use_cache)

    me_close, me_dates = month_end_observations(close)
    be_daily = pd.DataFrame(fred)
    me_be, _ = month_end_observations(be_daily)

    # --- window end -------------------------------------------------------
    as_of = dt.datetime.fromisoformat(yahoo_meta["retrieved_at"]).date()
    last_complete = {t: _last_complete_month(me_close[t].dropna().index, as_of) for t in cfg.tickers}
    last_complete["French RF"] = ff["RF"].dropna().index.max()
    for s in cfg.breakeven_series:
        last_complete[f"FRED {s}"] = _last_complete_month(me_be[s].dropna().index, as_of)
    latest_common = min(last_complete.values())
    binding = [k for k, v in last_complete.items() if v == latest_common]

    if cfg.end is None:
        last = latest_common
    else:
        last = pd.Period(cfg.end, freq="M")
        if last > latest_common:
            raise DataError(
                f"requested end {last} is later than the latest complete month {latest_common} "
                f"(binding: {binding})"
            )
    if verbose:
        print(f"Window: {first} .. {last}  ({(last - first).n + 1} monthly returns; base price month {base_month})")
        if cfg.end is None:
            print(f"Latest complete month in every series: {latest_common}; binding constraint: {', '.join(binding)}")
            print("  last complete month by series: " + ", ".join(f"{k}={v}" for k, v in last_complete.items()))

    # --- validate and trim ---------------------------------------------------
    for t in cfg.tickers:
        _require_months(me_close[t], base_month, last, f"Yahoo {t}")
    _require_months(ff["RF"], first, last, "French RF")
    for s in cfg.breakeven_series:
        _require_months(me_be[s], base_month, last, f"FRED {s}")

    px = me_close.loc[base_month:last, list(cfg.tickers)]
    returns = (px / px.shift(1) - 1.0).loc[first:last]
    rf = ff["RF"].loc[first:last].rename("RF")
    levels = me_be.loc[base_month:last, list(cfg.breakeven_series)]
    changes_bp = (levels.diff() * 100.0).loc[first:last]

    if not (returns.index.equals(rf.index) and returns.index.equals(changes_bp.index)):
        raise DataError("index mismatch after trimming; this should not happen")
    if returns.isna().any().any() or rf.isna().any() or changes_bp.isna().any().any():
        raise DataError("NaN inside the trimmed panel; this should not happen")

    # --- price_dates.csv ----------------------------------------------------
    rows = []
    for t in cfg.tickers:
        for m in px.index:
            rows.append({
                "ticker": t, "month": str(m),
                "price_date": pd.Timestamp(me_dates.loc[m, t]).date().isoformat(),
                "adj_close": float(px.loc[m, t]),
                "retrieved_at": yahoo_meta["retrieved_at"],
                "source": YAHOO_ENDPOINT.format(ticker=t),
            })
    price_dates = pd.DataFrame(rows)

    meta = {
        "mode": cfg.mode,
        "start": str(first), "end": str(last), "base_price_month": str(base_month),
        "n_months": int(len(returns)),
        "latest_complete_month": str(latest_common),
        "binding_constraint": binding,
        "last_complete_by_series": {k: str(v) for k, v in last_complete.items()},
        "sources": {
            "yahoo": yahoo_meta, "french": ff_meta,
            **{f"fred_{s}": fred_meta[s] for s in cfg.breakeven_series},
        },
    }
    return Panel(returns=returns, rf=rf, breakeven_levels=levels, breakeven_changes_bp=changes_bp,
                 price_dates=price_dates, meta=meta)


# -------------------------------------------------------------------
# Study outputs
# -------------------------------------------------------------------

def etf_table(panel: Panel, benchmark: str = BENCHMARK, addition_weight: float = 0.20) -> pd.DataFrame:
    """
    Per ETF: annualized standalone Sharpe, correlation of monthly excess returns
    with the benchmark, annualized Sharpe of (1-w) benchmark / w ETF, the
    small-addition hurdle rho * S_bench, and pass/fail.
    """
    ex = panel.returns.sub(panel.rf, axis=0)
    s_bench = sharpe(ex[benchmark]) * ANNUALIZE
    rows = []
    for t in panel.returns.columns:
        s_t = sharpe(ex[t]) * ANNUALIZE
        rho = float(ex[t].corr(ex[benchmark]))
        mix = panel.excess({benchmark: 1.0 - addition_weight, t: addition_weight}) if t != benchmark else ex[benchmark]
        hurdle = small_addition_hurdle(s_bench, rho)
        rows.append({
            "ticker": t,
            "sharpe_ann": s_t,
            "corr_with_spy": rho,
            "sharpe_80_20_ann": sharpe(mix) * ANNUALIZE,
            "hurdle": hurdle,
            "passes_hurdle": (s_t > hurdle) if t != benchmark else None,
        })
    return pd.DataFrame(rows).set_index("ticker")


def weak_bond_screen(panel: Panel, bond: str = "AGG", pct: float = 25.0) -> dict[str, Any]:
    """
    Calendar-year returns (full years only) for AGG, 60/40 and both 60/30/10s;
    select years where AGG's annual return <= its `pct` percentile (numpy
    linear interpolation) and count how often each 60/30/10 beat 60/40.
    """
    monthly = pd.DataFrame({bond: panel.returns[bond]})
    for name, w in PORTFOLIOS.items():
        monthly[name] = portfolio_returns(panel.returns, w)
    annual = compound_calendar_years(monthly)
    threshold = float(np.percentile(annual[bond].to_numpy(), pct))
    selected = annual[annual[bond] <= threshold]
    table = selected.copy()
    for etf, pname in COMMODITY_PORTFOLIOS.items():
        table[f"{etf} beats 60/40"] = selected[pname] > selected["60/40"]
    return {
        "annual": annual,
        "threshold": threshold,
        "years": [int(y) for y in selected.index],
        "table": table,
        "wins": {etf: int(table[f"{etf} beats 60/40"].sum()) for etf in COMMODITY_PORTFOLIOS},
        "n_years": int(len(selected)),
    }


def breakeven_conditional(panel: Panel, series: str = "T10YIE") -> dict[str, Any]:
    """Conditional stats for each portfolio in rising / falling months of `series`."""
    chg = panel.breakeven_changes_bp[series]
    masks = {"rising": chg > 0, "falling": chg < 0}
    out: dict[str, Any] = {"n_unchanged": int((chg == 0).sum()), "series": series}
    for regime, mask in masks.items():
        rows = {name: conditional_stats(panel.excess(w), mask) for name, w in PORTFOLIOS.items()}
        out[regime] = pd.DataFrame(rows).T[["n", "mean_excess_pct", "vol_pct", "sharpe"]]
        out[regime]["n"] = out[regime]["n"].astype(int)
    return out


def full_sample_sharpes(panel: Panel) -> pd.Series:
    """Unannualized full-window Sharpe for each portfolio."""
    return pd.Series({name: sharpe(panel.excess(w)) for name, w in PORTFOLIOS.items()}, name="sharpe")


def _half_split(months: pd.PeriodIndex) -> tuple[pd.PeriodIndex, pd.PeriodIndex]:
    half = len(months) // 2
    return months[:half], months[half:]


def stability_table(panel: Panel) -> pd.DataFrame:
    """Rising-month conditional Sharpes for four cuts: 10y full, 10y each half, 5y full."""
    first_half, second_half = _half_split(panel.months)
    cuts = [
        ("10y_full", "T10YIE", panel.months),
        ("10y_first_half", "T10YIE", first_half),
        ("10y_second_half", "T10YIE", second_half),
        ("5y_full", "T5YIE", panel.months),
    ]
    rows = []
    for key, series, months in cuts:
        chg = panel.breakeven_changes_bp.loc[months, series]
        mask = chg > 0
        row: dict[str, Any] = {
            "cut": key, "series": series,
            "window": f"{months[0]} – {months[-1]}",
            "n_rising": int(mask.sum()),
        }
        for name, w in PORTFOLIOS.items():
            row[name] = sharpe(panel.excess(w).loc[months][mask])
        rows.append(row)
    return pd.DataFrame(rows).set_index("cut")


def bootstrap_sharpe_diff(
    panel: Panel, series: str, block_length: int, *,
    reps: int = 20_000, seed_base: int = 20260921, chunk: int = 4000,
) -> dict[str, Any]:
    """
    Circular moving-block bootstrap of the rising-month conditional Sharpe
    difference (60/30/10 minus 60/40) for DBC and USO.

    Each resample re-draws rows of the original monthly panel (returns, RF and
    breakeven changes travel together, changes are never recomputed across
    block boundaries), selects the rising rows of that resample, and computes
    all three conditional Sharpes from the same rows.  rng = default_rng(seed_base + L).
    """
    n = panel.n
    rng = np.random.default_rng(seed_base + block_length)
    idx = block_bootstrap_indices(n, block_length, reps, rng)

    chg = panel.breakeven_changes_bp[series].to_numpy()
    ex = {name: panel.excess(w).to_numpy() for name, w in PORTFOLIOS.items()}

    diffs = {etf: np.empty(reps) for etf in COMMODITY_PORTFOLIOS}
    n_rising = np.empty(reps, dtype=int)
    for lo in range(0, reps, chunk):
        sl = slice(lo, min(lo + chunk, reps))
        ii = idx[sl]
        mask = chg[ii] > 0
        n_rising[sl] = mask.sum(axis=1)
        base = conditional_sharpe_rows(ex["60/40"][ii], mask)
        for etf, pname in COMMODITY_PORTFOLIOS.items():
            diffs[etf][sl] = conditional_sharpe_rows(ex[pname][ii], mask) - base

    point = point_estimates(panel, series)
    out: dict[str, Any] = {
        "series": series, "block_length": block_length, "reps": reps,
        "seed": seed_base + block_length, "n_blocks": math.ceil(n / block_length),
        "n_rising_mean": float(n_rising.mean()),
    }
    for etf in COMMODITY_PORTFOLIOS:
        lo_, hi_ = np.percentile(diffs[etf], [2.5, 97.5])
        out[etf] = {
            "point": point[etf], "ci_low": float(lo_), "ci_high": float(hi_),
            "n_nan": int(np.isnan(diffs[etf]).sum()),
        }
    return out


def point_estimates(panel: Panel, series: str = "T10YIE") -> dict[str, float]:
    """Full-sample rising-month conditional Sharpe difference (60/30/10 minus 60/40)."""
    mask = panel.breakeven_changes_bp[series] > 0
    base = sharpe(panel.excess(PORTFOLIOS["60/40"])[mask])
    return {etf: sharpe(panel.excess(PORTFOLIOS[p])[mask]) - base for etf, p in COMMODITY_PORTFOLIOS.items()}


@dataclass
class StudyResult:
    config: StudyConfig
    panel: Panel
    etf: pd.DataFrame
    weak_bond: dict[str, Any]
    cond_10y: dict[str, Any]
    full_sharpes: pd.Series
    stability: pd.DataFrame
    bootstrap: list[dict[str, Any]]
    tables: dict[str, Any] = field(default_factory=dict)  # post_tables.json contents
    run_dir: Path | None = None


def compute_all(panel: Panel, cfg: StudyConfig, *, verbose: bool = True) -> StudyResult:
    etf = etf_table(panel, cfg.benchmark)
    weak = weak_bond_screen(panel)
    cond = breakeven_conditional(panel, "T10YIE")
    full = full_sample_sharpes(panel)
    stab = stability_table(panel)
    boots = []
    for series, L in cfg.bootstrap_configs:
        if verbose:
            print(f"bootstrap {series} L={L}: {cfg.n_boot:,} reps ...", end=" ", flush=True)
        boots.append(bootstrap_sharpe_diff(panel, series, L, reps=cfg.n_boot, seed_base=cfg.seed_base))
        if verbose:
            print("done")
    res = StudyResult(config=cfg, panel=panel, etf=etf, weak_bond=weak, cond_10y=cond,
                      full_sharpes=full, stability=stab, bootstrap=boots)
    res.tables = post_tables(res)
    return res


# -------------------------------------------------------------------
# post_tables.json: same shape as the notebook's REPORTED dict
# -------------------------------------------------------------------

def _r(x: float, nd: int) -> float:
    return float(round(float(x), nd))


def post_tables(res: StudyResult) -> dict[str, Any]:
    """Every number in the post, keyed by the table it belongs to (unrounded floats)."""
    p = res.panel
    etf = {
        t: {
            "sharpe_ann": float(r.sharpe_ann),
            "corr_with_spy": float(r.corr_with_spy),
            "sharpe_80_20_ann": float(r.sharpe_80_20_ann),
            "hurdle": float(r.hurdle),
            "passes_hurdle": None if r.passes_hurdle is None else bool(r.passes_hurdle),
        }
        for t, r in res.etf.iterrows()
    }
    cond = {}
    for regime in ("rising", "falling"):
        df = res.cond_10y[regime]
        cond[regime] = {
            "n": int(df["n"].iloc[0]),
            **{
                stat: {name: float(df.loc[name, stat]) for name in PORTFOLIO_NAMES}
                for stat in ("mean_excess_pct", "vol_pct", "sharpe")
            },
        }
    cond["n_unchanged"] = int(res.cond_10y["n_unchanged"])
    stability = {
        key: {
            "series": row["series"], "window": row["window"], "n_rising": int(row["n_rising"]),
            **{name: float(row[name]) for name in PORTFOLIO_NAMES},
        }
        for key, row in res.stability.iterrows()
    }
    boot = {
        f"{'10y' if b['series'] == 'T10YIE' else '5y'}_L{b['block_length']}": {
            "series": b["series"], "block_length": b["block_length"], "reps": b["reps"], "seed": b["seed"],
            **{etf: {"ci_low": b[etf]["ci_low"], "ci_high": b[etf]["ci_high"]} for etf in COMMODITY_PORTFOLIOS},
        }
        for b in res.bootstrap
    }
    point = point_estimates(p, "T10YIE")
    weak = res.weak_bond
    return {
        "meta": {**p.meta, "portfolios": PORTFOLIOS, "generated_at": _utcnow_iso()},
        "table1_etf": etf,
        "table2_weak_bond_years": {
            "agg_threshold_pct25": float(weak["threshold"]),
            "years": weak["years"],
            "annual_returns": {
                str(y): {c: float(v) for c, v in row.items()} for y, row in weak["annual"].iterrows()
            },
            "selected": {
                str(y): {c: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for c, v in row.items()}
                for y, row in weak["table"].iterrows()
            },
            "wins_vs_60_40": weak["wins"],
        },
        "table3_rising_10y": cond["rising"],
        "table4_falling_10y": {**cond["falling"], "n_unchanged": cond["n_unchanged"]},
        "table5_full_sample_sharpe": {name: float(v) for name, v in res.full_sharpes.items()},
        "table6_stability": stability,
        "table7_bootstrap": {"point_estimates": point, "intervals": boot},
    }


# -------------------------------------------------------------------
# Comparison against the published values
# -------------------------------------------------------------------

TOL_SHARPE = 0.005
TOL_PCT = 0.02
TOL_CI = 0.01


def compare_to_reported(res: StudyResult, reported: Mapping[str, Any]) -> pd.DataFrame:
    """
    Diff the computed post tables against the notebook's REPORTED dict.

    Returns one row per published number with columns
    table, item, reported, computed, diff, tol, status ('pass' / 'FLAG').
    Counts and year sets must match exactly.
    """
    t = res.tables
    rows: list[dict[str, Any]] = []

    def add(table: str, item: str, rep: Any, comp: Any, tol: float | None) -> None:
        if tol is None:
            ok = rep == comp
            diff = None if ok else f"{comp!r} vs {rep!r}"
        else:
            diff = float(comp) - float(rep)
            ok = abs(diff) <= tol + 1e-12
        rows.append({"table": table, "item": item, "reported": rep, "computed": comp,
                     "diff": diff, "tol": tol, "status": "pass" if ok else "FLAG"})

    for tkr, (s, rho, s8020) in reported["etf_table"].items():
        e = t["table1_etf"][tkr]
        add("1 ETF", f"{tkr} Sharpe (ann)", s, e["sharpe_ann"], TOL_SHARPE)
        add("1 ETF", f"{tkr} corr w/ SPY", rho, e["corr_with_spy"], TOL_SHARPE)
        add("1 ETF", f"{tkr} 80/20 Sharpe (ann)", s8020, e["sharpe_80_20_ann"], TOL_SHARPE)
    add("1 ETF", "DBC hurdle rho*S_SPY", reported["dbc_hurdle"], t["table1_etf"]["DBC"]["hurdle"], TOL_SHARPE)
    for tkr in ("DBC", "USO"):
        add("1 ETF", f"{tkr} passes hurdle", False, t["table1_etf"][tkr]["passes_hurdle"], None)

    add("2 weak-bond", "years", sorted(reported["weak_bond_years"]), sorted(t["table2_weak_bond_years"]["years"]), None)
    for etf, w in reported["weak_bond_wins"].items():
        add("2 weak-bond", f"{etf} wins vs 60/40", w, t["table2_weak_bond_years"]["wins_vs_60_40"][etf], None)

    for key, table_name, label in (("rising_10y", "table3_rising_10y", "3 rising 10y"),
                                   ("falling_10y", "table4_falling_10y", "4 falling 10y")):
        rep = reported[key]
        comp = t[table_name]
        add(label, "n months", rep["n"], comp["n"], None)
        if "n_unchanged" in rep:
            add(label, "n unchanged", rep["n_unchanged"], comp["n_unchanged"], None)
        for stat, tol in (("mean_excess_pct", TOL_PCT), ("vol_pct", TOL_PCT), ("sharpe", TOL_SHARPE)):
            for name, val in zip(PORTFOLIO_NAMES, rep[stat]):
                add(label, f"{name} {stat}", val, comp[stat][name], tol)

    for name, val in zip(PORTFOLIO_NAMES, reported["full_sample_sharpe"]):
        add("5 full-sample", f"{name} Sharpe (monthly)", val, t["table5_full_sample_sharpe"][name], TOL_SHARPE)

    for key, (n, *sh) in reported["stability"].items():
        comp = t["table6_stability"][key]
        add("6 stability", f"{key} n rising", n, comp["n_rising"], None)
        for name, val in zip(PORTFOLIO_NAMES, sh):
            add("6 stability", f"{key} {name}", val, comp[name], TOL_SHARPE)

    for etf, val in reported["bootstrap_point"].items():
        add("7 bootstrap", f"{etf} point estimate", val, t["table7_bootstrap"]["point_estimates"][etf], TOL_SHARPE)
    for key, per_etf in reported["bootstrap_ci"].items():
        comp = t["table7_bootstrap"]["intervals"][key]
        for etf, (lo, hi) in per_etf.items():
            add("7 bootstrap", f"{key} {etf} 2.5%", lo, comp[etf]["ci_low"], TOL_CI)
            add("7 bootstrap", f"{key} {etf} 97.5%", hi, comp[etf]["ci_high"], TOL_CI)

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Output files
# -------------------------------------------------------------------

def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Period, pd.Timestamp, Path)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_outputs(res: StudyResult, run_dir: Path | None = None) -> Path:
    """Write post_tables.json, price_dates.csv, monthly_panel.csv into the run folder."""
    run_dir = Path(run_dir or res.config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "post_tables.json").write_text(json.dumps(res.tables, indent=2, default=_json_default))
    res.panel.price_dates.to_csv(run_dir / "price_dates.csv", index=False)
    res.panel.frame().to_csv(run_dir / "monthly_panel.csv")
    res.run_dir = run_dir
    return run_dir


# -------------------------------------------------------------------
# Display
# -------------------------------------------------------------------

def _display(obj: Any) -> None:
    try:
        from IPython.display import display
        display(obj)
    except ImportError:  # plain python
        print(obj)


def _heading(text: str) -> None:
    try:
        from IPython.display import HTML, display
        display(HTML(f'<h3 style="margin:18px 0 6px 0">{text}</h3>'))
    except ImportError:
        print(f"\n== {text} ==")


def display_results(res: StudyResult) -> None:
    """Print every table in the order the post presents them."""
    p = res.panel
    _heading(f"Panel: {p.meta['start']} – {p.meta['end']}, {p.n} monthly returns")
    _heading("1. ETF table (annualized Sharpe, corr with SPY, 80/20 Sharpe, hurdle)")
    _display(res.etf.round(3))

    wb = res.weak_bond
    _heading(f"2. Weak-bond years: AGG annual return ≤ 25th pct ({wb['threshold']*100:.2f}%) → {wb['years']}")
    shown = wb["table"].copy()
    for col in shown.columns:
        if shown[col].dtype != bool:
            shown[col] = (shown[col] * 100).round(2)
    _display(shown)  # annual returns in %, plus the beat-60/40 flags
    print(f"60/30/10 beat 60/40 in weak-bond years: DBC {wb['wins']['DBC']} of {wb['n_years']}, USO {wb['wins']['USO']} of {wb['n_years']}")

    c = res.cond_10y
    _heading(f"3. Rising 10y-breakeven months (n = {int(c['rising']['n'].iloc[0])})")
    _display(c["rising"].round(3))
    _heading(f"4. Falling 10y-breakeven months (n = {int(c['falling']['n'].iloc[0])}; unchanged months excluded: {c['n_unchanged']})")
    _display(c["falling"].round(3))

    _heading(f"5. Full-sample monthly Sharpes ({p.n} months, not annualized)")
    _display(res.full_sharpes.round(3).to_frame())

    _heading("6. Stability: rising-month conditional Sharpes")
    _display(res.stability.round(3))

    _heading("7. Bootstrap 95% intervals for the rising-month Sharpe difference (60/30/10 − 60/40)")
    point = res.tables["table7_bootstrap"]["point_estimates"]
    print(f"Full-sample point estimates (10y rising months): DBC {point['DBC']:+.3f}, USO {point['USO']:+.3f}")
    _display(bootstrap_table(res))


def bootstrap_table(res: StudyResult) -> pd.DataFrame:
    rows = []
    for b in res.bootstrap:
        rows.append({
            "series": b["series"], "L": b["block_length"], "reps": b["reps"], "seed": b["seed"],
            "DBC 2.5%": b["DBC"]["ci_low"], "DBC 97.5%": b["DBC"]["ci_high"],
            "USO 2.5%": b["USO"]["ci_low"], "USO 97.5%": b["USO"]["ci_high"],
        })
    return pd.DataFrame(rows).round(3)


def display_comparison(diff: pd.DataFrame) -> None:
    n_flag = int((diff["status"] == "FLAG").sum())
    _heading(f"Reproduction check: {len(diff) - n_flag} pass, {n_flag} flagged")
    try:
        from IPython.display import display
        styled = diff.style.apply(
            lambda col: ["background-color:#FDE68A;font-weight:600" if v == "FLAG" else "" for v in col],
            subset=["status"],
        ).format({"reported": _fmt, "computed": _fmt, "diff": _fmt, "tol": _fmt}, na_rep="")
        display(styled)
    except ImportError:
        print(diff.to_string())


def _fmt(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}" if not math.isnan(v) else ""
    return "" if v is None else str(v)


# -------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------

def run_study(cfg: StudyConfig, *, show: bool = True, write: bool = True) -> StudyResult:
    """Fetch the data, compute every table, display them, write the run folder."""
    panel = load_panel(cfg, verbose=show)
    res = compute_all(panel, cfg, verbose=show)
    if write:
        run_dir = write_outputs(res)
        if show:
            print(f"Outputs written to {run_dir}: post_tables.json, price_dates.csv, monthly_panel.csv")
    if show:
        display_results(res)
    return res
