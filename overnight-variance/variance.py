"""
Overnight vs. intraday variance — how much of a name's risk happens while
the market is closed, and whether weekends carry more than a weeknight.

Two log returns per day, anchored on the prior close:
    r_co = ln(open_t  / close_{t-1})   # overnight gap
    r_cc = ln(close_t / close_{t-1})   # full close-to-close day

Everything uses a ZERO-DRIFT variance (mean of squared returns, no mean
subtraction) — we only ever care about ratios, so drift and annualization
cancel. Summing squared returns or averaging them gives the identical ratio
because the window length divides out.

Two products:
  1. Rolling overnight variance ratio:  Sigma r_co^2 / Sigma r_cc^2  over a
     trailing window. >1 means the overnight move exceeds the full-day move
     (the session fades the gap).
  2. Weekend vs. regular overnight premium: variance of clean Fri->Mon gaps
     (exactly Sat+Sun closed, no adjacent holiday) vs. consecutive-business-
     day overnights, bucketed in half-years.

Usage:
    from variance import run_overnight_analysis
    res = run_overnight_analysis(["SPY", "USO"], start="2021-01-01")
    res.ratio      # rolling ratio, date x ticker
    res.premium    # weekend/regular table
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import fetch_ohlc


# ---------------------------------------------------------------- core math

def overnight_returns(px: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """r_co and r_cc panels (date x ticker) from an {'open','close'} dict."""
    prev_close = px["close"].shift(1)
    r_co = np.log(px["open"] / prev_close)
    r_cc = np.log(px["close"] / prev_close)
    return {"r_co": r_co, "r_cc": r_cc}


def rolling_variance_ratio(
    ret: dict[str, pd.DataFrame], window: int = 21
) -> pd.DataFrame:
    """Trailing zero-drift Sigma r_co^2 / Sigma r_cc^2, per ticker."""
    num = (ret["r_co"] ** 2).rolling(window).sum()
    den = (ret["r_cc"] ** 2).rolling(window).sum()
    return (num / den).dropna(how="all")


def _classify_gap(prev: pd.Timestamp, cur: pd.Timestamp) -> str:
    """'regular' (consecutive business days), 'weekend' (clean Fri->Mon), or ''."""
    diff = (cur - prev).days
    if diff == 1:
        return "regular"
    if diff == 3 and prev.weekday() == 4 and cur.weekday() == 0:
        return "weekend"
    return ""  # holiday-affected gap -> excluded from both buckets


def _half(ts: pd.Timestamp) -> str:
    return f"{ts.year}-H{1 if ts.month <= 6 else 2}"


def weekend_premium(
    ret: dict[str, pd.DataFrame], by_bucket: bool = True
) -> pd.DataFrame:
    """
    Per ticker (and half-year bucket, if by_bucket): zero-drift overnight
    variance for weekend vs. regular gaps, and the weekend/regular multiple.
    """
    r_co, r_cc = ret["r_co"], ret["r_cc"]
    rows = []
    for tk in r_co.columns:
        co = r_co[tk].dropna()
        cc = r_cc[tk].reindex(co.index)
        idx = co.index
        recs = []
        for i in range(1, len(idx)):
            kind = _classify_gap(idx[i - 1], idx[i])
            if not kind:
                continue
            recs.append((_half(idx[i]), kind, co.iloc[i], cc.iloc[i]))
        df = pd.DataFrame(recs, columns=["bucket", "kind", "r_co", "r_cc"])
        groups = df.groupby("bucket") if by_bucket else [("ALL", df)]
        for bucket, g in groups:
            rows.append(_summarize(tk, bucket, g))
        rows.append(_summarize(tk, "POOLED", df))
    cols = ["ticker", "bucket", "n_weekend", "n_regular",
            "var_co_weekend", "var_co_regular", "weekend_over_regular",
            "weekend_o_cc", "regular_o_cc"]
    return pd.DataFrame(rows, columns=cols)


def _zdvar(x: pd.Series) -> float:
    return float((x ** 2).mean()) if len(x) else np.nan


def _summarize(ticker: str, bucket: str, g: pd.DataFrame) -> dict:
    wk = g[g.kind == "weekend"]
    rg = g[g.kind == "regular"]
    vw, vr = _zdvar(wk.r_co), _zdvar(rg.r_co)
    return {
        "ticker": ticker, "bucket": bucket,
        "n_weekend": len(wk), "n_regular": len(rg),
        "var_co_weekend": vw, "var_co_regular": vr,
        "weekend_over_regular": vw / vr if vr else np.nan,
        "weekend_o_cc": vw / _zdvar(wk.r_cc) if len(wk) else np.nan,
        "regular_o_cc": vr / _zdvar(rg.r_cc) if len(rg) else np.nan,
    }


# ---------------------------------------------------------------- orchestration

@dataclass
class OvernightResult:
    prices: dict[str, pd.DataFrame]   # {'open','close'}
    returns: dict[str, pd.DataFrame]  # {'r_co','r_cc'}
    ratio: pd.DataFrame               # rolling variance ratio, date x ticker
    premium: pd.DataFrame             # weekend vs regular table (with POOLED rows)
    window: int

    @property
    def pooled(self) -> pd.DataFrame:
        """One row per ticker: the pooled weekend/regular multiple + vol equiv."""
        p = self.premium[self.premium.bucket == "POOLED"].copy()
        p["weekend_vol_mult"] = np.sqrt(p["weekend_over_regular"])
        return p.set_index("ticker")[
            ["n_weekend", "n_regular", "weekend_over_regular", "weekend_vol_mult"]
        ].round(3)


def run_overnight_analysis(
    tickers, *, start, end=None, window: int = 21,
) -> OvernightResult:
    """Fetch OHLC and compute the rolling ratio + weekend premium table."""
    px = fetch_ohlc(tickers, start=start, end=end)
    ret = overnight_returns(px)
    ratio = rolling_variance_ratio(ret, window=window)
    prem = weekend_premium(ret, by_bucket=True)
    return OvernightResult(prices=px, returns=ret, ratio=ratio,
                           premium=prem, window=window)


# ---------------------------------------------------------------- plotting

def plot_ratio(result: OvernightResult, tickers=None, ax=None):
    """Line chart of the rolling overnight variance ratio."""
    import matplotlib.pyplot as plt
    cols = tickers or list(result.ratio.columns)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.6))
    for t in cols:
        s = result.ratio[t].dropna()
        ax.plot(s.index, s.values, lw=1.3, label=f"{t}  (mean {s.mean():.2f})")
    ax.axhline(1.0, ls="--", lw=1, color="#999")
    ax.set_title(f"Overnight variance ratio  Sigma r_co^2 / Sigma r_cc^2  "
                 f"({result.window}d rolling, zero-drift)", fontsize=11)
    ax.set_ylabel("variance ratio")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    return ax


def plot_weekend_premium(result: OvernightResult, ax=None):
    """Bar chart of the pooled weekend/regular variance multiple per ticker."""
    import matplotlib.pyplot as plt
    p = result.pooled.sort_values("weekend_over_regular", ascending=False)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(p.index, p["weekend_over_regular"])
    ax.axhline(1.0, ls="--", lw=1.2, color="#666")
    ax.text(len(p) - 0.5, 1.02, "no premium (1.0x)", ha="right", fontsize=8, color="#666")
    for rect, v in zip(bars, p["weekend_over_regular"]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.03,
                f"{v:.2f}x", ha="center", fontsize=9, weight="bold")
    ax.set_ylabel("weekend / regular overnight variance")
    ax.set_title("Weekend overnight variance premium by asset\n"
                 "(clean Fri->Mon vs. consecutive-business-day gaps, zero-drift)",
                 fontsize=11)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    return ax
