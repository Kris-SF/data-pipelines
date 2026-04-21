"""
Pipeline for fetching aligned daily returns for a weighted portfolio.

Usage:
    from portfolio_vol import Portfolio, fetch_returns

    p = Portfolio.from_weights({"SPY": 0.6, "QQQ": 0.3, "TLT": 0.1})
    rets = fetch_returns(p, start="2024-01-01", end="2025-01-01")

`rets` is a DataFrame of daily log returns, one column per ticker, aligned on
the intersection of available trading days. Downstream code can compute
realized vol, weighted component vol, and the correlation matrix from it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

WEIGHT_TOLERANCE = 1e-6


class PortfolioError(ValueError):
    """Invalid portfolio specification (bad tickers or weights)."""


class FetchError(RuntimeError):
    """Price data could not be fetched or is unusable."""


@dataclass(frozen=True)
class Portfolio:
    """A validated set of tickers with weights that sum to 1.0."""

    weights: pd.Series  # index=ticker (uppercased, unique), dtype=float

    @property
    def tickers(self) -> list[str]:
        return list(self.weights.index)

    def __len__(self) -> int:
        return len(self.weights)

    @classmethod
    def from_weights(
        cls,
        weights: Mapping[str, float],
        *,
        normalize: bool = False,
    ) -> "Portfolio":
        return cls(weights=_validate_weights(weights, normalize=normalize))

    @classmethod
    def from_lists(
        cls,
        tickers: Iterable[str],
        weights: Iterable[float],
        *,
        normalize: bool = False,
    ) -> "Portfolio":
        tickers = list(tickers)
        weights = list(weights)
        if len(tickers) != len(weights):
            raise PortfolioError(
                f"tickers and weights length mismatch: "
                f"{len(tickers)} vs {len(weights)}"
            )
        return cls.from_weights(dict(zip(tickers, weights)), normalize=normalize)


def _validate_weights(
    weights: Mapping[str, float], *, normalize: bool
) -> pd.Series:
    if not weights:
        raise PortfolioError("weights mapping is empty")

    clean: dict[str, float] = {}
    for ticker, w in weights.items():
        if not isinstance(ticker, str) or not ticker.strip():
            raise PortfolioError(f"invalid ticker: {ticker!r}")
        t = ticker.strip().upper()
        if t in clean:
            raise PortfolioError(f"duplicate ticker after normalization: {t}")
        try:
            wf = float(w)
        except (TypeError, ValueError) as e:
            raise PortfolioError(f"{t}: weight {w!r} is not numeric") from e
        if not np.isfinite(wf):
            raise PortfolioError(f"{t}: weight {w!r} is not finite")
        clean[t] = wf

    s = pd.Series(clean, dtype=float).sort_index()
    total = s.sum()

    if normalize:
        if total <= 0:
            raise PortfolioError(
                f"weights sum to {total}; cannot normalize a non-positive total"
            )
        s = s / total
    elif abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise PortfolioError(
            f"weights must sum to 1.0 (got {total:.6f}); "
            "pass normalize=True to rescale automatically"
        )

    return s


def fetch_returns(
    portfolio: Portfolio,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    method: str = "log",
    max_retries: int = 3,
    retry_pause: float = 2.0,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    Fetch a panel of daily returns for the portfolio.

    Parameters
    ----------
    portfolio : Portfolio
    start, end : str | Timestamp
        Inclusive start, exclusive end (yfinance convention).
    method : {"log", "simple"}
        Log returns ln(p_t / p_{t-1}) or simple returns p_t/p_{t-1} - 1.
    max_retries : int
        Retry count for the underlying yfinance call. Each retry waits
        `retry_pause * attempt` seconds before trying again.
    min_obs : int
        After aligning on the common trading calendar, every ticker must
        have at least this many observations; otherwise FetchError.

    Returns
    -------
    DataFrame indexed by date with one column per ticker (same order as
    portfolio.tickers), containing daily returns.
    """
    if method not in ("log", "simple"):
        raise ValueError(f"method must be 'log' or 'simple'; got {method!r}")

    prices = _download_adj_close(
        portfolio.tickers,
        start=start,
        end=end,
        max_retries=max_retries,
        retry_pause=retry_pause,
    )

    missing = [
        t for t in portfolio.tickers
        if t not in prices.columns or prices[t].dropna().empty
    ]
    if missing:
        raise FetchError(
            f"yfinance returned no data for: {missing}. "
            "Check the ticker symbols and date range."
        )

    aligned = prices[portfolio.tickers].dropna(how="any")

    if aligned.empty:
        raise FetchError(
            "no overlapping trading days across tickers after alignment"
        )

    short = {
        t: int(aligned[t].notna().sum())
        for t in portfolio.tickers
        if aligned[t].notna().sum() < min_obs
    }
    if short:
        raise FetchError(
            f"fewer than {min_obs} aligned observations: {short}"
        )

    if method == "log":
        returns = np.log(aligned / aligned.shift(1))
    else:
        returns = aligned.pct_change()

    returns = returns.dropna(how="all")
    log.info(
        "fetched %d rows x %d tickers (%s to %s)",
        len(returns),
        len(portfolio),
        returns.index.min().date(),
        returns.index.max().date(),
    )
    return returns


def _download_adj_close(
    tickers: list[str],
    *,
    start,
    end,
    max_retries: int,
    retry_pause: float,
) -> pd.DataFrame:
    """Download auto-adjusted close prices; returns DataFrame indexed by date."""
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as e:  # network, HTTP, parsing — all retryable
            last_err = e
            log.warning(
                "yfinance download attempt %d/%d raised: %s",
                attempt, max_retries, e,
            )
        else:
            if raw is None or raw.empty:
                last_err = FetchError("yfinance returned an empty DataFrame")
                log.warning(
                    "yfinance download attempt %d/%d returned empty",
                    attempt, max_retries,
                )
            else:
                return _extract_close(raw, tickers)

        if attempt < max_retries:
            time.sleep(retry_pause * attempt)

    raise FetchError(
        f"yfinance download failed after {max_retries} attempts: {last_err}"
    )


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the Close column out of yfinance's response, single or multi-ticker."""
    if isinstance(raw.columns, pd.MultiIndex):
        # With group_by="column", level 0 is the field (Open/High/Low/Close/Volume).
        if "Close" not in raw.columns.get_level_values(0):
            raise FetchError(
                f"yfinance response missing 'Close' field; "
                f"got fields {sorted(set(raw.columns.get_level_values(0)))}"
            )
        close = raw["Close"].copy()
    else:
        # Single-ticker response: flat columns.
        if "Close" not in raw.columns:
            raise FetchError(
                f"yfinance response missing 'Close' column; got {list(raw.columns)}"
            )
        close = raw[["Close"]].copy()
        close.columns = tickers

    close.index = pd.to_datetime(close.index)
    close.index.name = "date"
    return close
