"""
Daily-returns fetcher shared by quant analyses in this folder.

Wraps yfinance's bulk download with retries, ticker validation, and
trading-day alignment. No portfolio/weights concept — just give it a
list of tickers and a date range and you get a clean returns panel.

Usage:
    from data import fetch_daily_returns
    returns = fetch_daily_returns(["SPY", "QQQ"], start="2020-01-01", end="2025-01-01")
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """yfinance data could not be retrieved or aligned."""


def fetch_daily_returns(
    tickers: Iterable[str],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    method: str = "log",
    max_retries: int = 3,
    retry_pause: float = 2.0,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    DataFrame of daily returns (one column per ticker, log by default),
    aligned on the intersection of trading days.
    """
    clean = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    if not clean:
        raise ValueError("no valid tickers given")
    if len(set(clean)) != len(clean):
        raise ValueError("duplicate tickers (case-insensitive)")
    if method not in ("log", "simple"):
        raise ValueError(f"method must be 'log' or 'simple'; got {method!r}")

    prices = _download(
        clean, start=start, end=end,
        max_retries=max_retries, retry_pause=retry_pause,
    )

    missing = [
        t for t in clean
        if t not in prices.columns or prices[t].dropna().empty
    ]
    if missing:
        raise FetchError(
            f"yfinance returned no data for: {missing}. "
            "Check the ticker symbols and date range."
        )

    aligned = prices[clean].dropna(how="any")
    if aligned.empty:
        raise FetchError("no overlapping trading days across tickers")

    short = {
        t: int(aligned[t].notna().sum())
        for t in clean
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
        len(returns), len(clean),
        returns.index.min().date(), returns.index.max().date(),
    )
    return returns


def _download(tickers, *, start, end, max_retries, retry_pause):
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
        except Exception as e:
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
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise FetchError(
                f"yfinance response missing 'Close' field; "
                f"got {sorted(set(raw.columns.get_level_values(0)))}"
            )
        close = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise FetchError(
                f"yfinance response missing 'Close' column; got {list(raw.columns)}"
            )
        close = raw[["Close"]].copy()
        close.columns = tickers
    close.index = pd.to_datetime(close.index)
    close.index.name = "date"
    return close
