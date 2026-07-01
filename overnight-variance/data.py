"""
OHLC fetcher for the overnight-variance study.

Unlike the close-only fetcher in ../quant-analysis/data.py, this one keeps
BOTH Open and Close — the overnight gap r_co = ln(open_t / close_{t-1})
needs the official open.

Uses auto_adjust=True on purpose: back-adjusting for splits AND dividends
means an ex-dividend price drop does NOT masquerade as an overnight gap.
That matters a lot here — dividend payers (TLT pays monthly, plus SPY/GLD/
FXE) would otherwise inject fake overnight variance on every ex-div date.

Usage:
    from data import fetch_ohlc
    px = fetch_ohlc(["SPY", "USO"], start="2021-01-01", end="2024-01-01")
    px["open"], px["close"]   # two aligned DataFrames, one column per ticker
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """yfinance data could not be retrieved or parsed."""


def fetch_ohlc(
    tickers: Iterable[str],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    auto_adjust: bool = True,
    max_retries: int = 3,
    retry_pause: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily Open and Close for each ticker.

    Returns {"open": DataFrame, "close": DataFrame}, both indexed by date
    with one column per ticker. Each ticker keeps its own trading calendar
    (no cross-ticker alignment) so single-name history is never truncated
    by another name's gaps; NaNs mark days a given ticker did not trade.
    """
    clean = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    if not clean:
        raise ValueError("no valid tickers given")
    if len(set(clean)) != len(clean):
        raise ValueError("duplicate tickers (case-insensitive)")

    raw = _download(
        clean, start=start, end=end, auto_adjust=auto_adjust,
        max_retries=max_retries, retry_pause=retry_pause,
    )
    opens = _field(raw, "Open", clean)
    closes = _field(raw, "Close", clean)

    missing = [t for t in clean if closes[t].dropna().empty]
    if missing:
        raise FetchError(
            f"yfinance returned no data for: {missing}. "
            "Check the ticker symbols and date range."
        )
    log.info(
        "fetched OHLC: %d tickers, %s to %s",
        len(clean), closes.index.min().date(), closes.index.max().date(),
    )
    return {"open": opens, "close": closes}


def _download(tickers, *, start, end, auto_adjust, max_retries, retry_pause):
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last_err = e
            log.warning("yfinance attempt %d/%d raised: %s", attempt, max_retries, e)
        else:
            if raw is None or raw.empty:
                last_err = FetchError("yfinance returned an empty DataFrame")
                log.warning("yfinance attempt %d/%d returned empty", attempt, max_retries)
            else:
                return raw
        if attempt < max_retries:
            time.sleep(retry_pause * attempt)
    raise FetchError(f"yfinance download failed after {max_retries} attempts: {last_err}")


def _field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """Pull one OHLC field out as a DataFrame with one column per ticker."""
    if isinstance(raw.columns, pd.MultiIndex):
        if field not in raw.columns.get_level_values(0):
            raise FetchError(
                f"yfinance response missing '{field}'; "
                f"got {sorted(set(raw.columns.get_level_values(0)))}"
            )
        out = raw[field].copy()
    else:  # single ticker -> flat columns
        if field not in raw.columns:
            raise FetchError(f"yfinance response missing '{field}'; got {list(raw.columns)}")
        out = raw[[field]].copy()
        out.columns = tickers
    out = out.reindex(columns=tickers)
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out
