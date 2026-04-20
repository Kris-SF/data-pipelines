"""
Thin client for the Moontower API (https://api.moontower.ai).

Usage:
    from moontower import call, to_df
    df = to_df(call("price", ticker=["SPY", "QQQ"]))
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; env vars also work

BASE_URL = "https://api.moontower.ai/v1"

_API_KEY = os.environ.get("MOONTOWER_API_KEY")
if not _API_KEY:
    raise RuntimeError(
        "MOONTOWER_API_KEY not set. Add it to a .env file next to this module, "
        "or export it before running."
    )

_SESSION = requests.Session()
_SESSION.headers.update({"X-API-Key": _API_KEY})


def call(endpoint: str, *, raw: bool = False, **params: Any):
    """
    GET https://api.moontower.ai/v1/{endpoint} with cleaned params.

    - Lists (e.g. ticker=["SPY", "QQQ"]) become repeated query params.
    - None values are dropped.
    - Raises requests.HTTPError with the API's error body on non-2xx.
    - Returns parsed JSON by default, or the raw Response if raw=True.
    """
    clean = {k: v for k, v in params.items() if v is not None}
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    resp = _SESSION.get(url, params=clean, timeout=35)

    if not resp.ok:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} on {resp.url}\n{resp.text[:500]}"
        )

    return resp if raw else resp.json()


def to_df(payload) -> pd.DataFrame:
    """Unwrap the common {'data': [...]} envelope into a flat DataFrame."""
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if not payload:
        return pd.DataFrame()
    return pd.json_normalize(payload)
