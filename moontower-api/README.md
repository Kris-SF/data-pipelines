# Moontower API Explorer

Scratchpad + thin Python client for [api.moontower.ai](https://api.moontower.ai/docs).

## Setup

```bash
cd moontower-api
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your MOONTOWER_API_KEY (never commit this file)

jupyter notebook moontower_api_explorer.ipynb
```

## The client

```python
from moontower import call, to_df

df = to_df(call("price", ticker=["SPY", "QQQ"]))

df = to_df(call("cmiv", ticker="SPY",
                start_date="2026-03-01",
                end_date="2026-03-31"))

resp = call("price", ticker="SPY", format="csv", raw=True)
```

`call()` handles repeated list params, dropping `None`, surfacing API error bodies. `to_df()` unwraps the `{"data": [...]}` envelope.

## Endpoints

| Endpoint | Notes |
|---|---|
| `/v1/tickers` | Directory — run first to see what's available |
| `/v1/price` | OHLCV + mid, 15-min delayed intraday |
| `/v1/impliedvol` | Full IV surface by delta level |
| `/v1/realvol` | **EOD-only**, empty intraday — use `trade_date=<yesterday>` |
| `/v1/cmiv` | Constant-maturity IV at fixed tenors |
| `/v1/ivrank` | IV rank + percentile, 1m/3m/1y lookbacks |
| `/v1/rviv` | RV vs IV, VRP = 100·ln(IV₃₀/RV₃₀) |
| `/v1/skew` | 10Δ/25Δ call/put skew + percentiles |
| `/v1/optionchain` | **EOD-only**, heavy — filter by `expiry_date` |
| `/v1/cockpit` | Bundle of price/IV/returns/RVIV stats |
| `/v1/trade-ideas` | `ideas`/`categories`/`liquidity_levels` are **comma-separated strings**, not repeated |

## Limits

- 100 tickers / request
- 31-day date range max
- 30s query timeout
- 1000 req/min (headers: `X-RateLimit-Limit/Remaining/Reset`)

## Repo is public — security notes

- `.env` must NEVER be committed. Root `.gitignore` already covers it.
- Clear notebook outputs before committing: `jupyter nbconvert --clear-output --inplace moontower_api_explorer.ipynb`
- If a key is ever committed: rotate the key first, then worry about git history.
