# Overnight Variance — where does the risk actually happen?

How much of an asset's daily variance is born **overnight** (while the market
is closed) versus **intraday**, and whether **weekends** carry more risk than a
regular weeknight.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kris-SF/data-pipelines/blob/main/overnight-variance/overnight_variance.ipynb)

## Method

Two log returns per day, anchored on the prior close:

- `r_co = ln(open_t / close_{t-1})` — the overnight gap
- `r_cc = ln(close_t / close_{t-1})` — the full close-to-close day

Zero-drift variance (mean of squared returns — no mean subtraction, no
annualization; it all cancels in a ratio). Two outputs:

1. **Rolling overnight variance ratio** `Σ r_co² / Σ r_cc²` over a trailing
   window. `> 1` means the overnight move exceeds the full day — the session
   *fades* the gap.
2. **Weekend premium** — variance of clean `Fri→Mon` gaps (exactly Sat+Sun
   closed, no adjacent holiday) vs. consecutive-business-day overnights, in
   half-year buckets.

## Data

`yfinance` daily OHLC with `auto_adjust=True`, so split/dividend adjustments
keep an ex-dividend price drop from masquerading as an overnight gap (matters
for dividend payers — TLT pays monthly).

## Files

| File | What |
|---|---|
| `overnight_variance.ipynb` | Colab-ready notebook (edit the ticker list + dates and run) |
| `data.py` | yfinance OHLC fetcher (keeps Open *and* Close) |
| `variance.py` | returns, rolling ratio, weekend-premium table, plots |

## Quickstart (local)

```bash
pip install -r requirements.txt
```

```python
from variance import run_overnight_analysis, plot_weekend_premium
res = run_overnight_analysis(["SPY", "USO", "QQQ", "GLD", "TLT", "FXY", "FXE"],
                             start="2023-06-01", end="2026-06-29")
print(res.pooled)          # weekend / regular multiple per ticker
plot_weekend_premium(res)  # the cross-section bar chart
```
