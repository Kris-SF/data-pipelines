# Factor Lab

A single-page, self-contained explorer for learning how equity factors work. It's an interactive
version of the beta-adjusted residual method in `../factor_residual_correlations.ipynb`.

Lessons on the page:
1. **Take the market out**: an ETF-vs-SPY scatter with the fitted beta, and cumulative return split
   into the market part (β × SPY), the factor part (residual), and the naive `ETF − SPY` shortcut.
2. **Scoreboard**: residual returns over 1W/1M/3M/1Y, plus the latest day's residual as a z-score.
3. **Correlation map**: residual vs raw correlation heatmap with an as-of slider, and a
   "21d − 252d" change mode.
4. **Short vs long lookbacks**: rolling 21/63/252-day correlation for any pair, plus a list of the
   pairs whose short-window correlation has broken furthest from the long-window value.
5. **Regime**: average pairwise correlation within sectors, style factors and international.

The beta window (21/63/126/252) in the sticky bar drives every residual on the page.

## Data

Moontower EOD closes from BigQuery `moontower-rwjs.lake.eod_price` (`data_source = 1`), from
2023-11-01 onward, for 36 ETFs. Known fix-up: in source 1, XLB/XLY/XLE/XLK closes for 2024-11-01
through 2024-11-29 are not split-adjusted (2× the rest of the series). `data.json` divides
those closes by 2. Missing days are forward-filled, which gives a 0 return on those days.

Query used (one row per ticker, closes as a comma string on the SPY calendar):

```sql
WITH px AS (
  SELECT t.ticker, p.trade_date d, ANY_VALUE(p.close) c
  FROM `moontower-rwjs.lake.eod_price` p JOIN `moontower-rwjs.lake.ticker` t USING(ticker_id)
  WHERE p.trade_date BETWEEN '2022-10-01' AND CURRENT_DATE() AND p.data_source = 1
    AND t.ticker IN ('SPY','MTUM','VTV','VUG','IWM','IJR','VYM','SCHD','RSP','QQQ','ARKK',
                     'XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU','XLB','XLRE','XLC','SMH','KRE',
                     'EFA','EEM','EWJ','EWZ','FXI','TLT','IEF','HYG','GLD','SLV','USO','VNQ')
  GROUP BY 1, 2),
cal AS (SELECT DISTINCT d FROM px WHERE ticker = 'SPY'),
grid AS (SELECT tk.ticker, cal.d FROM cal CROSS JOIN (SELECT DISTINCT ticker FROM px) tk)
SELECT '_DATES' ticker, STRING_AGG(FORMAT_DATE('%y%m%d', d), ',' ORDER BY d) s FROM cal
UNION ALL
SELECT g.ticker, STRING_AGG(IFNULL(FORMAT('%.2f', px.c), ''), ',' ORDER BY g.d)
FROM grid g LEFT JOIN px USING (ticker, d) GROUP BY 1
```

`data.json` = `{"d": "yymmdd,...", "r": {ticker: "int,..."}}` where each int is a daily log
return × 1e5.

## Method

Log returns. For each ETF, a rolling OLS on SPY over the beta window, ending on day t (same as the
notebook). The residual is `r − (α + β·r_SPY)`. Correlations are Pearson over the chosen lookback.
All math runs client-side in the page.
