"""Turn the BigQuery export (see README query) into data.json for the page.

    python prep_data.py <bq_result.json>

Input: the BigQuery jobs.query JSON (rows of [ticker, "c1,c2,..."], plus a "_DATES" row).
Fix-ups applied:
  * XLB/XLY/XLE/XLK closes for 2024-11-01..2024-11-29 are not split-adjusted in source 1 -> /2.
  * UUP closes are not dividend-adjusted; its annual December distribution shows up as a
    ~-3.7% one-day drop. Daily UUP moves above 3% are zeroed.
  * Missing closes are forward-filled (0 return that day).
Output: {"d": "yymmdd,...", "r": {ticker: "int,..."}} with daily log returns x 1e5.
"""
import json, sys
import numpy as np

j = json.load(open(sys.argv[1]))
rows = {r["f"][0]["v"]: r["f"][1]["v"].split(",") for r in j["rows"]}
dates = rows.pop("_DATES")
out = {}
for t, v in rows.items():
    a = np.array([float(x) if x else np.nan for x in v])
    if t in ("XLB", "XLY", "XLE", "XLK"):
        for i, d in enumerate(dates):
            if "241101" <= d <= "241129":
                a[i] /= 2
    for i in range(1, len(a)):
        if np.isnan(a[i]):
            a[i] = a[i - 1]
    r = np.diff(np.log(a))
    r[np.isnan(r)] = 0
    if t == "UUP":
        bad = np.abs(r) > 0.03
        print("UUP zeroed:", [dates[i + 1] for i in np.where(bad)[0]])
        r[bad] = 0
    out[t] = ",".join(str(int(round(x * 1e5))) for x in r)
json.dump({"d": ",".join(dates[1:]), "r": out}, open("data.json", "w"), separators=(",", ":"))
print(len(dates) - 1, "days,", len(out), "tickers")
