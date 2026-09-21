"""
Offline tests for diversifier.py (no network).

Run from quant-analysis/:  python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diversifier as dv  # noqa: E402

# Toy portfolio from the post: A 12% / 20%, B 5% / 15%, cash 2%, rho 0.20.
TOY = dict(mu_a=0.12, sig_a=0.20, mu_b=0.05, sig_b=0.15, rf=0.02, rho=0.20)


# --- toy portfolio ---------------------------------------------------------

@pytest.mark.parametrize("w, expected", [(0.0, 0.500), (0.20, 0.5101), (0.50, 0.4763)])
def test_toy_mix_sharpe(w, expected):
    assert dv.mix_sharpe(w, **TOY) == pytest.approx(expected, abs=5e-5)


def test_analytic_derivative_matches_finite_difference():
    rng = np.random.default_rng(0)
    for _ in range(200):
        mu_a, mu_b = rng.uniform(-0.05, 0.20, 2)
        sig_a, sig_b = rng.uniform(0.05, 0.40, 2)
        rf = rng.uniform(0.0, 0.05)
        rho = rng.uniform(-0.9, 0.9)
        s_a = (mu_a - rf) / sig_a
        s_b = (mu_b - rf) / sig_b
        analytic = dv.sharpe_derivative_at_zero(s_a, s_b, sig_a, sig_b, rho)
        h = 1e-6
        fd = (dv.mix_sharpe(h, mu_a, sig_a, mu_b, sig_b, rf, rho)
              - dv.mix_sharpe(-h, mu_a, sig_a, mu_b, sig_b, rf, rho)) / (2 * h)
        assert analytic == pytest.approx(fd, rel=1e-5, abs=1e-7)


def test_finite_weight_hurdle_equivalent_to_sharpe_improvement():
    rng = np.random.default_rng(1)
    n_checked = 0
    for _ in range(2000):
        mu_a, mu_b = rng.uniform(-0.05, 0.20, 2)
        sig_a, sig_b = rng.uniform(0.05, 0.40, 2)
        rf = rng.uniform(0.0, 0.05)
        rho = rng.uniform(-0.9, 0.9)
        w = rng.uniform(0.01, 0.95)
        s_a = (mu_a - rf) / sig_a
        s_b = (mu_b - rf) / sig_b
        k = sig_b / sig_a
        hurdle = dv.finite_weight_hurdle(s_a, w, k, rho)
        s_p = dv.mix_sharpe(w, mu_a, sig_a, mu_b, sig_b, rf, rho)
        if abs(s_b - hurdle) < 1e-9 or abs(s_p - s_a) < 1e-9:
            continue  # on the boundary; skip
        assert (s_b > hurdle) == (s_p > s_a)
        n_checked += 1
    assert n_checked > 1500


def test_finite_weight_hurdle_limits_to_small_addition_hurdle():
    s_a, rho, k = 0.7, 0.3, 1.2
    assert dv.finite_weight_hurdle(s_a, 1e-7, k, rho) == pytest.approx(dv.small_addition_hurdle(s_a, rho), abs=1e-5)


# --- coin flips -----------------------------------------------------------

def test_expected_heads_times_tails_100_fair_flips():
    assert dv.expected_heads_times_tails(100) == 2475.0


# --- Sharpe / conditional stats -------------------------------------------

def test_sharpe_uses_ddof_1():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.02])
    assert dv.sharpe(x) == pytest.approx(x.mean() / x.std(ddof=1))


def test_conditional_sharpe_rows_matches_scalar_sharpe():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 40))
    mask = rng.uniform(size=(5, 40)) > 0.5
    rows = dv.conditional_sharpe_rows(x, mask)
    for i in range(5):
        assert rows[i] == pytest.approx(dv.sharpe(x[i][mask[i]]))


def test_portfolio_returns_are_weighted_sums():
    idx = pd.period_range("2020-01", "2020-04", freq="M")
    r = pd.DataFrame({"SPY": [0.01, 0.02, -0.01, 0.03], "AGG": [0.00, 0.01, 0.01, -0.01]}, index=idx)
    p = dv.portfolio_returns(r, {"SPY": 0.6, "AGG": 0.4})
    np.testing.assert_allclose(p.to_numpy(), 0.6 * r["SPY"].to_numpy() + 0.4 * r["AGG"].to_numpy())
    with pytest.raises(ValueError):
        dv.portfolio_returns(r, {"SPY": 0.6, "AGG": 0.5})


def test_compound_calendar_years_keeps_only_full_years():
    idx = pd.period_range("2019-11", "2021-12", freq="M")
    s = pd.Series(0.01, index=idx)
    annual = dv.compound_calendar_years(s)
    assert list(annual.index) == [2020, 2021]
    assert annual.loc[2020] == pytest.approx(1.01**12 - 1)


# --- bootstrap indices ----------------------------------------------------

def test_bootstrap_indices_shape_and_range():
    rng = np.random.default_rng(20260921 + 6)
    idx = dv.block_bootstrap_indices(240, 6, 50, rng)
    assert idx.shape == (50, 240)
    assert idx.min() >= 0 and idx.max() <= 239
    assert idx.dtype.kind == "i"


def test_bootstrap_indices_wrap_at_end_of_sample():
    n, L = 240, 6

    class FixedStarts:
        """Stand-in rng that returns chosen start indices."""
        def integers(self, low, high, size):
            starts = np.array([[238, 5] + [0] * (size[1] - 2)])
            return np.tile(starts, (size[0], 1))

    idx = dv.block_bootstrap_indices(n, L, 1, FixedStarts())  # type: ignore[arg-type]
    assert idx[0, :6].tolist() == [238, 239, 0, 1, 2, 3]
    assert idx[0, 6:12].tolist() == [5, 6, 7, 8, 9, 10]
    assert idx.shape == (1, n)


def test_bootstrap_indices_blocks_are_consecutive_and_truncated():
    n, L = 240, 7  # ceil(240/7) = 35 blocks -> 245 rows, truncated to 240
    rng = np.random.default_rng(0)
    idx = dv.block_bootstrap_indices(n, L, 20, rng)
    assert idx.shape == (20, n)
    for row in idx:
        for b in range(n // L):
            block = row[b * L:(b + 1) * L]
            assert np.all((block - block[0]) % n == np.arange(L))


def test_bootstrap_indices_reproducible_under_fixed_seed():
    a = dv.block_bootstrap_indices(240, 12, 100, np.random.default_rng(20260921 + 12))
    b = dv.block_bootstrap_indices(240, 12, 100, np.random.default_rng(20260921 + 12))
    c = dv.block_bootstrap_indices(240, 12, 100, np.random.default_rng(20260921 + 3))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# --- parsers and month-end sampling (synthetic text, no network) ----------

FRENCH_SAMPLE = """This file was created using the 202607 CRSP database.
The 1-month TBill rate ...

,Mkt-RF,SMB,HML,RF
200606,   -0.50,   1.00,   2.00,   0.40
200607,    0.10,  -1.00,   0.50,   0.40
200608,    2.00,   0.30,  -0.20,   0.42

 Annual Factors: January-December 
,Mkt-RF,SMB,HML,RF
  2006,   10.00,   0.00,   0.00,   4.80
"""


def test_parse_french_monthly_block_only_and_divides_by_100():
    df = dv.parse_french_monthly(FRENCH_SAMPLE)
    assert list(df.index.astype(str)) == ["2006-06", "2006-07", "2006-08"]
    assert df.loc[pd.Period("2006-08", "M"), "RF"] == pytest.approx(0.0042)
    assert 2006 not in df.index.year or len(df) == 3


FRED_SAMPLE = """observation_date,T10YIE
2006-07-27,2.55
2006-07-28,2.57
2006-07-31,
2006-08-01,2.50
2006-08-31,.
"""


def test_parse_fred_and_month_end_takes_last_non_missing():
    s = dv.parse_fred_csv(FRED_SAMPLE, "T10YIE")
    assert s.isna().sum() == 2
    vals, dates = dv.month_end_observations(s)
    assert vals.loc[pd.Period("2006-07", "M"), "T10YIE"] == 2.57
    assert pd.Timestamp(dates.loc[pd.Period("2006-07", "M"), "T10YIE"]) == pd.Timestamp("2006-07-28")
    assert vals.loc[pd.Period("2006-08", "M"), "T10YIE"] == 2.50


def test_month_end_observations_last_trading_day_per_ticker():
    idx = pd.to_datetime(["2006-07-28", "2006-07-31", "2006-08-30", "2006-08-31"])
    px = pd.DataFrame({"SPY": [1.0, 2.0, 3.0, 4.0], "GLD": [1.0, 2.0, 3.0, np.nan]}, index=idx)
    vals, dates = dv.month_end_observations(px)
    assert vals.loc[pd.Period("2006-08", "M"), "SPY"] == 4.0
    assert vals.loc[pd.Period("2006-08", "M"), "GLD"] == 3.0
    assert pd.Timestamp(dates.loc[pd.Period("2006-08", "M"), "GLD"]) == pd.Timestamp("2006-08-30")


def test_require_months_fails_loudly_on_gap():
    idx = pd.PeriodIndex(["2006-07", "2006-08", "2006-10"], freq="M")
    s = pd.Series(1.0, index=idx)
    with pytest.raises(dv.DataError, match="2006-09"):
        dv._require_months(s, pd.Period("2006-07", "M"), pd.Period("2006-10", "M"), "x")


# --- end-to-end on a synthetic panel ------------------------------------

def _synthetic_panel(n=240, seed=7) -> dv.Panel:
    rng = np.random.default_rng(seed)
    months = pd.period_range("2006-08", periods=n, freq="M")
    rets = pd.DataFrame(rng.normal(0.005, 0.04, size=(n, len(dv.ETF_TICKERS))), index=months, columns=list(dv.ETF_TICKERS))
    rets[["DBC", "USO"]] -= 0.02  # commodities lose money so they fail the hurdle, as in the post
    rf = pd.Series(rng.uniform(0, 0.004, n), index=months, name="RF")
    chg = pd.DataFrame(rng.normal(0, 10, size=(n, 2)).round(0), index=months, columns=list(dv.BREAKEVEN_SERIES))
    levels = chg.cumsum() / 100 + 2.0
    return dv.Panel(returns=rets, rf=rf, breakeven_levels=levels, breakeven_changes_bp=chg,
                    price_dates=pd.DataFrame(), meta={"start": str(months[0]), "end": str(months[-1])})


def test_compute_all_on_synthetic_panel_runs_and_is_self_consistent():
    panel = _synthetic_panel()
    cfg = dv.StudyConfig(mode="reproduce", n_boot=200)
    res = dv.compute_all(panel, cfg, verbose=False)
    t = res.tables
    # rising + falling + unchanged = n
    assert t["table3_rising_10y"]["n"] + t["table4_falling_10y"]["n"] + t["table4_falling_10y"]["n_unchanged"] == panel.n
    # stability full cut equals table 3
    assert t["table6_stability"]["10y_full"]["60/40"] == pytest.approx(t["table3_rising_10y"]["sharpe"]["60/40"])
    assert t["table6_stability"]["10y_first_half"]["n_rising"] + t["table6_stability"]["10y_second_half"]["n_rising"] == t["table6_stability"]["10y_full"]["n_rising"]
    # point estimate is the difference of table 3 conditional Sharpes
    s = t["table3_rising_10y"]["sharpe"]
    assert t["table7_bootstrap"]["point_estimates"]["DBC"] == pytest.approx(s["60/30/10 DBC"] - s["60/40"])
    # SPY correlation with itself is 1 and its hurdle equals its own Sharpe
    assert t["table1_etf"]["SPY"]["corr_with_spy"] == pytest.approx(1.0)
    assert t["table1_etf"]["SPY"]["hurdle"] == pytest.approx(t["table1_etf"]["SPY"]["sharpe_ann"])
    # every published key exists in the intervals
    assert set(t["table7_bootstrap"]["intervals"]) == {"10y_L3", "10y_L6", "10y_L12", "5y_L6"}
    # comparison runs against a REPORTED dict built from the computed values -> all pass
    reported = {
        "etf_table": {k: (v["sharpe_ann"], v["corr_with_spy"], v["sharpe_80_20_ann"]) for k, v in t["table1_etf"].items()},
        "dbc_hurdle": t["table1_etf"]["DBC"]["hurdle"],
        "weak_bond_years": t["table2_weak_bond_years"]["years"],
        "weak_bond_wins": t["table2_weak_bond_years"]["wins_vs_60_40"],
        "rising_10y": {"n": t["table3_rising_10y"]["n"], **{k: tuple(t["table3_rising_10y"][k][p] for p in dv.PORTFOLIO_NAMES) for k in ("mean_excess_pct", "vol_pct", "sharpe")}},
        "falling_10y": {"n": t["table4_falling_10y"]["n"], "n_unchanged": t["table4_falling_10y"]["n_unchanged"], **{k: tuple(t["table4_falling_10y"][k][p] for p in dv.PORTFOLIO_NAMES) for k in ("mean_excess_pct", "vol_pct", "sharpe")}},
        "full_sample_sharpe": tuple(t["table5_full_sample_sharpe"][p] for p in dv.PORTFOLIO_NAMES),
        "stability": {k: (v["n_rising"], *(v[p] for p in dv.PORTFOLIO_NAMES)) for k, v in t["table6_stability"].items()},
        "bootstrap_point": t["table7_bootstrap"]["point_estimates"],
        "bootstrap_ci": {k: {e: (v[e]["ci_low"], v[e]["ci_high"]) for e in ("DBC", "USO")} for k, v in t["table7_bootstrap"]["intervals"].items()},
    }
    assert not t["table1_etf"]["DBC"]["passes_hurdle"] and not t["table1_etf"]["USO"]["passes_hurdle"]
    diff = dv.compare_to_reported(res, reported)
    assert (diff["status"] == "pass").all(), diff[diff["status"] != "pass"]
