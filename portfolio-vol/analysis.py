"""
One-call portfolio realized vol analysis with moontower-styled output.

Usage (Jupyter / VS Code notebook):

    from analysis import run_portfolio_analysis

    result = run_portfolio_analysis(
        weights={"SPY": 0.5, "QQQ": 0.3, "TLT": 0.2},
        start="2024-01-01",
        end="2025-01-01",
    )

`weights` are the T0 portfolio weights (no rebalancing assumption). All
downstream math uses population covariance (ddof=0) on log returns of
the fetched adjusted closes, to match the Excel VBA reference.

Returns an `AnalysisResult` dataclass holding every number (portfolio
vol, component vols, cov, corr, returns DataFrame) so you can keep
poking at the result after the plots render.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.colors import LinearSegmentedColormap

from portfolio_vol import Portfolio, fetch_returns

TRADING_DAYS_PER_YEAR = 252

# Moontower palette (from the styling guide)
INDIGO_600 = "#4F46E5"
INDIGO_700 = "#4338CA"
BLUE_500 = "#3B82F6"
GREEN_500 = "#22C55E"
RED_500 = "#EF4444"
GRAY_900 = "#111827"
GRAY_700 = "#374151"
GRAY_600 = "#4B5563"
GRAY_500 = "#6B7280"
GRAY_300 = "#D1D5DB"
GRAY_200 = "#E5E7EB"
GRAY_100 = "#F3F4F6"
GRAY_50 = "#F9FAFB"

_CORR_CMAP = LinearSegmentedColormap.from_list(
    "moontower_corr",
    [(0.0, RED_500), (0.5, "#FFFFFF"), (1.0, BLUE_500)],
)


@dataclass
class AnalysisResult:
    portfolio: Portfolio
    returns: pd.DataFrame
    cov: pd.DataFrame
    corr: pd.DataFrame
    daily_vol: float
    ann_vol: float
    component_daily_vol: pd.Series
    component_ann_vol: pd.Series
    wavg_daily_vol: float
    wavg_ann_vol: float
    diversification_ratio: float


def run_portfolio_analysis(
    weights: Mapping[str, float],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    normalize: bool = False,
    show: bool = True,
) -> AnalysisResult:
    """Fetch, compute, and render a full realized vol analysis."""
    portfolio = Portfolio.from_weights(weights, normalize=normalize)
    returns = fetch_returns(portfolio, start=start, end=end)

    cov = returns.cov(ddof=0)
    w = portfolio.weights.loc[cov.columns]

    daily_var = float(w @ cov @ w)
    daily_vol = float(np.sqrt(max(daily_var, 0.0)))
    ann_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    component_daily = pd.Series(
        np.sqrt(np.maximum(np.diag(cov), 0.0)),
        index=cov.columns,
        name="daily_vol",
    )
    component_ann = component_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    wavg_daily = float((w * component_daily).sum())
    wavg_ann = wavg_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    diversification = wavg_daily / daily_vol if daily_vol > 0 else float("nan")

    result = AnalysisResult(
        portfolio=portfolio,
        returns=returns,
        cov=cov,
        corr=returns.corr(),
        daily_vol=daily_vol,
        ann_vol=ann_vol,
        component_daily_vol=component_daily,
        component_ann_vol=component_ann,
        wavg_daily_vol=wavg_daily,
        wavg_ann_vol=wavg_ann,
        diversification_ratio=diversification,
    )

    if show:
        _apply_mpl_style()
        _render_summary_card(result)
        _render_components_table(result)
        _render_equity_curve(result)
        _render_component_vol_bar(result)
        _render_correlation_heatmap(result)

    return result


def _apply_mpl_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRAY_300,
            "axes.linewidth": 0.8,
            "axes.labelcolor": GRAY_700,
            "axes.labelsize": 10,
            "axes.titlecolor": GRAY_900,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlepad": 14,
            "axes.grid": True,
            "grid.color": GRAY_100,
            "grid.linewidth": 1,
            "xtick.color": GRAY_500,
            "ytick.color": GRAY_500,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "sans-serif",
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


# ---------- HTML summary card + components table ----------

def _render_summary_card(r: AnalysisResult) -> None:
    sample_range = (
        f"{r.returns.index.min().date()} &rarr; {r.returns.index.max().date()}"
    )
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                padding: 12px 0;">
      <div style="font-size: 11px; text-transform: uppercase;
                  letter-spacing: 0.08em; color: {INDIGO_600}; font-weight: 600;">
        Portfolio Realized Vol
      </div>
      <div style="font-size: 22px; font-weight: 700; color: {GRAY_900};
                  margin: 4px 0 18px 0;">
        {_portfolio_label(r.portfolio)}
      </div>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
        {_tile("Portfolio Vol (ann.)", f"{r.ann_vol*100:.2f}%")}
        {_tile("Wtd-Avg Component Vol (ann.)", f"{r.wavg_ann_vol*100:.2f}%")}
        {_tile("Diversification Ratio", f"{r.diversification_ratio:.3f}")}
      </div>
      <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;
                  margin-top: 12px;">
        {_tile("Portfolio Vol (daily)", f"{r.daily_vol*100:.4f}%")}
        {_tile("Sample", f"{len(r.returns)} days", sublabel=sample_range)}
      </div>
    </div>
    """
    display(HTML(html))


def _portfolio_label(p: Portfolio) -> str:
    parts = [f"{t} {w*100:.1f}%" for t, w in p.weights.items()]
    return " &middot; ".join(parts)


def _tile(label: str, value: str, sublabel: str | None = None) -> str:
    sub = (
        f'<div style="font-size: 11px; color: {GRAY_500}; '
        f'font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin-top: 4px;">'
        f"{sublabel}</div>"
        if sublabel
        else ""
    )
    return f"""
    <div style="background: white; border: 1px solid {GRAY_200}; border-radius: 8px;
                padding: 14px 16px; text-align: center;">
      <div style="font-size: 20px; font-weight: 700; color: {INDIGO_600};
                  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;">
        {value}
      </div>
      <div style="font-size: 10.5px; color: {GRAY_500}; text-transform: uppercase;
                  letter-spacing: 0.06em; margin-top: 6px;">
        {label}
      </div>
      {sub}
    </div>
    """


def _render_components_table(r: AnalysisResult) -> None:
    df = pd.DataFrame(
        {
            "weight": r.portfolio.weights,
            "daily_vol": r.component_daily_vol,
            "ann_vol": r.component_ann_vol,
        }
    )
    styled = (
        df.style.format(
            {"weight": "{:.4f}", "daily_vol": "{:.4%}", "ann_vol": "{:.2%}"}
        )
        .set_caption("Component Vols")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": (
                        f"caption-side: top; text-align: left; font-weight: 700; "
                        f"text-transform: uppercase; letter-spacing: 0.06em; "
                        f"color: {GRAY_900}; padding: 12px 0 8px 0; font-size: 13px;"
                    ),
                },
                {
                    "selector": "th.col_heading",
                    "props": (
                        f"background: {GRAY_100}; color: {GRAY_700}; "
                        f"border-bottom: 1px solid {GRAY_300}; padding: 8px 12px; "
                        "text-align: right; text-transform: uppercase; "
                        "letter-spacing: 0.04em; font-size: 11px;"
                    ),
                },
                {
                    "selector": "th.row_heading",
                    "props": (
                        f"background: {GRAY_50}; color: {GRAY_900}; "
                        f"border-right: 1px solid {GRAY_200}; padding: 8px 12px; "
                        "text-align: left; font-weight: 600;"
                    ),
                },
                {
                    "selector": "td",
                    "props": (
                        f"border-bottom: 1px solid {GRAY_100}; padding: 8px 12px; "
                        "font-family: ui-monospace, SFMono-Regular, Menlo, monospace; "
                        "text-align: right;"
                    ),
                },
                {
                    "selector": "",
                    "props": "border-collapse: collapse;",
                },
            ]
        )
    )
    display(styled)


# ---------- matplotlib figures ----------

def _render_equity_curve(r: AnalysisResult) -> None:
    port_ret = r.returns @ r.portfolio.weights
    cum = (1.0 + port_ret).cumprod() * 100.0

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(cum.index, cum.values, color=INDIGO_600, linewidth=2)
    ax.axhline(100, color=GRAY_300, linewidth=1, linestyle="--", zorder=0)
    ax.set_title("Portfolio Equity Curve  ·  growth of $100")
    ax.set_ylabel("Value ($)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0)
    plt.tight_layout()
    plt.show()


def _render_component_vol_bar(r: AnalysisResult) -> None:
    s = (r.component_ann_vol * 100.0).sort_values(ascending=True)
    height = max(2.5, 0.42 * len(s) + 1.0)

    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(s.index, s.values, color=INDIGO_600, edgecolor="white", height=0.7)
    max_v = float(s.max()) if len(s) else 0.0
    for i, v in enumerate(s.values):
        ax.text(
            v + max_v * 0.01,
            i,
            f"{v:.1f}%",
            va="center",
            color=GRAY_700,
            fontsize=9,
            family="monospace",
        )
    ax.set_title("Annualized Component Vol")
    ax.set_xlabel("Annualized vol (%)")
    ax.set_xlim(0, max_v * 1.12 if max_v > 0 else 1)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()


def _render_correlation_heatmap(r: AnalysisResult) -> None:
    corr = r.corr
    n = len(corr)
    size = max(5.0, 0.6 * n + 2.5)

    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(corr.values, cmap=_CORR_CMAP, vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for i in range(n):
        for j in range(n):
            v = float(corr.values[i, j])
            text_color = "white" if abs(v) > 0.55 else GRAY_900
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                family="monospace",
            )

    ax.set_title("Correlation Matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=GRAY_500)
    plt.tight_layout()
    plt.show()
