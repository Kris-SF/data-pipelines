import type { PriceSeries } from "./yahoo"

export type AlignedPanel = {
  dates: string[]
  tickers: string[]
  prices: number[][] // prices[t][i] = price of ticker i on date t
}

export function alignPanels(
  panels: Record<string, PriceSeries>,
  tickers: string[],
): AlignedPanel {
  const dateSets = tickers.map((t) => new Set(panels[t].dates))
  const common = panels[tickers[0]].dates.filter((d) =>
    dateSets.every((s) => s.has(d)),
  )
  const priceByTicker = new Map<string, Map<string, number>>()
  for (const t of tickers) {
    const m = new Map<string, number>()
    panels[t].dates.forEach((d, i) => m.set(d, panels[t].prices[i]))
    priceByTicker.set(t, m)
  }
  const prices = common.map((d) =>
    tickers.map((t) => priceByTicker.get(t)!.get(d)!),
  )
  return { dates: common, tickers, prices }
}

export function logReturns(panel: AlignedPanel): {
  dates: string[]
  returns: number[][]
} {
  const { dates, prices } = panel
  const out: number[][] = []
  for (let t = 1; t < prices.length; t++) {
    const row: number[] = new Array(prices[t].length)
    for (let i = 0; i < prices[t].length; i++) {
      row[i] = Math.log(prices[t][i] / prices[t - 1][i])
    }
    out.push(row)
  }
  return { dates: dates.slice(1), returns: out }
}

/**
 * Population covariance (ddof=0) to match Excel's WorksheetFunction.Covar.
 */
export function populationCovariance(returns: number[][], n: number): number[][] {
  const T = returns.length
  const means = new Array(n).fill(0)
  for (const row of returns) for (let i = 0; i < n; i++) means[i] += row[i]
  for (let i = 0; i < n; i++) means[i] /= T

  const cov: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
  for (const row of returns) {
    for (let i = 0; i < n; i++) {
      const di = row[i] - means[i]
      for (let j = 0; j < n; j++) {
        cov[i][j] += di * (row[j] - means[j])
      }
    }
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cov[i][j] /= T
  return cov
}

export function portfolioVol(cov: number[][], weights: number[]): number {
  const n = weights.length
  let variance = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      variance += weights[i] * cov[i][j] * weights[j]
    }
  }
  return Math.sqrt(Math.max(variance, 0))
}

export function correlationFromCov(cov: number[][]): number[][] {
  const n = cov.length
  const sd = cov.map((row, i) => Math.sqrt(row[i]))
  const out: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const denom = sd[i] * sd[j]
      out[i][j] = denom > 0 ? cov[i][j] / denom : 0
    }
  }
  return out
}

export function componentVol(cov: number[][]): number[] {
  return cov.map((row, i) => Math.sqrt(Math.max(row[i], 0)))
}
