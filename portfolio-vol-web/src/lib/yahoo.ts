type YahooChartResponse = {
  chart: {
    result?: Array<{
      timestamp?: number[]
      indicators: {
        adjclose?: Array<{ adjclose?: Array<number | null> }>
        quote: Array<{ close?: Array<number | null> }>
      }
    }>
    error?: { code: string; description: string } | null
  }
}

export type PriceSeries = { dates: string[]; prices: number[] }

/**
 * Fetch daily auto-adjusted closes for a single ticker from Yahoo's chart API.
 * start/end are ISO dates (YYYY-MM-DD). end is inclusive.
 */
export async function fetchAdjClose(
  ticker: string,
  start: string,
  end: string,
): Promise<PriceSeries> {
  const period1 = Math.floor(new Date(`${start}T00:00:00Z`).getTime() / 1000)
  const period2 = Math.floor(new Date(`${end}T23:59:59Z`).getTime() / 1000)
  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}` +
    `?period1=${period1}&period2=${period2}&interval=1d&events=div%2Csplit`

  const res = await fetch(url, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      Accept: "application/json",
    },
    cache: "no-store",
  })

  if (!res.ok) {
    throw new Error(`Yahoo returned HTTP ${res.status} for ${ticker}`)
  }

  const json = (await res.json()) as YahooChartResponse

  if (json.chart.error) {
    throw new Error(
      `Yahoo rejected ${ticker}: ${json.chart.error.description}`,
    )
  }
  const result = json.chart.result?.[0]
  if (!result) {
    throw new Error(`Yahoo returned no data for ${ticker}`)
  }

  const ts = result.timestamp ?? []
  const adj = result.indicators.adjclose?.[0]?.adjclose
  const close = result.indicators.quote[0]?.close
  const series = adj ?? close ?? []

  const dates: string[] = []
  const prices: number[] = []
  for (let i = 0; i < ts.length; i++) {
    const p = series[i]
    if (p != null && Number.isFinite(p)) {
      dates.push(new Date(ts[i] * 1000).toISOString().slice(0, 10))
      prices.push(p)
    }
  }
  if (!prices.length) {
    throw new Error(`Yahoo returned an empty series for ${ticker}`)
  }
  return { dates, prices }
}
