"use client"

import { useMemo, useState } from "react"
import Link from "next/link"
import { Header } from "../components/Header"
import { Footer } from "../components/Footer"
import RoundButton from "../components/RoundButton"

type Result = {
  tickers: string[]
  weights: number[]
  dates: { start: string; end: string; count: number }
  portfolio: { daily: number; annualized: number }
  wavgComponent: { daily: number; annualized: number }
  diversificationRatio: number | null
  components: Array<{
    ticker: string
    weight: number
    daily: number
    annualized: number
  }>
  correlation: number[][]
}

type Holding = { ticker: string; weight: number }

function isoDaysAgo(days: number): string {
  return new Date(Date.now() - days * 24 * 3600 * 1000)
    .toISOString()
    .slice(0, 10)
}

function parseHoldings(raw: string): Holding[] {
  return raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(/[,\s]+/).filter(Boolean)
      const ticker = (parts[0] ?? "").toUpperCase()
      const weight = Number(parts[1])
      return { ticker, weight }
    })
}

function pct(v: number): string {
  return (v * 100).toFixed(2) + "%"
}

function pct4(v: number): string {
  return (v * 100).toFixed(4) + "%"
}

export default function Page() {
  const [raw, setRaw] = useState("SPY, 0.5\nQQQ, 0.3\nTLT, 0.2")
  const [start, setStart] = useState(isoDaysAgo(365))
  const [end, setEnd] = useState(isoDaysAgo(1))
  const [normalize, setNormalize] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState<string | null>(null)

  const weightSum = useMemo(() => {
    const h = parseHoldings(raw)
    return h.reduce(
      (a, x) => a + (Number.isFinite(x.weight) ? x.weight : 0),
      0,
    )
  }, [raw])

  async function onCompute() {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const holdings = parseHoldings(raw)
      if (!holdings.length) throw new Error("Enter at least one holding.")

      const res = await fetch("/api/vol", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ holdings, start, end, normalize }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
      setResult(data as Result)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Header />
      <main className="bg-white min-h-screen">
        <div className="mx-auto max-w-7xl px-6 py-16 lg:px-8">
          <div className="mx-auto max-w-4xl">
            <p className="text-base font-semibold leading-7 text-indigo-600">
              <Link href="/tools-and-games">Tools and Games</Link>
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
              Portfolio Realized Vol
            </h1>
            <p className="mt-6 text-lg leading-8 text-gray-600">
              Enter tickers and weights, pick a date range, and get realized
              portfolio vol, weighted-average component vol, and a correlation
              matrix over the full sample. Math uses population covariance
              (divide by n) on Yahoo adjusted closes.
            </p>

            {/* Inputs */}
            <div className="mt-10 bg-gray-50 p-6 rounded-lg border border-gray-200">
              <label
                htmlFor="holdings"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Holdings &mdash; one per line as{" "}
                <span className="font-mono">TICKER, WEIGHT</span>
              </label>
              <textarea
                id="holdings"
                value={raw}
                onChange={(e) => setRaw(e.target.value)}
                rows={8}
                spellCheck={false}
                className="w-full px-3 py-2 border border-gray-300 rounded-md font-mono text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <p className="mt-2 text-xs text-gray-500 uppercase tracking-wider">
                Weights sum to{" "}
                <span
                  className={
                    Math.abs(weightSum - 1) < 1e-6
                      ? "text-green-600 font-mono"
                      : "text-red-600 font-mono"
                  }
                >
                  {weightSum.toFixed(4)}
                </span>
              </p>

              <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label
                    htmlFor="start"
                    className="block text-sm font-medium text-gray-700 mb-2"
                  >
                    Start date
                  </label>
                  <input
                    id="start"
                    type="date"
                    value={start}
                    onChange={(e) => setStart(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
                <div>
                  <label
                    htmlFor="end"
                    className="block text-sm font-medium text-gray-700 mb-2"
                  >
                    End date
                  </label>
                  <input
                    id="end"
                    type="date"
                    value={end}
                    onChange={(e) => setEnd(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
              </div>

              <label className="mt-4 flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={normalize}
                  onChange={(e) => setNormalize(e.target.checked)}
                  className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                />
                Auto-normalize weights to sum to 1
              </label>

              <div className="mt-6">
                <button
                  onClick={onCompute}
                  disabled={loading}
                  className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {loading ? "Computing…" : "Compute"}
                </button>
              </div>
            </div>

            {error && (
              <div className="mt-8 bg-red-50 text-red-700 border border-red-300 p-4 rounded-lg">
                {error}
              </div>
            )}

            {result && (
              <>
                {/* Headline metrics */}
                <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-6">
                  <MetricTile
                    label="Portfolio Vol (ann.)"
                    value={pct(result.portfolio.annualized)}
                  />
                  <MetricTile
                    label="Wtd-Avg Component Vol (ann.)"
                    value={pct(result.wavgComponent.annualized)}
                  />
                  <MetricTile
                    label="Diversification Ratio"
                    value={
                      result.diversificationRatio != null
                        ? result.diversificationRatio.toFixed(3)
                        : "—"
                    }
                  />
                </div>

                <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-6">
                  <MetricTile
                    label="Portfolio Vol (daily)"
                    value={pct4(result.portfolio.daily)}
                  />
                  <MetricTile
                    label="Sample"
                    value={`${result.dates.count} days`}
                    sublabel={`${result.dates.start} → ${result.dates.end}`}
                  />
                </div>

                {/* Components table */}
                <div className="mt-8 bg-white p-6 rounded-lg border border-gray-200">
                  <h3 className="text-lg font-bold text-center uppercase tracking-wide mb-6">
                    Component Vols
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse border border-gray-300 text-sm">
                      <thead>
                        <tr className="bg-gray-100">
                          <th className="border border-gray-300 px-3 py-2 text-left font-semibold">
                            Ticker
                          </th>
                          <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                            Weight
                          </th>
                          <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                            Daily Vol
                          </th>
                          <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                            Annualized Vol
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.components.map((c, i) => (
                          <tr
                            key={c.ticker}
                            className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}
                          >
                            <td className="border border-gray-300 px-3 py-2 font-semibold">
                              {c.ticker}
                            </td>
                            <td className="border border-gray-300 px-3 py-2 text-right font-mono">
                              {c.weight.toFixed(4)}
                            </td>
                            <td className="border border-gray-300 px-3 py-2 text-right font-mono">
                              {pct4(c.daily)}
                            </td>
                            <td className="border border-gray-300 px-3 py-2 text-right font-mono">
                              {pct(c.annualized)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Correlation heatmap */}
                <div className="mt-8 bg-white p-6 rounded-lg border border-gray-200">
                  <h3 className="text-lg font-bold text-center uppercase tracking-wide mb-6">
                    Correlation Matrix
                  </h3>
                  <CorrelationHeatmap
                    tickers={result.tickers}
                    corr={result.correlation}
                  />
                </div>
              </>
            )}

            <div className="mt-8 py-8 border-t border-gray-300 text-center">
              <RoundButton text="View Other Tools" url="/tools-and-games" />
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </>
  )
}

function MetricTile({
  label,
  value,
  sublabel,
}: {
  label: string
  value: string
  sublabel?: string
}) {
  return (
    <div className="bg-white p-4 rounded-lg border border-gray-200 text-center">
      <div className="text-xl font-bold text-indigo-600 font-mono">{value}</div>
      <div className="text-xs text-gray-500 uppercase tracking-wider mt-1">
        {label}
      </div>
      {sublabel && (
        <div className="text-xs text-gray-500 font-mono mt-1">{sublabel}</div>
      )}
    </div>
  )
}

function CorrelationHeatmap({
  tickers,
  corr,
}: {
  tickers: string[]
  corr: number[][]
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse border border-gray-300 text-sm">
        <thead>
          <tr className="bg-gray-100">
            <th className="border border-gray-300 px-3 py-2"></th>
            {tickers.map((t) => (
              <th
                key={t}
                className="border border-gray-300 px-3 py-2 text-center font-semibold"
              >
                {t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {corr.map((row, i) => (
            <tr key={tickers[i]}>
              <th className="border border-gray-300 px-3 py-2 text-left bg-gray-100 font-semibold">
                {tickers[i]}
              </th>
              {row.map((v, j) => (
                <td
                  key={j}
                  className="border border-gray-300 px-3 py-2 text-center font-mono"
                  style={{ backgroundColor: corrColor(v) }}
                >
                  {v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function corrColor(v: number): string {
  // Positive correlations → blue (#3B82F6), negative → red (#EF4444).
  // Opacity scales with |v| so the diagonal is deep-blue, near-zero is white.
  const alpha = Math.min(0.75, Math.abs(v))
  if (v >= 0) return `rgba(59, 130, 246, ${alpha.toFixed(2)})`
  return `rgba(239, 68, 68, ${alpha.toFixed(2)})`
}
