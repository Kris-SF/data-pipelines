import { NextResponse } from "next/server"
import { fetchAdjClose, type PriceSeries } from "../../../lib/yahoo"
import {
  alignPanels,
  componentVol,
  correlationFromCov,
  logReturns,
  populationCovariance,
  portfolioVol,
} from "../../../lib/vol"

export const runtime = "nodejs"
export const maxDuration = 30

const TRADING_DAYS_PER_YEAR = 252
const WEIGHT_TOLERANCE = 1e-6
const MIN_OBS = 20

type Holding = { ticker: string; weight: number }

type ApiBody = {
  holdings?: Holding[]
  start?: string
  end?: string
  normalize?: boolean
}

export async function POST(req: Request) {
  let body: ApiBody
  try {
    body = (await req.json()) as ApiBody
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 })
  }

  const { holdings, start, end, normalize } = body

  if (!holdings?.length) {
    return NextResponse.json({ error: "No holdings provided." }, { status: 400 })
  }
  if (!start || !end) {
    return NextResponse.json(
      { error: "Missing start or end date." },
      { status: 400 },
    )
  }
  if (new Date(start) >= new Date(end)) {
    return NextResponse.json(
      { error: "start must be before end." },
      { status: 400 },
    )
  }

  // --- validate holdings ---
  const seen = new Set<string>()
  const tickers: string[] = []
  let weights: number[] = []
  for (const h of holdings) {
    const t = (h.ticker ?? "").trim().toUpperCase()
    if (!t) {
      return NextResponse.json(
        { error: `Invalid ticker: ${JSON.stringify(h.ticker)}` },
        { status: 400 },
      )
    }
    if (seen.has(t)) {
      return NextResponse.json(
        { error: `Duplicate ticker: ${t}` },
        { status: 400 },
      )
    }
    const w = Number(h.weight)
    if (!Number.isFinite(w)) {
      return NextResponse.json(
        { error: `${t}: weight is not a finite number` },
        { status: 400 },
      )
    }
    seen.add(t)
    tickers.push(t)
    weights.push(w)
  }

  const total = weights.reduce((a, b) => a + b, 0)
  if (normalize) {
    if (total <= 0) {
      return NextResponse.json(
        { error: `Cannot normalize: weights sum to ${total}` },
        { status: 400 },
      )
    }
    weights = weights.map((w) => w / total)
  } else if (Math.abs(total - 1) > WEIGHT_TOLERANCE) {
    return NextResponse.json(
      {
        error: `Weights must sum to 1.0 (got ${total.toFixed(
          4,
        )}). Enable "auto-normalize" to rescale.`,
      },
      { status: 400 },
    )
  }

  // --- fetch prices ---
  const panels: Record<string, PriceSeries> = {}
  const fetchErrors: string[] = []
  await Promise.all(
    tickers.map(async (t) => {
      try {
        panels[t] = await fetchAdjClose(t, start, end)
      } catch (e) {
        fetchErrors.push((e as Error).message)
      }
    }),
  )
  if (fetchErrors.length) {
    return NextResponse.json(
      { error: fetchErrors.join("; ") },
      { status: 502 },
    )
  }

  // --- align and compute ---
  const aligned = alignPanels(panels, tickers)
  if (aligned.dates.length < MIN_OBS) {
    return NextResponse.json(
      {
        error: `Only ${aligned.dates.length} overlapping trading days across tickers — need at least ${MIN_OBS}.`,
      },
      { status: 400 },
    )
  }

  const { dates, returns } = logReturns(aligned)
  const cov = populationCovariance(returns, tickers.length)
  const dailyVol = portfolioVol(cov, weights)
  const componentDaily = componentVol(cov)
  const componentAnn = componentDaily.map(
    (v) => v * Math.sqrt(TRADING_DAYS_PER_YEAR),
  )
  const wavgDaily = weights.reduce(
    (a, w, i) => a + w * componentDaily[i],
    0,
  )
  const corr = correlationFromCov(cov)

  return NextResponse.json({
    tickers,
    weights,
    dates: {
      start: dates[0],
      end: dates[dates.length - 1],
      count: dates.length,
    },
    portfolio: {
      daily: dailyVol,
      annualized: dailyVol * Math.sqrt(TRADING_DAYS_PER_YEAR),
    },
    wavgComponent: {
      daily: wavgDaily,
      annualized: wavgDaily * Math.sqrt(TRADING_DAYS_PER_YEAR),
    },
    diversificationRatio: dailyVol > 0 ? wavgDaily / dailyVol : null,
    components: tickers.map((t, i) => ({
      ticker: t,
      weight: weights[i],
      daily: componentDaily[i],
      annualized: componentAnn[i],
    })),
    correlation: corr,
  })
}
