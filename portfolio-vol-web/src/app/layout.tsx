import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "Portfolio Realized Vol — Moontower",
  description:
    "Compute realized portfolio vol, weighted-average component vol, and a correlation matrix from a list of tickers and weights.",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  )
}
