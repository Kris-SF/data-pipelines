import Link from "next/link"

export function Header() {
  return (
    <header className="bg-white border-b border-gray-200">
      <div className="mx-auto max-w-7xl px-6 py-4 lg:px-8 flex items-center justify-between">
        <Link
          href="/"
          className="text-xl font-bold tracking-tight text-gray-900"
        >
          Moontower
        </Link>
        <nav className="text-sm text-gray-600">
          <Link
            href="/tools-and-games"
            className="hover:text-indigo-600 transition-colors"
          >
            Tools and Games
          </Link>
        </nav>
      </div>
    </header>
  )
}
