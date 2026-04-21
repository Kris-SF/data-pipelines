import Link from "next/link"

export default function RoundButton({
  text,
  url,
}: {
  text: string
  url: string
}) {
  return (
    <Link
      href={url}
      className="inline-block bg-indigo-600 text-white px-6 py-3 rounded-full font-semibold hover:bg-indigo-700 transition-colors"
    >
      {text}
    </Link>
  )
}
