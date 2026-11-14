import type { ReactNode } from 'react'

export default function ReadingPage({ label, title, intro, children }: {
  label: string
  title: string
  intro: string
  children: ReactNode
}) {
  return (
    <article className="border-b border-steel-800">
      <div className="mx-auto max-w-4xl px-6 py-14 sm:px-10 sm:py-20">
        <nav aria-label="Breadcrumb" className="mb-14 flex flex-wrap gap-x-5 gap-y-3 font-mono text-xs text-steel-300">
          <a href="/" className="hover:text-forge-300">← Damascus</a>
          <a href="/axel-mckenna" className="hover:text-forge-300">Axel McKenna</a>
          <a href="/writing/building-damascus" className="hover:text-forge-300">Engineering notes</a>
        </nav>
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-forge-300">{label}</p>
        <h1 className="mt-6 text-4xl font-extralight leading-tight sm:text-6xl">{title}</h1>
        <p className="mt-7 max-w-2xl font-mono text-sm leading-relaxed text-steel-300">{intro}</p>
        <div className="mt-14 space-y-12 text-base leading-8 text-steel-300 [&_h2]:mb-5 [&_h2]:text-2xl [&_h2]:font-light [&_h2]:text-steel-50 [&_p+p]:mt-5 [&_a]:text-forge-300 [&_a]:underline [&_a]:underline-offset-4 [&_code]:break-words [&_code]:font-mono [&_code]:text-sm">
          {children}
        </div>
      </div>
    </article>
  )
}
