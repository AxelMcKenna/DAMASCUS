import { useReveal } from '../useReveal'

// Section wrapper - monospace index header + hairline rule.
export function Block({ index, label, title, intro, children }) {
  const ref = useReveal()
  return (
    <section className="border-b border-steel-800">
      <div className="mx-auto max-w-[90rem] px-6 py-20 sm:px-10 sm:py-28">
        <div ref={ref} className="reveal">
          <div className="flex items-center gap-4 font-mono text-xs">
            <span className="text-forge-300 glow">[ {index} ]</span>
            <span className="uppercase tracking-[0.32em] text-steel-400">{label}</span>
            <span className="h-px flex-1 bg-steel-800" />
          </div>
          {title && (
            <h2 className="display mt-7 max-w-3xl text-3xl font-extralight leading-[1.15] sm:text-5xl">
              {title}
            </h2>
          )}
          {intro && (
            <p className="mt-6 max-w-2xl font-mono text-sm leading-relaxed text-steel-400">
              {intro}
            </p>
          )}
        </div>
        <div className="mt-14">{children}</div>
      </div>
    </section>
  )
}

const GLYPH = {
  done: { c: '■', cls: 'text-steel-400' },
  active: { c: '▸', cls: 'text-forge-300 glow' },
  todo: { c: '□', cls: 'text-steel-700' },
}

export function Glyph({ status }) {
  const g = GLYPH[status] || GLYPH.todo
  return <span className={`${g.cls} font-mono`}>{g.c}</span>
}

export const STATUS_LABEL = {
  done: 'done',
  active: 'active',
  todo: 'queued',
}

// ASCII progress bar built from block characters.
export function Bar({ label, fill, note, accent }) {
  const total = 16
  const f = Math.max(0, Math.min(total, Math.round(fill * total)))
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-sm">
      <span className="w-20 shrink-0 text-steel-400">{label}</span>
      <span className={accent ? 'text-forge-300 glow' : 'text-steel-500'}>
        [{'█'.repeat(f)}
        <span className="text-steel-800">{'░'.repeat(total - f)}</span>]
      </span>
      <span className="text-xs text-steel-500">{note}</span>
    </div>
  )
}
