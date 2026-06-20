import { Block, Glyph, STATUS_LABEL } from './Term'
import { COMPARTMENTS } from '../data'

export default function Compartments() {
  return (
    <Block
      index="02"
      label="the pipeline"
      title="Nine compartments. Each ships only when it passes its oracle."
      intro="Correctness before speed, always - no optimizing a kernel that hasn't first matched its reference within tolerance."
    >
      <div className="border-t border-steel-800 font-mono text-sm">
        {COMPARTMENTS.map((c) => (
          <div
            key={c.n}
            className="group grid grid-cols-[auto_auto_1fr_auto] items-baseline gap-x-4 border-b border-steel-800/70 py-4 transition-colors hover:bg-steel-900/40"
          >
            <span className="text-steel-700">{String(c.n).padStart(2, '0')}</span>
            <Glyph status={c.status} />
            <div className="min-w-0">
              <span className={c.headline ? 'text-forge-300 glow' : 'text-steel-100'}>
                {c.name}
              </span>
              {c.headline && (
                <span className="ml-2 text-xs text-forge-300">[headline]</span>
              )}
              <span className="mt-0.5 block truncate text-xs text-steel-500">{c.job}</span>
            </div>
            <span
              className={`text-xs ${
                c.status === 'active'
                  ? 'text-forge-300'
                  : c.status === 'done'
                    ? 'text-steel-400'
                    : 'text-steel-600'
              }`}
            >
              {STATUS_LABEL[c.status]}
            </span>
          </div>
        ))}
      </div>
    </Block>
  )
}
