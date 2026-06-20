import { Block } from './Term'
import { MILESTONES } from '../data'

export default function Milestones() {
  return (
    <Block
      index="05"
      label="the road to E"
      title="Five milestones. Every speed number proven, never a lucky run."
      intro="Same machine, model, quant, prompt, context length and thermal state - reported with variance. Baselines are measured before the first kernel is written."
    >
      <div className="border-t border-steel-800 font-mono text-sm">
        {MILESTONES.map((m) => {
          const glyph = m.current ? '▸' : m.headline ? '◆' : '□'
          const tone = m.current
            ? 'text-forge-300 glow'
            : m.headline
              ? 'text-forge-300'
              : 'text-steel-700'
          return (
            <div
              key={m.id}
              className="grid grid-cols-[auto_auto_1fr] items-baseline gap-x-4 border-b border-steel-800/70 py-4"
            >
              <span className={`${tone} w-4`}>{glyph}</span>
              <span className="w-6 text-steel-100">{m.id}</span>
              <div>
                <div className="flex flex-wrap items-baseline gap-x-3">
                  <span className="text-steel-100">{m.title}</span>
                  {m.current && <span className="text-xs text-forge-300">- we're here</span>}
                  {m.headline && <span className="text-xs text-forge-300 glow">- the win</span>}
                </div>
                <p className="mt-0.5 text-xs text-steel-500">{m.def}</p>
              </div>
            </div>
          )
        })}
      </div>
    </Block>
  )
}
