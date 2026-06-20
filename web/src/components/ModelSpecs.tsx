import { Block } from './Term'
import { MODEL_SPECS } from '../data'

export default function ModelSpecs() {
  return (
    <Block
      index="06"
      label="the target"
      title="Llama 3.2 3B Instruct · Q4_K_M"
      intro="Big enough that prefill matmuls are real work for AMX, small enough to iterate on a 16 GB M1. Every number is read from the GGUF and asserted - never hardcoded."
    >
      <div className="grid gap-x-12 gap-y-0 border-t border-steel-800 font-mono text-sm sm:grid-cols-2">
        {MODEL_SPECS.map(([k, v]) => (
          <div
            key={k}
            className="flex items-baseline gap-3 border-b border-steel-800/70 py-3"
          >
            <span className="text-steel-400">{k}</span>
            <span className="min-w-0 flex-1 translate-y-[-3px] border-b border-dotted border-steel-800" />
            <span className="text-steel-100">{v}</span>
          </div>
        ))}
      </div>
    </Block>
  )
}
