import { Block } from './Term'
import { FLOW } from '../data'

export default function DataFlow() {
  return (
    <Block
      index="04"
      label="one decode step"
      title="Token in, logits out - 28 layers of the same graph."
      intro="Prefill runs this exact graph over every prompt position at once (matrices, not vectors), filling the KV cache so decode can begin. Prefill is where AMX earns its keep."
    >
      <div className="border border-steel-800 bg-steel-900/40 p-7 font-mono text-sm">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-3">
          {FLOW.map((step, i) => {
            const loop = step === '×28 layers'
            const terminal = step === 'token id' || step === 'logits' || step === 'sample'
            return (
              <span key={step} className="flex items-center gap-2">
                <span
                  className={`border px-2.5 py-1 ${
                    loop
                      ? 'border-forge-500/50 text-forge-300 glow'
                      : terminal
                        ? 'border-metal-500/40 text-metal-400'
                        : 'border-steel-800 text-steel-300'
                  }`}
                >
                  {step}
                </span>
                {i < FLOW.length - 1 && <span className="text-steel-700">→</span>}
              </span>
            )
          })}
        </div>

        <div className="mt-7 grid gap-px overflow-hidden border border-steel-800 bg-steel-800 sm:grid-cols-2">
          <div className="bg-steel-950 p-5">
            <p className="text-xs uppercase tracking-widest text-forge-300 glow">prefill</p>
            <p className="mt-2 text-xs leading-relaxed text-steel-400">
              whole prompt at once. GEMM-heavy, compute-bound - the target for the fused dequant→AMX
              kernel.
            </p>
          </div>
          <div className="bg-steel-950 p-5">
            <p className="text-xs uppercase tracking-widest text-metal-400">decode</p>
            <p className="mt-2 text-xs leading-relaxed text-steel-400">
              one token at a time. GEMV-heavy, bandwidth-bound - reads and writes the KV cache each
              step.
            </p>
          </div>
        </div>
      </div>
    </Block>
  )
}
