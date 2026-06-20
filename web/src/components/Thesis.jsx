import { Block, Bar } from './Term'

export default function Thesis() {
  return (
    <Block
      index="01"
      label="the thesis"
      title="Decode is capped. Prefill has headroom no public runtime touches."
      intro="On an M1, decode is bandwidth-bound - ~35 tok/s on 3B Q4 is roughly the ceiling no matter how clever the kernel. Prefill is compute-bound and underused: the GPU and the AMX coprocessor can run in parallel, and AMX's opcodes are undocumented, so nobody exploits them."
    >
      <div className="grid gap-px overflow-hidden border border-steel-800 bg-steel-800 lg:grid-cols-2">
        <Pane
          tag="decode"
          state="bandwidth-bound"
          fill={0.28}
          note="~35 tok/s ceiling"
          points={[
            'one token at a time - GEMV-heavy',
            'limited by weight streaming from memory',
            'clever kernels reach the ceiling, never move it',
            'C / D targets: reach decode parity with llama.cpp',
          ]}
        />
        <Pane
          tag="prefill"
          accent
          state="compute-bound"
          fill={0.92}
          note="+30% target"
          points={[
            'whole prompt at once - GEMM-heavy, dense matmuls',
            'GPU + AMX coprocessor run simultaneously',
            'fused dequant→AMX never materializes FP16 weights',
            'E target: +30%, 10-run medians, non-overlapping p95',
          ]}
        />
      </div>

      <p className="mt-8 max-w-3xl font-mono text-xs leading-relaxed text-steel-500">
        <span className="text-forge-300">&gt;</span> AMX is driven via raw{' '}
        <span className="text-steel-300">.word</span>-encoded instructions - bootstrapped from{' '}
        <span className="text-steel-300">cblas_sgemm</span> (Accelerate already uses AMX) to get the
        through-the-framework number, then hand-rolled to beat it.
      </p>
    </Block>
  )
}

function Pane({ tag, state, fill, note, points, accent }) {
  return (
    <div className="bg-steel-950 p-7">
      <div className="flex items-center justify-between font-mono text-xs">
        <span className={accent ? 'uppercase tracking-widest text-forge-300 glow' : 'uppercase tracking-widest text-steel-400'}>
          {tag}
        </span>
        <span className="text-steel-500">{state}</span>
      </div>
      <div className="mt-5">
        <Bar label="compute" fill={fill} note={note} accent={accent} />
      </div>
      <ul className="mt-6 space-y-2.5 font-mono text-sm text-steel-300">
        {points.map((p) => (
          <li key={p} className="flex gap-2.5">
            <span className={accent ? 'text-forge-300' : 'text-steel-600'}>-</span>
            {p}
          </li>
        ))}
      </ul>
    </div>
  )
}
