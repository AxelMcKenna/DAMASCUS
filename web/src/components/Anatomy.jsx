import { Block } from './Term'

const ROWS = [
  ['attention', '→ GPU', 'SIMD-friendly, runs as Metal compute shaders'],
  ['FFN', '→ AMX', 'outer-product-dense - what the matrix unit wants'],
  ['buffer', '→ shared', 'no copies between them; the hardware agrees'],
]

export default function Anatomy() {
  return (
    <Block
      index="03"
      label="zero-copy / unified memory"
      title="One pool of memory. Two engines reading it at once."
      intro="On Apple Silicon the CPU and GPU share physical memory, so a shared-storage MTLBuffer needs no copy - and the same memory is what AMX reads. That is the basis of the GPU+AMX hybrid: attention on the GPU, the dense FFN on AMX, coordinated through one buffer with no transfers."
    >
      <div className="max-w-2xl border border-steel-800 font-mono text-sm">
        <div className="border-b border-steel-800 px-5 py-3 text-xs text-steel-500">
          schedule.map
        </div>
        {ROWS.map(([a, b, d]) => (
          <div key={a} className="border-b border-steel-800/70 px-5 py-4 last:border-b-0">
            <div className="flex items-baseline gap-3">
              <span className="text-steel-100">{a}</span>
              <span className="text-forge-300 glow">{b}</span>
            </div>
            <p className="mt-1 text-xs text-steel-500">{d}</p>
          </div>
        ))}
      </div>
    </Block>
  )
}
