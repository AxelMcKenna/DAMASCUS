const LOAD_OUTPUT = [
  ['gguf', 'v3 · Q4_K_M · 1.9 GiB'],
  ['arch', 'llama · 28 layers · 3072 hidden'],
  ['ffn', 'SwiGLU · 8192 · RMSNorm'],
  ['attn', 'GQA · 24 heads / 8 kv'],
  ['vocab', '128,256 · tied embeddings'],
]

export default function Hero() {

  return (
    <header className="relative overflow-hidden">
      <div className="absolute inset-x-0 top-0 h-px flow-line" />
      <div
        className="pattern-breathe absolute inset-0"
        style={{ backgroundImage: 'url(/img/damascus.png)', backgroundSize: 'cover', backgroundPosition: 'center' }}
      />

      <div className="relative mx-auto max-w-[90rem] px-6 sm:px-10">
        <div className="py-20 lg:py-28">
          <p
            className="lift font-mono text-xs uppercase tracking-[0.4em] text-forge-300 glow"
            style={{ animationDelay: '0.05s' }}
          >
            // solo build · in progress
          </p>

          <div className="mt-6 grid grid-cols-1 items-start gap-14 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
            {/* left */}
            <div className="min-w-0">
              <h1
                className="lift display font-thin leading-[0.95]"
                style={{ animationDelay: '0.15s', fontSize: 'clamp(2.6rem, 10vw, 8rem)', letterSpacing: '0.14em' }}
              >
                Damascus
              </h1>
              <p
                className="lift mt-8 max-w-xl font-mono text-sm leading-relaxed text-steel-300"
                style={{ animationDelay: '0.25s' }}
              >
                I'm building a from-scratch LLM inference engine for Apple&nbsp;M1 - every line by hand
                in C++20 + Metal. The bet I'm chasing: a hand-rolled{' '}
                <span className="text-forge-300 glow">AMX prefill</span> path that beats{' '}
                <span className="text-steel-100">llama.cpp</span> where the M1 has real headroom.
              </p>
            </div>

            {/* right - terminal usage example */}
            <div className="lift w-full min-w-0 max-w-md lg:mt-3 lg:justify-self-end" style={{ animationDelay: '0.4s' }}>
            <div className="border border-steel-800 bg-steel-900/50 backdrop-blur-sm">
              <div className="flex items-center justify-between border-b border-steel-800 px-5 py-3 font-mono text-xs text-steel-500">
                <span>damascus - zsh</span>
                <span>apple m1</span>
              </div>
              <div className="overflow-x-auto px-5 py-5 font-mono text-sm leading-relaxed">
                <div className="whitespace-nowrap text-steel-200">
                  <span className="text-forge-300">$</span> damascus load llama-3.2-3b-q4_k_m.gguf
                </div>
                <div className="mt-2 space-y-0.5">
                  {LOAD_OUTPUT.map(([k, v]) => (
                    <div key={k} className="flex gap-3 whitespace-nowrap">
                      <span className="w-14 shrink-0 text-steel-600">{k}</span>
                      <span className="text-steel-300">{v}</span>
                    </div>
                  ))}
                  <div className="mt-1.5 whitespace-nowrap text-steel-400">
                    loaded in 0.4s · config <span className="text-forge-300">✓</span> matches reference
                  </div>
                </div>
                <div className="mt-3 text-steel-200">
                  <span className="text-forge-300">$</span> <span className="caret text-forge-300">█</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    </header>
  )
}
