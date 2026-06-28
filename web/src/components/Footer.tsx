export default function Footer() {
  return (
    <footer className="relative overflow-hidden">
      <div
        className="pattern-breathe absolute inset-0"
        style={{ backgroundImage: 'url(/img/damascus.png)', backgroundSize: 'cover', backgroundPosition: 'center' }}
      />
      <div className="relative mx-auto max-w-[90rem] px-6 py-24 sm:px-10 sm:py-28">
        <p className="font-mono text-xs uppercase tracking-[0.4em] text-steel-500">// eof</p>
        <h2 className="display mt-6 text-5xl font-thin leading-none sm:text-7xl" style={{ letterSpacing: '0.14em' }}>Damascus</h2>
        <p className="mt-6 max-w-xl font-mono text-sm leading-relaxed text-steel-400">
          A from-scratch LLM inference engine for Apple M1, written by hand. Currently at compartment
          6 of 9, Milestone A reached - the source and per-session write-ups are public.
        </p>

        <div className="mt-10 flex flex-wrap gap-x-6 gap-y-2 font-mono text-xs">
          <a href="https://github.com/AxelMcKenna/DAMASCUS" target="_blank" rel="noreferrer" className="text-steel-300 transition-colors hover:text-forge-300">
            github.com/AxelMcKenna/DAMASCUS ↗
          </a>
          <a href="https://github.com/AxelMcKenna/DAMASCUS/tree/main/docs" target="_blank" rel="noreferrer" className="text-steel-400 transition-colors hover:text-forge-300">
            session write-ups ↗
          </a>
        </div>

        <div className="mt-14 flex flex-col gap-3 border-t border-steel-800 pt-6 font-mono text-xs text-steel-500 sm:flex-row sm:items-center sm:justify-between">
          <span className="tracking-[0.3em] text-steel-300">DAMASCUS - by Axel McKenna</span>
          <span>c++20 · metal · apple m1</span>
        </div>
      </div>
    </footer>
  )
}
