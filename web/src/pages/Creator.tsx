import ReadingPage from '../components/ReadingPage'

export default function Creator() {
  return (
    <ReadingPage
      label="The creator"
      title="Axel McKenna"
      intro="Creator of DAMASCUS, a from-scratch LLM inference engine for Apple Silicon, and Liquorfy, a New Zealand liquor price comparison platform."
    >
      <section>
        <h2>Building systems by understanding them</h2>
        <p>
          I’m Axel McKenna. I build software that connects low-level engineering with practical
          applications. DAMASCUS is my exploration of what it takes to run a language model:
          loading its weights, turning text into tokens, computing a transformer forward pass,
          and checking every stage against an independent reference.
        </p>
        <p>
          The engine is written in C++20, with Metal as the planned GPU compute path on Apple M1.
          The source and session write-ups document the implementation, the mistakes, and the
          reasoning behind each step. Correctness comes before performance claims.
        </p>
      </section>
      <section>
        <h2>DAMASCUS: from model file to logits</h2>
        <p>
          The verified CPU path loads Llama 3.2 3B Instruct in GGUF format, implements byte-level
          BPE tokenization, dequantizes Q4_K and Q6_K weights, and runs the 28-layer transformer.
          Its forward pass reaches per-layer cosine similarity of 1.0000000 against a
          HuggingFace reference using the same quantized weights, with exact next-token agreement.
        </p>
        <p>
          Metal kernels and an AMX prefill path remain work in progress. The intended performance
          improvement is a research target, not a measured result. My{' '}
          <a href="/writing/building-damascus">technical write-up on building DAMASCUS</a>{' '}
          explains the architecture, validation, and benchmark plan in detail.
        </p>
      </section>
      <section>
        <h2>Liquorfy: making prices comparable</h2>
        <p>
          I also created <a href="https://www.liquorfy.co.nz/about">Liquorfy</a>, which brings
          together liquor prices from New Zealand retailers. The engineering challenge is
          turning inconsistent retailer data into useful comparisons: identifying the same
          product across stores, normalising quantities, and showing prices near the user.
        </p>
      </section>
      <section>
        <h2>Source code and engineering notes</h2>
        <p>
          Explore <a href="https://github.com/AxelMcKenna" rel="me">Axel McKenna on GitHub</a>,{' '}
          the <a href="https://github.com/AxelMcKenna/DAMASCUS">DAMASCUS repository</a>, and{' '}
          the <a href="https://github.com/AxelMcKenna/DAMASCUS/tree/main/docs">session write-ups</a>.
          For the project overview, return to <a href="/">DAMASCUS</a>.
        </p>
      </section>
    </ReadingPage>
  )
}
