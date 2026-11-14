import ReadingPage from '../components/ReadingPage'

export default function TechnicalArticle() {
  return (
    <ReadingPage
      label="Engineering notes · by Axel McKenna"
      title="Building DAMASCUS: a language model from file to forward pass"
      intro="Why I’m building an inference engine for Apple M1, how I verify the CPU path, and what a fair Metal and AMX performance comparison would need to demonstrate."
    >
      <p>By <a href="/axel-mckenna" rel="author">Axel McKenna</a> · Based on the project’s documented Milestone A results.</p>
      <section>
        <h2>Start with the boundaries of the problem</h2>
        <p>
          DAMASCUS targets one model, one stream, and one machine: Llama 3.2 3B Instruct,
          quantized as Q4_K_M GGUF, on an Apple M1 with 16 GB of unified memory.
          That scope makes the work inspectable. Each of the nine compartments has an explicit
          responsibility and a correctness check before it can become a performance experiment.
        </p>
        <p>
          Those compartments cover platform setup, tensor storage, model loading, tokenization,
          compute kernels, transformer composition, sampling, the generation runtime, and
          quantization with AMX. The first six form the path from bytes on disk to next-token
          logits. Sampling and sustained generation come after that path is verified.
        </p>
      </section>
      <section>
        <h2>Read the model, then reproduce its operations</h2>
        <p>
          A GGUF loader must interpret the header, metadata, tensor directory, and quantized
          blocks consistently. Shapes and model configuration come from the file; a tensor
          with the right byte count can still be wrong if its dimensions or quantization layout
          are interpreted incorrectly. The Q4_K and Q6_K dequantizers are checked against the
          gguf library before their output feeds the transformer.
        </p>
        <p>
          The tokenizer is a separate correctness boundary. Byte-level BPE must produce the
          same token IDs as the reference, and decoding must preserve the original text.
          A plausible-looking token sequence is insufficient: a different tokenization changes
          every later activation, making a forward-pass comparison meaningless.
        </p>
        <p>
          Inside a decoder layer, RMSNorm precedes the QKV projections. RoPE supplies positional
          information, grouped-query attention reads the KV cache, and the attention output
          returns through a residual connection. A second RMSNorm and a SwiGLU feed-forward
          network complete the layer. After 28 layers, final normalization and the tied
          embedding matrix produce logits over 128,256 vocabulary entries.
        </p>
      </section>
      <section>
        <h2>A same-weights oracle makes the comparison useful</h2>
        <p>
          Comparing quantized inference to an unquantized model would mix implementation error
          with quantization error. Instead, the HuggingFace oracle loads the same Q4_K_M GGUF
          weights and dequantizes the same blocks. For the prompt “The capital of France is”,
          the documented CPU run reports per-layer cosine similarity of 1.0000000 and exact
          top-1 agreement. The next token is “ Paris”, at 68.65% in DAMASCUS and 68.63% in the oracle.
        </p>
        <div className="overflow-x-auto border border-steel-800">
          <table className="w-full min-w-[32rem] text-left font-mono text-xs leading-6">
            <caption className="p-4 text-left text-steel-300">Documented CPU correctness results, not throughput benchmarks</caption>
            <thead className="border-y border-steel-800 text-steel-50"><tr><th scope="col" className="p-4">Comparison</th><th scope="col" className="p-4">Cosine similarity</th><th scope="col" className="p-4">Max absolute error</th></tr></thead>
            <tbody><tr className="border-b border-steel-800"><th scope="row" className="p-4 font-normal">Hidden states (29 × 5 × 3072)</th><td className="p-4">1.0000000</td><td className="p-4">1.27e-2</td></tr><tr><th scope="row" className="p-4 font-normal">Logits (5 × 128256)</th><td className="p-4">1.0000000</td><td className="p-4">1.33e-2</td></tr></tbody>
          </table>
        </div>
        <p>
          The absolute maximum exceeds the original 1e-3 acceptance threshold. The project
          documents that limitation: the largest differences occur on large activation values,
          while cosine similarity, exact next-token agreement, and top-five overlap support
          the CPU result. Publishing the error alongside the agreement is more informative
          than describing the result as bit-exact.
        </p>
        <p>
          The <a href="https://github.com/AxelMcKenna/DAMASCUS#verified-results-cpu-path">README’s verified results</a>{' '}
          and <a href="https://github.com/AxelMcKenna/DAMASCUS/tree/main/tools/oracle">oracle harness</a>{' '}
          contain the comparison details. The commands <code>make cpu-check</code>,{' '}
          <code>make oracle-gguf</code>, and <code>make forward-check</code> reproduce the
          tokenizer, kernel, and forward-pass checks with the required model and Docker environment.
        </p>
      </section>
      <section>
        <h2>Prefill and decode need different performance questions</h2>
        <p>
          Decode processes one token at a time. Its matrix-vector operations repeatedly stream
          model weights, so memory bandwidth can dominate. Prefill processes many prompt tokens
          together and uses matrix-matrix operations, giving more opportunities to reuse weights
          and occupy compute units. An optimization that helps one phase may barely affect the other.
        </p>
        <p>
          The planned Metal path maps suitable operations to the GPU. The AMX experiment asks
          whether fused dequantization and matrix computation can reduce intermediate memory
          traffic during prefill, eventually alongside GPU work using unified memory.
          A 30% prefill improvement over llama.cpp is the target. It has not been established
          by the current results, and this article does not report a Metal or AMX speedup.
        </p>
      </section>
      <section>
        <h2>What a reproducible benchmark must record</h2>
        <p>
          A throughput claim needs a pinned model hash, quantization format, llama.cpp revision,
          hardware configuration, macOS version, prompt length, and generation length.
          Both implementations need the same workload and a controlled thermal state.
          The planned baseline measures 1,024-token prefill separately from 256-token decode,
          with warmup excluded and repeated runs rather than a single favourable measurement.
        </p>
        <p>
          The project’s <a href="https://github.com/AxelMcKenna/DAMASCUS/blob/main/baseline.md">baseline protocol</a>{' '}
          calls for ten-run medians, p95, and variation, as well as cold-start latency, peak
          resident memory, and sustained throughput after thermal load. Its results table is
          still unpopulated. Until those measurements exist, memory use and speed remain
          questions to measure, not numbers to infer from model file size or correctness tests.
        </p>
      </section>
      <section>
        <h2>Current limits and the next step</h2>
        <p>
          The published milestone is a verified CPU forward pass. The documented development
          environment used a Linux Docker guest to run the CPU checks when native binary
          execution was blocked; that environment cannot validate Metal or AMX. GPU validation,
          coherent generation, and reproducible performance measurements remain separate gates.
        </p>
        <p>
          Follow the <a href="https://github.com/AxelMcKenna/DAMASCUS">source repository</a> and{' '}
          <a href="https://github.com/AxelMcKenna/DAMASCUS/tree/main/docs">session write-ups</a>{' '}
          for the evidence behind each change, or read more about <a href="/axel-mckenna">my work</a>.
        </p>
      </section>
    </ReadingPage>
  )
}
