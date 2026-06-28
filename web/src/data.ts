// Content sourced from the project's README.md and ARCHITECTURE.md.

export type Status = 'done' | 'active' | 'todo'

export interface Compartment {
  n: number
  name: string
  job: string
  win: string
  status: Status
  headline?: boolean
}

export interface Milestone {
  id: string
  title: string
  def: string
  done?: boolean
  current?: boolean
  headline?: boolean
}

export const COMPARTMENTS: Compartment[] = [
  {
    n: 1,
    name: 'Platform',
    job: 'Metal device/queue/buffer setup, unified memory, --device-info',
    win: 'Scaffold compiles; prints device, GPU family, memory',
    status: 'done',
  },
  {
    n: 2,
    name: 'Tensor & memory',
    job: 'Tensor type (shape, dtype, stride, MTLBuffer backing), views, lifetime',
    win: 'Allocate / fill / read-back a tensor; clean under ASan',
    status: 'done',
  },
  {
    n: 3,
    name: 'Model loading',
    job: 'Parse GGUF, map weights to tensors, expose config',
    win: 'Loads 3B; reported config matches HF',
    status: 'done',
  },
  {
    n: 4,
    name: 'Tokenizer',
    job: 'Byte-level BPE encode/decode against the model vocab',
    win: 'Byte-identical round-trip vs HF tokenizer',
    status: 'done',
  },
  {
    n: 5,
    name: 'Compute kernels',
    job: 'GEMV/GEMM, RMSNorm, RoPE, softmax, SwiGLU, residual - CPU reference (Metal port pending)',
    win: 'CPU kernels match hand-computed values to f32 epsilon; Metal validates vs these next',
    status: 'done',
  },
  {
    n: 6,
    name: 'Transformer block',
    job: 'One decoder layer (GQA + KV cache, SwiGLU FFN), stacked',
    win: 'Forward pass matches HF same-weights oracle - cosine 1.0, exact top-1 (Milestone A)',
    status: 'done',
  },
  {
    n: 7,
    name: 'Sampling',
    job: 'Logits → token: greedy, temperature, top-k, top-p',
    win: 'Greedy is deterministic; output is coherent',
    status: 'active',
  },
  {
    n: 8,
    name: 'Runtime',
    job: 'Generation loop: prefill vs decode, KV-cache mgmt, command buffers',
    win: 'Sustained generation without recompute blowup',
    status: 'todo',
  },
  {
    n: 9,
    name: 'Quantization & AMX',
    job: 'Q4_K_M dequant; quantized matmul; fused dequant→AMX prefill; GPU+AMX hybrid',
    win: 'Quant within accuracy budget; AMX prefill beats baseline',
    status: 'todo',
    headline: true,
  },
]

export const MILESTONES: Milestone[] = [
  { id: '0', title: 'Scaffold compiles', def: '`--device-info` runs and prints the device', done: true },
  { id: 'A', title: 'Forward pass correct', def: 'Logits match HF same-weights oracle - per-layer cosine 1.0, exact top-1 (CPU path)', done: true },
  { id: 'B', title: 'Coherent generation', def: 'Greedy decode reads as fluent, sane English', current: true },
  { id: 'C', title: 'Within 50% decode', def: '≥ 0.5× llama.cpp decode throughput' },
  { id: 'D', title: 'Within 10% decode', def: '≥ 0.9× llama.cpp decode, reproducibly' },
  { id: 'E', title: '+30% prefill via AMX', def: '≥ 1.30× llama.cpp prefill - 10-run medians, non-overlapping p95', headline: true },
]

export const MODEL_SPECS: [string, string][] = [
  ['Layers', '28'],
  ['Hidden dim', '3072'],
  ['FFN dim', '8192'],
  ['Attention heads', '24'],
  ['KV heads (GQA)', '8'],
  ['Head dim', '128'],
  ['Vocab', '128256'],
  ['RoPE theta', '500000'],
  ['Embeddings', 'tied'],
  ['Norm', 'RMSNorm'],
  ['FFN', 'SwiGLU'],
  ['Quant', 'Q4_K_M'],
]

export const FLOW: string[] = [
  'token id',
  'embedding lookup',
  'RMSNorm',
  'QKV proj',
  'RoPE',
  'attn · KV cache (GQA)',
  '+ residual',
  'RMSNorm',
  'SwiGLU FFN',
  '+ residual',
  '×28 layers',
  'final RMSNorm',
  'tied LM head',
  'logits',
  'sample',
]
