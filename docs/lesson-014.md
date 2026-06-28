# Lesson 014 — Compartment 6: The forward pass → Milestone A

Date: 2026-06-

## What I set out to do
<!-- DoD: the full decoder forward pass produces logits that match the HF oracle
     (Milestone A). What exactly is being composed here — name the Compartment 5
     kernels and the Compartment 4/3 inputs this stitches together. -->


## The Model accessor — just-in-time dequant
<!-- Model::get(name) finds a tensor and dequantizes it to f32 on demand. Why pull
     each weight just-in-time inside the matmul instead of materializing all ~3B
     params in f32 up front? What's the peak-memory argument (≈ one big weight at
     a time)? -->


## The Llama decoder graph
<!-- Sketch it: embed → 28 × [RMSNorm → GQA attention (RoPE) → residual,
     RMSNorm → SwiGLU FFN → residual] → final RMSNorm → tied LM head → logits.
     Where do the two residual adds per layer attach, and why does the residual
     stream `x` carry through untouched between the norm taps? -->


## GQA: how query heads share KV heads
<!-- n_rep = n_head / n_kv, and query head h reads kv head h / n_rep. With 24 query
     heads and 8 kv heads, which queries share a kv head? Why does K/V have stride
     kv_dim (n_kv*head_dim) while Q has stride d? -->


## Causal attention and the 1/√head_dim scale
<!-- Keys only run j = 0..i (causal). Where does the 1/sqrt(head_dim) factor go,
     and why before the softmax rather than after? -->


## Why rope_interleaved here
<!-- Tie back to Lesson 012: the forward pass calls rope_interleaved on raw GGUF
     Q and K. Restate in one or two sentences why the GGUF weight permutation
     forces the interleaved (NORM) convention, and what the logits would look like
     if you'd used the half-split variant by mistake. -->


## The tied LM head
<!-- The final logits are token_embd reused as the output projection (mm(embd,
     final_norm)). What does "tied embeddings" mean and why can the same [d, vocab]
     tensor serve both the input lookup and the output projection? -->


## Matching the oracle's hidden-state layout
<!-- res.hidden is (n_layers+1) × seq × d. Index 0 = embeddings, index k = residual
     after layer k-1, and final_norm is exposed separately. Why does this layout
     have to match the oracle's hidden_index_legend exactly — what does per-layer
     diffing let you localize when logits are wrong? -->


## The weight caveat — what number actually counts as PASS
<!-- The engine runs Q4_K_M weights; the oracle runs fp32 HF weights. They differ
     by quantization noise, normally bigger than 1e-3 on raw logits. So for the
     run you can do NOW, which metrics are trustworthy (cosine > 0.9999, top-1 /
     top-5 token agreement) vs. the literal max-abs < 1e-3 (which needs a
     same-weights oracle)? -->


## What's verified — Milestone A, for real
<!-- This is the big one. Did the forward pass actually run end-to-end and get
     scored by compare.py? Through what path — the Docker/Linux escape (notarized
     Docker launcher → Linux guest where unsigned torch/numpy/the engine run),
     since the host EDR still SIGKILLs native binaries? Record the numbers:
     per-layer cosine ≈ 1.0, top-1 agreement, where (if anywhere) divergence
     first appears. Contrast with lessons 9–10 where nothing could be run. -->


## What I'd revisit
<!-- No KV cache yet (prefill recomputes all positions — that's Compartment 8), the
     same-weights 1e-3 oracle (llama.cpp-on-GGUF or a GGUF-dequant injection mode),
     and the O(seq²) attention loop. -->
