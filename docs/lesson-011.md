# Lesson 011 — Compartment 4: Llama 3 tokenizer (byte-level BPE)

Date: 2026-06-

## What I set out to do
<!-- Goal + DoD. The acceptance bar is byte-identical round-trip vs the HF
     tokenizer AND reproducing the oracle's token ids for a fixed prompt. Why is
     "round-trip" not enough on its own — what does matching the oracle ids add? -->


## The encode pipeline, end to end
<!-- Walk the four stages: text --pretokenize--> chunks --byte_encode-->
     byte-char strings --BPE merges--> pieces --vocab lookup--> ids. What is the
     unit that flows between each stage, and where does each stage's table come
     from (vocab, merges, byte<->unicode)? -->


## The GPT-2 byte <-> unicode table
<!-- Why map every one of the 256 bytes to a *visible* codepoint (printable ones
     to themselves, the rest to U+0100..)? What goes wrong if a raw 0x20 space or
     a control byte flows into the BPE merge keys directly? Tie this to why a
     space becomes 'Ġ'. -->


## Why splitting a merge rule on the first space is unambiguous
<!-- merge_rank_ keys are "left + ' ' + right". Given the byte<->unicode mapping
     above, why can neither side ever contain a literal space? What bug would
     appear if that assumption were false? -->


## Reimplementing the llama-3 pre-tokenizer regex by hand
<!-- Why hand-roll the 7 alternatives (contractions, letters, numbers{1,3},
     punctuation, newline runs, whitespace) instead of std::regex? Pick two
     alternatives and explain what text they're meant to catch. -->


## The (?!\S) trailing-space rule (alt6)
<!-- The whitespace-run alternative leaves a single leading space behind for the
     *next* chunk when non-whitespace follows. Why? What byte-identical mismatch
     vs HF would show up if you greedily swallowed that space instead? -->


## The ASCII \p{L} / \p{N} approximation
<!-- is_letter() treats all non-ASCII as "letter unless whitespace". Why is this
     safe for the Milestone-A oracle prompt, and why is it nonetheless a known
     follow-up (what real input would it mis-segment)? -->


## Greedy BPE by rank
<!-- Why merge the *lowest-rank* adjacent pair first, repeatedly, until none
     apply? What is the complexity of the scan-per-merge loop, and why is that
     acceptable here (chunk sizes)? -->


## decode(): the inverse, and the non-byte-level fallback
<!-- Decode concatenates byte-char pieces then maps each codepoint back to its
     original byte. What is the fallback branch for a piece that isn't in
     unicode_to_byte_ (e.g. a special token), and why emit it literally? -->


## The TokenizerData handoff from Compartment 3
<!-- Where does the constructor's input come from, and why the
     data.model != "gpt2" guard up front? -->


## What's verified vs. not
<!-- Be specific about HOW it was verified this time. Did the round-trip /
     oracle-id check actually run (Docker/Linux CPU path, since the tokenizer has
     no Metal dependency)? Contrast with the EDR-blocked native path. -->


## What I'd revisit
<!-- ICU-grade Unicode property tables, special/added-token handling, the O(n^2)
     BPE loop if chunks ever get large. -->
