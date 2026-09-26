# Dedicated coding models vs your best coder, 2026-09-26

**Hardware:** RTX 5070 Ti Laptop 12 GB (Blackwell), 15.4 GB RAM, Windows, llama.cpp b11041 (stock, self-built 2026-09-18).
**Your best coder:** Huihui Qwen3.8-27B abliterated `GSQ-RCO-IQ3_XXS-mtp` (10,442,827,296 B = 9.73 GiB, fully on GPU).
Measured here 2026-09-26: ~48-51 tok/s prose with MTP, ~145-196 tok/s on copy-heavy code with `draft-mtp,ngram-mod`;
24,576 ctx with 629-633 MiB VRAM free after use. Prior report: `2026-09-23-local-inference-report.md` (a different report type: inference tooling).

**Labels:** [IND] independent measurement, [COMM] community, one person/rig, [VENDOR] the model author's own claim (or a lab's own
published number).
Sources are primary model cards, the Hugging Face API and the actual GGUF headers where possible. SEO "best coding LLM for 12 GB" listicles
(which still recommend Qwen2.5-Coder 14B in 2026) were discarded.

---

## Bottom line
**No dedicated coding model that runs on this machine has credible evidence of beating Qwen3.8-27B.** Everything with stronger claims is far
too large; everything that fits is either older, or a vendor-only claim on a different harness with no head-to-head against Qwen3.8-27B.
The only candidate worth a local head-to-head is Ornith-1.5-35B-A3B (section 3).

## 1. Bigger than your card (ruled out on size, not merit)
| model | size | note |
|---|---|---|
| Qwen3-Coder-Next 80B-A3B (Jan 2026) | UD-Q2_K_XL 26.8 GB; 1-bit-class UD-TQ1_0 18.9 GB, UD-IQ1_S 21.5 GB | unsloth GGUF sizes; exceeds 12 GB VRAM + free RAM; 1-bit quants are not a serious option |
| Qwen3.8-Flash-Next (125B/6B active) | ~94 GB smallest good quant (9/23 report) | no |
| Kimi-K2.7-Code (1.03T), GLM-5.3 (753B), DeepSeek-V4.1-Flash (763B), MiniMax-M3 (427B), MiMo-V2.6-Flash (311B) | hundreds of GB | no |
| Qwen 4 27B | announced 9/22 | still **no weights on Hugging Face** (searched 2026-09-26) |
Not relevant: MiniMax-H3 (video generation), Qwen-AgentWorld-35B-A3B (environment-simulation world model).

## 2. The baseline
**Qwen3.8-27B** (Aug 2026, Apache-2.0). [VENDOR, Qwen card] SWE-bench Pro 61.7, Terminal-Bench 2.1 (Terminus) 73.0, LiveCodeBench v6 90.3.
Vendor-aggregate ranking of open weights on SWE-bench Pro: Flash-Next 62.5, GLM-5.2 62.1, Qwen3.8-27B 61.7 (a 27B within one point of models
10-30x its size; secondary aggregator).

## 3. What fits: 35B MoE (3B active) models
Fit note: at IQ3_XXS (~15 GB) roughly 10 GB stays in VRAM with the KV cache and about 5 GB of experts sit in system RAM, which your 15.4 GB
handles. The 21+ GB Q4_K_M files (10-11 GB of experts in RAM) are the tight ones. Speeds on this machine are unmeasured. 3-bit MoE quality is
unmeasured for all of these.

**Ornith-1.5-35B-A3B** (ornith-ai / deep-reinforce, 2026-08-18, MIT; a different model from the Ornith-1.5-9B you ran and removed).
- [VENDOR] its own table (single runs, OpenHands / Terminus-2 / Claude Code harnesses): SWE-bench Verified 79, Pro 59.6, Multilingual 71.4,
  Terminal-Bench 2.1 67.8 (Terminus-2) / 68.5 (Claude Code), NL2Repo 46.2. Same table: Qwen3.6-35B-A3B 73.4 / 49.5 / 67.2 / 52.5.
- It does **not** compare against Qwen3.8-27B. Vendor numbers on different harnesses are close (Pro 59.6 vs 61.7, TB 67.8 vs 73.0), so no
  ranking is possible. Footnote says the Qwen chat template needs modifying for their harness.
- GGUF: `bartowski/Ornith-1.5-35B-A3B-GGUF` IQ3_XXS 15.34 GB, IQ2_M 12.54 GB, IQ3_XS 16.69 GB; `ornith-ai/Ornith-1.5-35B-A3B-GGUF` Q4_K_M 21.71 GB.
- Verdict: the strongest vendor claims of anything that can run here, MIT, and 3B active means fast decode. Unverified independently.

**KAT-Coder-V2.5-Dev** (Kwaipilot, 2026-07-23), 35B MoE / 3B active, fine-tune of Qwen3.6-35B-A3B, Apache-2.0, text-only.
- [VENDOR] its table compares against **Qwen3.5-27B, not Qwen3.8-27B** (released two weeks later): SWE-bench Verified 69.4 vs 68.6, Multilingual
  63.0 vs 57.7, Pro 45.96 vs 42.13, Terminal-Bench 2.1 41.0 vs 34.8.
- [VENDOR vs VENDOR] The model-card thread "community benchmark results" (#12) lists each lab's own published scores: KAT 69.4 / 45.96 against
  Qwen's own 73.4 / 49.5 for Qwen3.6-35B-A3B and Ornith-1.0's 75.6 / 50.4. The poster asks why they differ; Kwaipilot answers that it re-ran
  everything on one harness, single run, and that Qwen optimised for those benchmarks. Not an independent re-run.
- [VENDOR, third lab] NVIDIA's own table for its Nemotron model also runs Qwen3.6-35B-A3B: SWE-bench Verified 70.12, Multilingual 63.40,
  Terminal-Bench 2.1 44.38 (NVIDIA's harness), i.e. about KAT's level (69.4 / 63.0 / 41.0 on KAT's harness). No evidence KAT improved on its base.
- [COMM] anecdotes: many fewer thinking tokens (one tester about 1/7 of Qwen3.6-35B's on a local agent task), stable at long context; mixed
  quality reports, and one tester retracted a "beat Qwen 27B" result after re-running ("might have just gotten lucky").
- GGUF (bartowski): IQ2_XXS 9.8 GB, IQ3_XXS 14.9 GB, Q4_K_M 21.4 GB.
- Verdict: no independent head-to-head against Qwen3.8-27B exists. One generation older; a token-efficiency experiment, not a quality upgrade.

**Nemotron-3.5-Lightning-30B-A3B** (NVIDIA, 2026-08-01): [VENDOR] its own column: SWE-bench Verified 51.56, Terminal-Bench 2.1 24.58
(the bold 70.12 / 44.38 in that table are Qwen3.6-35B-A3B). Model card says it is a general model meant for customization. Out.

**Older dedicated coders (superseded):** Devstral Small 2 24B (2025-12) [VENDOR] SWE-bench Verified 68.0, Multilingual 55.7, Terminal-Bench 2
22.5. Qwen3-Coder-30B-A3B (2025-07) 31.8 Verified in KAT's harness. gpt-oss-20b, Seed-Coder-8B (2025). Gemma-4-31B 60.6 Verified in KAT's
harness. Nothing suggests they beat a 2026-08 flagship.

**Community "coder" merges of Qwen3.8-27B** (Signal-/Swift-/NEO-CODER-MAX-...-Terse-Coder): the Signal-3.8-27B-Terse-Coder card says it is a LoRA merge that
cuts reasoning tokens, states **no independent benchmarks**, and its adapter author notes 70% to 60-62% on a held-out set after merging. Avoid.

**Independent leaderboard check:** [IND] SWE-rebench (fresh, contamination-controlled) currently covers tasks from 2026-05-15 to 07-01, which
**predates both Qwen3.8-27B and KAT-Coder**; neither is on it. (Page summarised by a fetch tool; verify before quoting.)

## 4. Options inside your current family (each needs a download you approve)
**A. Official Qwen3.8-27B GGUFs from `unsloth/Qwen3.8-27B-GGUF`.** Verified from the remote GGUF header (in memory, no download): the
official UD-IQ3_XXS has the same 866 tensors as your file, including all 15 `blk.64.*` tensors (attention, FFN and the four `nextn.*`), so the MTP
head is complete in the main file. Coding does not need refusal removal, and abliteration plus a different 3-bit recipe is unmeasured against it.
**It is not a drop-in on VRAM:**
| file | size | vs your file | consequence at MTP n=2, ctx 24576 |
|---|---|---|---|
| `Qwen3.8-27B-UD-IQ3_XXS.gguf` | 10.93 GB (10.18 GiB) | +~467 MiB | ~165 MiB free (your gate is 600). Keep MTP -> ctx must drop to about 12K (about 154 MiB per 4K at q8_0); drop MTP -> about 32K ctx at ~34 tok/s instead of ~50 |
| `Qwen3.8-27B-UD-Q2_K_XL.gguf` | 9.83 GB (9.15 GiB) | -~0.6 GiB | fits MTP at 24576 with ~1.2 GB spare (room for ~32K ctx); quality risk |
Quality evidence: [IND, from the 9/23 report] Quesma measured UD-Q2_K_XL as 10.7 GB at that time (GPQA about 85 vs about 87 BF16, Terminal-Bench about 70
vs about 75; 3-bit not tested by Quesma). Kaitchup [IND] found Q3-class quants such as UD-IQ3_XXS (10.6 GB then) and UD-Q3_K_XL close to baseline.
File sizes have changed since (UD-IQ3_XXS 10.6 -> 10.93 GB, UD-Q2_K_XL 10.7 -> 9.83 GB), so those evaluations may be of earlier uploads.
Test locally (PPL plus a small coding pass-rate) before switching.

## 5. Side notes
- **Ternary-Bonsai-2-27B** (PrismML, 2026-09-16, Apache-2.0; Qwen3.8-27B distilled to ternary; 5.95 GB PTQ1_0 / 7.21 GB PQ2_0; 262K ctx).
  [VENDOR] coding parity on H100/vLLM: HumanEval+ 95.1, MBPP+ 83.1, LiveCodeBench 90.07 vs FP16 90.05 (IQ2_XXS collapses to 56.4).
  [COMM] HF #54 on a V100: KL divergence about 40x a normal Q4_K_XL's, different top token about 1 in 4.5, 38% faster decode, "different, not
  necessarily worse", task evals pending. [COMM] two first-hand coding failures: an unplayable Tetris page (HF #63; Qwen3.8-27B Q8_0 worked for the
  same tester) and a planted bug left unfixed plus a dead button (MindStudio hands-on, 3 correction attempts). Requires PrismML's llama.cpp fork
  (stock rejects the types; the fork is 127 commits ahead, 589 behind upstream, merge base 2026-08-25, older than your b11041). Not proven for coding.
- **MiMo-V2.6-Distill-Qwen-9B** (Xiaomi, 2026-09-21, MIT): only relevant to the 9B slot and it cannot answer this question (a 9B does not beat the 27B).
  [VENDOR] SFT of Qwen3.5-9B: SWE Pro 44.6 vs 32.0, SWE Verified 61.1 vs 60.0, Terminal-Bench 2.1 37.1 vs 27.0. GGUFs by bartowski and ggml-org.
  Nothing new since the 9/23 discussion; still vendor-only. Watch item.

## 6. What I would do
- Keep the current setup for coding.
- If you want to test something: (1) Ornith-1.5-35B-A3B IQ3_XXS as a head-to-head (fits; only candidate with strong claims), (2) official
  UD-Q2_K_XL or UD-IQ3_XXS with the context/speed tradeoffs above.
- Watch: Qwen 4 27B weights, a Qwen3.8-27B-based coder from Qwen or Mistral, SWE-rebench rows for Qwen3.8-27B.

## Things I could not verify
- No independent head-to-head of Qwen3.8-27B vs KAT or Ornith-1.5 exists. The SWE-rebench summary came from a fetch tool. "1/7 tokens" is a single
  anecdote. Speeds and 3-bit MoE quality on this machine are unmeasured. Whether abliteration costs coding quality is unmeasured.
