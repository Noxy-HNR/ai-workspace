# Local LLM inference: research shortlist, 2026-09-23

**Hardware:** RTX 5070 Ti Laptop 12 GB (Blackwell, sm_120), Core Ultra 7 255HX, **16 GB system RAM**, Windows 11.
**Your llama.cpp:** b11041 (built 2026-09-18). **Latest:** b11147 / v0.5.0 (2026-09-23).
**Your models:** Qwen3.5-9B UD-Q6_K_XL (active, 114K context, f16 KV) and Qwen3.8-27B-Uncensored IQ3_M (12 GB, 60 GPU layers, 20K context, q8_0 KV).

**Baseline report:** I couldn't find an earlier report on this machine or in past sessions, so this one is the baseline. It covers roughly mid-August to today. Older items appear only if they affect your setup and you aren't using them yet. The next report should compare against this file.

**Labels used below:**
- **[IND]** — measured by an independent third party
- **[COMM]** — community-measured, one person or one rig, not peer-reviewed
- **[VENDOR]** — claimed by the model author, the quantizer or the tool's own maintainers

---

## 1. Worth acting on

### 1a. Turn on MTP for your Qwen3.8-27B. It's already in your file.
- **What I checked:** your `Qwen3.8-27B-Uncensored-IQ3_M.gguf` contains the MTP tensors (`qwen35.nextn_predict_layers`, `blk.64.nextn.*`). Your b11041 build already has `--spec-type draft-mtp`, and the CUDA-graph fix for MTP drafts (#28549, merged 2026-09-16) is included.
- **Evidence:**
  - [COMM] RTX PRO 3000 Blackwell **Laptop 12 GB**, 38 of 66 layers on the GPU: **5.3 → 10.8 tok/s (+104%)**. On an RTX 5060 Laptop 8 GB, IQ4_XS, partial offload: 4.2 → 7.3 tok/s. The same source says MTP costs about **1–2 GB extra VRAM**, gains grow with output length, and it only helps with a single request at a time (`--parallel 1`). ([sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp))
  - [COMM] RTX 3090, UD-Q4_K_XL, greedy decoding: 41.6 → 66.4 tok/s at n-max 2 (+60%). n-max 3 and 5 were slower. ([thc1006, HackMD](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090))
  - [VENDOR] The llama.cpp maintainer's PR reports about 75% acceptance with 3 draft tokens and a speedup above 2× (Qwen3.6). Prompt processing gets slightly slower. ([#22673](https://github.com/ggml-org/llama.cpp/pull/22673))
- **Try:** add `--spec-type draft-mtp --spec-draft-n-max 2` to `extraArgs`. Because the 1–2 GB overhead pushes more layers onto the CPU, compare it against the current config at the same context length.
- **Risk:** yours is an *uncensored fine-tune*. If the MTP head wasn't retrained alongside the weights, fewer draft tokens will be accepted. Check the `draft acceptance` value in the llama-server log. None of the tests above covered a fine-tune, and none covered sampling at temperature 1.0 with thinking on.

### 1b. For the 27B on 12 GB: a ~10.6–10.7 GB quant that fits fully on the GPU probably beats your 12 GB IQ3_M
Your current 12 GB IQ3_M can't fit entirely in VRAM (60 of about 65 layers on the GPU), and the 16 GB of system RAM leaves little room for spillover.
- [IND] **Quesma** (2026-08-26; GPQA-D, IFBench, Terminal-Bench 2.1; single runs with 95% confidence intervals): **UD-Q2_K_XL (10.7 GB)** keeps about 85% on GPQA vs about 87% at BF16, and about 70% vs 75% on Terminal-Bench. Q4_K_M scores the same as BF16. 1-bit quants collapse. 3-bit wasn't tested. ([Quesma](https://quesma.com/blog/qwen38-27b-quantizations-benchmarked/))
- [IND] **Kaitchup** (2026-09-01; about 950 prompts from MMLU-Pro, LiveCodeBench and GPQA, 3 sampled runs each): Q3 variants such as **UD-IQ3_XXS (10.6 GB)** and UD-Q3_K_XL (12.8 GB) stay close to baseline. Lower quants lose accuracy steadily. It only tested single-turn tasks, and losses in agent loops are likely larger. Per-quant numbers are partly paywalled. ([Kaitchup](https://kaitchup.substack.com/p/qwen38-27b-gguf-benchmark-q4-to-q1))
- [VENDOR, quantizer comparing its own files] Atomic Chat: its AD-IQ3_S (13.8 GB) has 33% lower KLD than Q3_K_M at the same size. It also says Unsloth's UD 2-bit files beat its own below 11 GB. ([HF discussion](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/65))
- **Implication:** the official (not uncensored) **UD-IQ3_XXS** or **UD-Q2_K_XL** plus q8_0 KV plus MTP is the combination most likely to run fully on the GPU. It's still tight, so check the llama-server log. Unsloth's Qwen3.8 GGUFs keep the MTP head ([sudoingX list](https://github.com/sudoingX/qwen38-mtp)). Switching away from the uncensored fine-tune is your call.

### 1c. Qwen3.8-27B thinks too much by default. Lower `reasoning_effort`.
- [IND] **Artificial Analysis:** Intelligence Index **34**, ranked #1 of 142 open models between 4B and 40B parameters. It is **very verbose**: 200M output tokens across the index vs a median of 82M. ([AA](https://artificialanalysis.ai/models/qwen3-8-27b)) Some secondary sites quote other AA numbers (for example "52"), probably from an older index version. Rely on AA's own page.
- [COMM] Simon Willison: the default `xhigh` effort causes minutes of overthinking. He recommends `low` or `medium`, or turning thinking off. ([simonwillison.net](https://simonwillison.net/2026/Aug/16/qwen-38-27b/))
- [VENDOR] Qwen model card: SWE-bench Pro **61.7**, LiveCodeBench v6 **90.3**, Terminal-Bench 2.1 **73.0**. Its internal benchmarks can't be reproduced independently. ([HF card](https://huggingface.co/Qwen/Qwen3.8-27B))
- **Try:** `--chat-template-kwargs "{\"reasoning_effort\":\"medium\"}"` (PowerShell escaping, per [Unsloth docs](https://unsloth.ai/docs/models/qwen3.8)). Check that the uncensored fine-tune's chat template still supports it. On a partially offloaded 27B, cutting thinking tokens likely saves more wall-clock time than any kernel improvement.

### 1d. Your Qwen3.5-9B file has no MTP head. An MTP version exists.
- I checked your `Qwen3.5-9B-UD-Q6_K_XL.gguf`: it has no `nextn` tensors, so `draft-mtp` can't work with it. [unsloth/Qwen3.5-9B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF) includes them.
- [VENDOR] Unsloth claims about 1.5–2× faster decoding with no accuracy loss. I found no independent 9B measurement. This isn't new (it dates from around May), but you aren't using it.
- **VRAM:** with 114K context at f16 KV, a Q6 9B already fills most of the card. To make room for MTP's 1–2 GB, you'd likely need q8_0 KV or a shorter context.

---

## 2. llama.cpp changes b11041 → b11147 (what matters for you)
- **No must-have CUDA change for your GPU since your build.** Kernel work you already have: K-quant MMVQ→MMQ crossover tuned per GPU, including sm_120 (#26079, 08-20); branchless Q4_K/Q5_K unpack (#26705, 09-07); CUDA graphs for MTP (#28549, 09-16).
- New since your build:
  - Gemma 4 FlashAttention tuning on Ampere and newer (#29152): **about 0–2%** [VENDOR, maintainer's own benchmark on RTX 3090].
  - **Gemma 4 DSpark draft support** (#29226, merged today): speculative decoding for Gemma 4 12B using separate draft GGUFs ([example draft](https://huggingface.co/ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF)). No independent llama.cpp speed numbers yet. [VENDOR paper: [DSpark](https://arxiv.org/pdf/2607.05147)]
  - DFlash speculative decoding for HunyuanOCR (#28890).
  - CUB argsort corruption fix (#28389).
  - Top-k MoE fix (#28432).
  - Sampler environment variables (#27380).
  - Server can bind to multiple addresses (#28690).
  - Version bumped to 0.5.0.
- **DFlash2** (#27342, 08-27) supports external draft models. On an RTX 3090 it performed about the same as built-in MTP (63 vs 66 tok/s [COMM, thc1006]), but it needs a separate draft model in VRAM, so **on 12 GB, MTP is the better choice**. The same PR says GGUFs converted before 08-27 need reconverting if you use vision.
- **Windows:** prebuilt CUDA 13.4 x64 zips are now published next to the 12.4 ones. You build from source, so this only matters if you want a quick fallback binary.
- **Unverified digests:** a Buttondown "Weekly GitHub Report" misreported PR contents (for example, #28815 described as NVFP4 Blackwell work). I dropped claims I couldn't match to GitHub.

## 3. Models: new, but skip or wait
- **Qwen3.8-Flash-Next** (08-26): 125B total / 6B active parameters plus a 51B n-gram table. The smallest good quant is about 94 GB, so it **won't fit in 16 GB RAM + 12 GB VRAM**. [VENDOR-ish leaderboard] It tops open models on SWE-bench Pro at 62.5% ([morphllm](https://www.morphllm.com/swe-bench-pro)).
- **Qwen 4 27B** was announced at Apsara on 09-22 with **no weights, card or benchmarks yet** ([orcarouter](https://www.orcarouter.ai/blog/qwen-4-max-lineup-announced-apsara-2026)). It's the most likely next upgrade for 12 GB. Watch for it.
- **"Qwen3.8-9B-Distill"** (Empero) is a community fine-tune of Qwen3.5-9B. Its benchmarks are self-reported and mixed (GSM8K slightly down). Not recommended without independent evaluation ([HF](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill)).
- **Gemma 4 12B** (June, not new): [IND] AA Intelligence Index **14** vs a class median of 8, 113 tok/s via API ([AA](https://artificialanalysis.ai/models/gemma-4-12b)). It's far below Qwen3.8-27B's 34 on AA's current index. DSpark support makes it a fast fallback, not a quality upgrade.

## 4. KV cache and long context: research, low actionability
- [IND, academic] Agrawal & Mayer ([arXiv 2607.05399](https://arxiv.org/abs/2607.05399)): **compression ratio is a poor predictor of end-to-end performance**. 4-bit KIVI was the most consistent for quality. Tested on Llama-3.1-8B and Mistral-7B, not llama.cpp.
- [IND, academic] KVDiagnosis ([arXiv 2608.09412](https://arxiv.org/abs/2608.09412), 08-10): compressed-KV failures split into likelihood drift (the quantization-like kind) and lost evidence position (the eviction-like kind). About 60K runs on Qwen3-8B.
- **For you:** Qwen3.5/3.8 are hybrid models where only one layer in four uses full attention (Gated DeltaNet), so their KV cache is already small. [IND] Quesma measured **about 2.3 GB per 32K tokens at f16** for Qwen3.8-27B, so q8_0 is about 1.2 GB per 32K. q8_0 KV remains the low-risk default. No new llama.cpp KV cache format landed since your build. Hadamard K-rotation for quantized KV was already in before July.

---

## Suggested test order (all reversible, profile edits in `Config/models.json`)
1. 27B: add `--spec-type draft-mtp --spec-draft-n-max 2` and reasoning effort `medium`. Compare tok/s and the draft acceptance rate against the current profile.
2. If acceptance is poor or VRAM is too tight: add a profile for the official Qwen3.8-27B **UD-IQ3_XXS** (about 10.6 GB) with MTP and q8_0 KV. That's a download of about 10.6 GB, so ask me first.
3. 9B: switch to the Unsloth **Qwen3.5-9B MTP** GGUF with q8_0 KV and MTP.
4. Upgrading llama.cpp to b11147 is optional. Do it only if you want Gemma 4 DSpark.
