# 04 — Tweaking: what helps, what does not (all measured)

Hardware: free Colab T4 (15 360 MB VRAM, ~12.7 GB RAM), Colab image
`torch 2.11.0+cu130`, Python 3.13, model Qwen-Image-2.1 **Q4_K_M** GGUF
(fully resident in VRAM), int8 text encoder, bf16 VAE. Same seed (42) and
prompts across all rows. Numbers trace to the GPUTests lab repo,
`projects/2026-W39-qwen-image21-t4/` (`timings-v2b.json`,
`results-v3/timings_v3.json`, `results-v4/timings_v4_turbo.json`).

## The measured table

| Config | s/step | Cold image* | Warm image | Peak VRAM |
|---|---|---|---|---|
| 1024², 20 steps, fp32 (ComfyUI default cast on T4) | 38.0 | 825.6 s | 765.6 s | 14 039 MB (91 %) |
| 768², 12 steps, fp32 | ~21.5 | 342.6 s | not measured | — |
| 768², 12 steps, fp16 + `--disable-comfy-compiler` | 6.20 | 127.9 s | 79.2 s | not instrumented (fp16 lowers it) |
| 768², 12 steps, fp16, unsloth weights (in-session anchor) | 5.29 | — | 69.1 s | — |
| **768², 6 steps, fp16 + Viggle turbo LoRA, CFG off** | **2.68** | **72.1 s** | **21.0 s** | ~13.3/15.64 GB |
| 768², 8 steps, fp16 + Viggle turbo LoRA, CFG off | 3.13 | — | 30.0 s | — |

\* cold = first image after server boot; includes GGUF load + text-encode
warm-up. The turbo rows and the anchor row are one session on the same
unsloth weights (2026-09-26, `timings_v4_turbo.json`); the older rows ran on
abenzerps' Q4_K_M before it 404ed. Cross-weight ratios are not valid — see
the in-session anchor rule in SKILL.md invariant 12.

**The turbo LoRA is the big lever: 3.3× over the 12-step fp16 combo on
identical weights** (21.0 s vs 69.1 s, same session), ~3.8× over the
historical 79.2 s default. The fp16 flags still apply on top — the turbo
recipe assumes them.

## The turbo recipe (Viggle v0.2.1, verified 2026-09-26)

[Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)
is a DMD2-distilled LoRA claiming 6 steps at competitive quality. The claim
held on free-tier GGUF weights, **but the recipe has three load-bearing
parts** — drop any one and you are back to mush:

1. **Runtime-hook LoRA, never merged.** The author's `comfyui/viggle_turbo.py`
   (`ViggleTurboLora`) applies the LoRA inside the forward pass. Their own
   measurement: merging into bf16 weights keeps ~70 % of the update; on
   quantized weights the requant noise is ~4× the update size. The 679.6 MB
   r128 file at strength 1.0 is their ComfyUI-recommended artifact.
2. **The sigma schedule is part of the recipe.** `ViggleTurboSigmas` builds
   `1.0, 0.9375, 0.875, 0.75, 0.5, 0.25` shifted per-resolution (dynamic
   exponential shift from token count). Stock KSampler cannot express this —
   sampling needs BasicGuider + KSamplerSelect(euler) + SamplerCustomAdvanced.
   Need denser small text? Add steps **only at the high-noise end**, keeping
   `0.875, 0.75, 0.5, 0.25` fixed: 8 steps =
   `1.0, 0.96875, 0.9375, 0.90625, 0.875, 0.75, 0.5, 0.25` (30.0 s measured).
3. **CFG fully off.** BasicGuider runs the positive pass only — 6 forward
   passes per image vs 24 for the 12-step cfg-2.5 combo. That is where most
   of the speedup comes from (per-step cost halves: 2.68 vs 5.29 s/step).

Quality at 6 steps on Q4_K_M, eyeballed from committed PNGs: the neon sign
spells **FREE GPU LAB** perfectly (the same canary that collapses at 4 naive
steps); the photoreal cat keeps every prompt element. Known gaps per the
author: multi-reference edits, small dense text at 6 steps (the 8-step ladder
narrows it). License: Qwen Research (non-commercial) — same as the base model.

## Knobs, with expected effects

| Knob | Effect | Notes |
|---|---|---|
| turbo LoRA (workflow swap) | **× 3.3** | verified (rows above). Needs the custom node + LoRA file + CFG off, all three |
| `--steps` (baseline workflow) | linear in time | 12 = verified sweet spot without the LoRA. Naive 4 steps WITHOUT the LoRA collapses text rendering (measured: sign illegible at cfg 2.5, partially readable at cfg 1.0) — few-step claims need a distilled LoRA behind them |
| turbo steps 6 → 8 | +~43 % time | only for small/dense text; add high-noise sigmas only (recipe above) |
| `--width/--height` (workflow) | ~quadratic | 768² verified. 1024² fp16 should be ~6.2×(1024/768)² ≈ 11 s/step baseline (untested); turbo at 1024² untested — the sigma shift auto-adapts to resolution by design |
| `--cfg` (baseline workflow) | ~2× if set to 1.0 | cfg > 1 runs TWO forward passes per step. cfg 1.0 alone (no LoRA) is measured only at 4 steps, where quality partially collapsed — treat as untested at 12 |
| `--negative` (turbo) | none | the turbo graph has CFG off; BasicGuider consumes no negative. Do not "restore" CFG to use negatives |
| `--seed` (workflow) | free | fixes composition for A/B comparisons |
| sampler/scheduler | ~0 | baseline: `res_multistep`/`simple`. turbo: euler is the author's choice; the sigmas node carries the schedule |
| quant (Q4_K_M vs Q8_0 etc.) | ~0 speed | **compute-bound**: same FLOPs, same cast. Q8_0 (7.59 GB) should still fit VRAM at 768² (untested); quants decide *what fits*, not speed |
| `--lowvram` | hurts or nothing | nothing is offloaded at these sizes (GGUF "full load: True"); do not add preemptively. Turbo adds the 0.68 GB r128 LoRA — VRAM stayed at ~13.3/15.64 GB measured |
| smaller text encoder (w4a8, 6.31 GB) | ~0 | the int8 encoder costs ~10 s warm; swap only if RAM-constrained elsewhere |

## VRAM arithmetic (when the user wants more)

- Diffusion GGUF file GB × 1024 + ~1500 MB activations/VAE + ~700 MB turbo
  LoRA ≤ free VRAM. Q4_K_M at 1024² fp32 peaked 91 % — fp16 at 768² has
  room; Q8_0 at 1024² fp16 is the plausible next rung (untested — attempt
  and catch the OOM; OOM is a result, record it).
- System RAM is NOT the constraint people fear: the 9.35 GB int8 encoder
  peaks ~75–78 % of the 12.7 GB VM because ComfyUI streams it ("dynamic
  VRAM loading"). Do not pre-optimize this.

## Sampling settings in the verified workflows

- Turbo (`workflows/t2i-turbo.json`): 6 sigma nodes `1.0 … 0.25`, euler,
  CFG off, LoRA strength 1.0, denoise implicit 1.0. Per the Viggle card.
- Baseline (`workflows/t2i-api.json`): `steps 12, cfg 2.5, sampler
  res_multistep, scheduler simple, denoise 1.0, negative prompt ""`. The
  official 2.1 card publishes no sampler settings and demonstrates 40 steps
  at 2048² in diffusers — ours is the T4-budget adaptation, quality verified
  visually, not assumed.
