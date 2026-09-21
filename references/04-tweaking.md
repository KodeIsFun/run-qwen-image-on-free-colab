# 04 — Tweaking: what helps, what does not (all measured)

Hardware: free Colab T4 (15 360 MB VRAM, ~12.7 GB RAM), Colab image
`torch 2.11.0+cu128`, Python 3.13, model Qwen-Image-2.1 **Q4_K_M** GGUF
(4.60 GB, fully resident in VRAM), int8 text encoder, bf16 VAE. Same seed
(42) and prompts across all rows. Numbers trace to the GPUTests lab repo,
`projects/2026-W39-qwen-image21-t4/`.

## The measured table

| Config | s/step | Cold image* | Warm image | Peak VRAM |
|---|---|---|---|---|
| 1024², 20 steps, fp32 (ComfyUI default cast on T4) | 38.0 | 825.6 s | 765.6 s | 14 039 MB (91 %) |
| 768², 12 steps, fp32 | ~21.5 | 342.6 s | not measured | — |
| **768², 12 steps, fp16 + `--disable-comfy-compiler`** | **6.20** | **127.9 s** | **79.2 s** | not instrumented (fp16 lowers it) |

\* cold = first image after server boot; includes GGUF load + text-encode
warm-up. Reproduction run of the combo row: 6.81 s/step — same conclusion.

**9.7× warm-image speedup over defaults**, and the two knobs are
independent:

- **fp16 cast** (× ~2.1): the T4 has no bf16, so ComfyUI default-casts the
  model to **fp32**, leaving the fp16 tensor cores idle. Requires BOTH
  `--force-fp16` AND `--disable-comfy-compiler` (troubleshooting §1).
- **fewer steps + smaller canvas** (× ~3.0): 20→12 steps is linear; 1024²→768²
  is ~× 1.8 (FLOPs scale with pixels). Quality loss at 12 steps/768² is small;
  text rendering still lands perfectly.

## Knobs, with expected effects

| Knob | Effect | Notes |
|---|---|---|
| `--steps` (workflow) | linear in time | 12 = verified sweet spot. Below ~8, flow-matching models get visibly mushy (untested here — labeled untested) |
| `--width/--height` (workflow) | ~quadratic | 768² verified. 1024² fp16 should be ~6.2×(1024/768)² ≈ 11 s/step (untested) |
| `--cfg` (workflow) | ~2× if set to 1.0 | cfg > 1 runs TWO forward passes per step (cond + uncond). cfg 1.0 halves compute but may flatten detail without a distilled model — **untested on 2.1; eyeball before shipping** |
| `--seed` (workflow) | free | fixes composition for A/B comparisons |
| sampler/scheduler | ~0 | `res_multistep`/`simple` carried from the Qwen-Image template; other samplers change speed little (compute-bound) |
| quant (Q4_K_M vs Q8_0 etc.) | ~0 speed | **compute-bound**: same FLOPs, same cast. Q8_0 (7.59 GB) should still fit VRAM at 768² (untested); quants decide *what fits*, not speed |
| Lightning / turbo LoRA | potentially × 5–10 | 4–8 step distills existed for Qwen-Image 1.0; a 2.1 equivalent was **not verified** — treat any claim as untested until run here |
| `--lowvram` | hurts or nothing | nothing is offloaded at these sizes (GGUF "full load: True"); do not add preemptively |
| smaller text encoder (w4a8, 6.31 GB) | ~0 | the int8 encoder costs ~10 s warm; swap only if RAM-constrained elsewhere |

## VRAM arithmetic (when the user wants more)

- Diffusion GGUF file GB × 1024 + ~1500 MB activations/VAE ≤ free VRAM.
  Q4_K_M at 1024² fp32 peaked 91 % — fp16 at 768² has room; Q8_0 at 1024²
  fp16 is the plausible next rung (untested — attempt and catch the OOM;
  OOM is a result, record it).
- System RAM is NOT the constraint people fear: the 9.35 GB int8 encoder
  peaks ~75–78 % of the 12.7 GB VM because ComfyUI streams it ("dynamic
  VRAM loading"). Do not pre-optimize this.

## Sampling settings in the verified workflow

`steps 12, cfg 2.5, sampler res_multistep, scheduler simple, denoise 1.0,
negative prompt ""`. The official 2.1 card publishes no sampler settings and
demonstrates 40 steps at 2048² in diffusers — ours is the T4-budget
adaptation, and quality was verified visually, not assumed.
