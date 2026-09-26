---
name: run-qwen-image-on-free-colab
description: Run Qwen-Image-2.1 (7B diffusion transformer, GGUF quantized) text-to-image
  on a free Colab T4 with ComfyUI headless, and expose the ComfyUI API through a public
  tunnel consumable from anywhere. Use when the user wants free-GPU image generation,
  a local text-to-image API without paying for GPU, or to tune/serve Qwen-Image GGUF
  models. Includes Colab CLI install/auth, a verified notebook, a stdlib-only client,
  the measured turbo recipe (Viggle distilled LoRA, 6 steps, CFG off, ~21 s per warm
  image), the 12-step no-LoRA baseline, and every gotcha paid for in real failed runs.
---

# Run Qwen-Image-2.1 on a free Colab T4

Everything here was **measured on real free T4s** (2026-09-21, Colab
`torch 2.11.0+cu128`, Python 3.13), not assembled from forum posts. The
numbers you quote live in
[references/04-tweaking.md](references/04-tweaking.md) and trace to
`projects/2026-W39-qwen-image21-t4/` in the GPUTests lab repo
(<https://github.com/KodeIsFun/GPUTests>).

## The one decision that routes everything

Ask (or infer): **what does the user actually want?**

| They want | Path | First action |
|---|---|---|
| To generate an image and see it work | Notebook in browser | Send [colab/run-qwen-image-t4.ipynb](colab/run-qwen-image-t4.ipynb) via the "Open in Colab" badge in the README — the human needs zero installed tools |
| A text-to-image API reachable from anywhere | Same notebook | Cells 6–7 print a public `trycloudflare.com` URL; then [clients/txt2img.py](clients/txt2img.py) from ANY machine |
| Agent-driven / headless runs | Colab CLI lane | [references/02-colab-cli-lane.md](references/02-colab-cli-lane.md) |
| Faster / bigger / different quality | Tuning | [references/04-tweaking.md](references/04-tweaking.md) |
| Something broke | Troubleshooting | [references/05-troubleshooting.md](references/05-troubleshooting.md) |
| What the human must do by hand | Manual steps | [references/01-manual-steps.md](references/01-manual-steps.md) |

**Default config (verified, turbo):** Qwen-Image-2.1 **Q4_K_M** GGUF (base
model from **unsloth**, 4.20 GB — abenzerps' original file 404s, issue #1) +
int8 text encoder + bf16 VAE via ComfyUI + the **leejet** fork of
ComfyUI-GGUF + Viggle's **turbo LoRA** via the author's `viggle_turbo.py`
custom node, launched with `--force-fp16 --disable-comfy-compiler`,
**768×768, 6 steps, CFG fully off** (BasicGuider), euler + the
ViggleTurboSigmas schedule → **~2.7 s/step, 21.0 s per warm image**, text
rendering intact ("FREE GPU LAB" neon sign, spelled correctly).

**Baseline config (verified, no LoRA):** same weights and flags, KSampler
**12 steps, cfg 2.5, res_multistep/simple** → **5.29 s/step, 69.1 s warm**.
Use it as the fallback if the turbo custom node ever breaks on a newer
ComfyUI, and as the apples-to-apples anchor for speed ratios.

## Invariants — never violate these

1. **Never leave a Colab VM running.** After CLI work: `colab stop -s <name>`,
   then `colab sessions` must be empty. **Download every artifact and verify
   each with `ls -la` BEFORE the stop** — stop is terminal, `/content` dies
   with the VM. (A bundled download+stop lost a full artifact set once.)
2. **The T4 flags are not optional.** Without `--force-fp16` the DiT runs a
   bf16→fp32 manual cast at 38.0 s/step. Without `--disable-comfy-compiler`
   the fp16 forward **crashes** with `RuntimeError: aimdo memory compile
   error` (ComfyUI's model compiler cannot compile the fp16 path on Turing).
   Both flags together = 6.20 s/step. See troubleshooting §1.
3. **The leejet fork only.** `git clone https://github.com/leejet/ComfyUI-GGUF`
   into `custom_nodes/`. The older city96 fork throws
   `Unknown model architecture!` on Qwen-Image-2.1 GGUFs (they are sd.cpp-
   flavored, `[arch:qwen_image21]`).
4. **Verify the GPU and the tunnel from OUTSIDE the VM** before telling the
   user anything works: read `nvidia-smi`, and `curl <tunnel-url>/system_stats`
   from your own machine. A URL printed on the VM is not proof it is reachable.
5. **`colab` CLI timeouts default to 30 SECONDS**, `colab run` executes .py
   only (notebooks go through `colab new` → `colab exec -f <nb.ipynb> →
   colab stop`), and every job script must be **fresh-VM safe** (carry its own
   install + download steps with cached-file skips — never assume a previous
   session's `/content`).
6. **The RAM fear is unfounded — do not "fix" it.** The 9.35 GB int8 text
   encoder fits on a 12.7 GB-RAM VM because ComfyUI stages it ("dynamic VRAM
   loading"); peak RAM measured 75–78 %. Swapping in smaller encoders or
   `--lowvram` preemptively is wasted motion.
7. **Quant size does not buy speed.** The step is compute-bound: Q4_K_M vs
   Q8_0 runs the same FLOPs through the same cast. Quants decide *what fits
   in VRAM*, nothing else. Say this when the user asks for "a smaller quant
   to go faster".
8. **License**: Qwen-Image-2.1 is under the **Qwen Research License** — fine
   for experimentation and research use; flag it if the user's use looks
   commercial.
9. **Every number you quote traces to the measured table** in
   references/04-tweaking.md. Configurations not yet measured (cfg 1.0
   without the LoRA, Q8_0 speed, 1024²-fp16) are labeled untested there — do
   not present them as results.
10. **The turbo LoRA is never merged and CFG is never turned back on.** The
    recipe needs the author's `viggle_turbo.py` node: the LoRA applied by
    runtime hooks (merging drops most of its update — lossy on every weight
    format) and the resolution-shifted sigma schedule that stock KSampler
    cannot express. "Just set steps=6 on the old graph" produces mush — that
    is the measured naive few-step failure, see tweaking.
11. **The diffusion GGUF comes from unsloth** (`unsloth/Qwen-Image-2.1-GGUF`,
    base model). abenzerps' repo was restructured to Uncensored-only and the
    non-UC file 404s — that is user-facing issue #1 in this repo. TE + VAE
    still resolve at abenzerps' paths. Always assert downloaded file sizes.
12. **Speed ratios come from in-session anchors only.** The unsloth and
    abenzerps Q4_K_M files are different quant runs (4.20 vs 4.60 GB) with
    measurably different speed; never divide a turbo timing by a timing from
    another session or another weight file.

## The 60-second version (agent driving the CLI lane)

```bash
uv tool install google-colab-cli       # or: pipx install google-colab-cli
colab --auth=oauth2 status             # human clicks one OAuth consent in browser
colab new -s qwenimg --gpu T4
colab exec -s qwenimg -f colab/run-qwen-image-t4.ipynb --timeout 1500
# grep the executed notebook output for "OK:" markers + "API URL:"
colab stop -s qwenimg && colab sessions   # must be empty
```

Full operational details, polling patterns, and artifact handling:
[references/02-colab-cli-lane.md](references/02-colab-cli-lane.md).
