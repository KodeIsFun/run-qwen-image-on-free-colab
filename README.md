# run-qwen-image-on-free-colab

Run **Qwen-Image-2.1** (7B diffusion transformer, GGUF-quantized) as a real
text-to-image service on a **free** Colab T4 — and call it **from any
machine** through a public API URL. Every number in this repo was measured on
an actual free T4, and every gotcha was paid for in a real failed run.
Nothing is aspirational.

Written to be **executed by an agent** (any coding agent — no special tools
or MCPs required, only Bash + Python) while keeping the human's manual part
to a couple of clicks. If you are an agent: read
[SKILL.md](SKILL.md) first and follow its routing table.

## What you get

| Path | What | Human effort |
|---|---|---|
| **Notebook** ([colab/run-qwen-image-t4.ipynb](colab/run-qwen-image-t4.ipynb)) | open in Colab, pick T4, Run all → installs ComfyUI, downloads the model, generates a sample, prints a public API URL | ~2 min of clicking + ~12 min waiting |
| **Public text-to-image API** | `clients/txt2img.py` (Python stdlib only) from any machine against the printed URL | zero — the URL is the token |
| **Agent/CLI lane** | the same notebook run headlessly by an agent | one OAuth consent |
| **Tuning receipts** | the measured speed table and which knobs do/don't help | reading |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KodeIsFun/run-qwen-image-on-free-colab/blob/main/colab/run-qwen-image-t4.ipynb)

## The measured table (free Colab T4, Q4_K_M GGUF fully in VRAM)

| Config | s/step | Warm image |
|---|---|---|
| 1024², 20 steps, fp32 (ComfyUI's default cast on T4) | 38.0 | 765.6 s |
| **768², 12 steps, fp16** (this repo's settings) | **6.20** | **79.2 s** |

The two lines differ by **9.7×** — and neither knob is what people guess:

- The T4 has no bf16, so ComfyUI silently runs the model in **fp32** unless
  you force fp16 — but fp16 then crashes ComfyUI's model compiler on this
  card (`aimdo memory compile error`), so the flags are a pair:
  `--force-fp16 --disable-comfy-compiler`.
- "Use a smaller quant for speed" does nothing: the step is compute-bound.
  Quants decide *what fits in VRAM*, not how fast it runs.

## The images these settings produce

Both generated on a free Colab T4 with exactly this repo's combo —
768×768, 12 steps, cfg 2.5, Q4_K_M GGUF, fp16, seed 42 (raw outputs from the
measured run, untouched):

| Photoreal | Text rendering |
|---|---|
| ![A siamese cat in a tiny yellow raincoat on wet cobblestones at night, neon city bokeh](samples/v2b_fp16_p1_cat_raincoat.png) | ![A neon shop sign reading FREE GPU LAB, rainy night, reflections on wet pavement](samples/v2b_fp16_p2_freegpulab_sign.png) |
| "a siamese cat wearing a tiny yellow raincoat, sitting on wet cobblestones at night, neon city lights bokeh, 50mm lens" — 127.9 s incl. model loads | "a neon shop sign that reads \"FREE GPU LAB\", rainy night, reflections on wet pavement" — every word spelled correctly, 79.2 s warm |

That second one is the point: **text rendering survives quantization +
fp16 + 12 steps.**

Text rendering survives quantization + fp16 + 12 steps: the standard test
prompt produces a neon sign that reads **FREE GPU LAB**, spelled correctly.
Proof artifacts from the verification run ship in
[colab/run-qwen-image-t4_output.ipynb](colab/run-qwen-image-t4_output.ipynb)
and [colab/proof-served-through-tunnel.png](colab/proof-served-through-tunnel.png)
(generated on the VM, fetched through the public tunnel from a different
machine).

## Quickstart (human, browser only)

1. Click the badge above → sign in with Google.
2. *Runtime → Change runtime type → T4 GPU → Save.*
3. *Runtime → Run all* → approve → wait ~12 minutes.
4. Copy the `API URL: https://...trycloudflare.com` line, then from any
   machine:

   ```bash
   python3 clients/txt2img.py --url https://...trycloudflare.com \
     --prompt 'a neon shop sign that reads "FREE GPU LAB", rainy night' \
     --out my-first-image.png
   ```

Details, caveats (ephemeral URLs, idle VMs, license), and the agent-driven
path: [references/01-manual-steps.md](references/01-manual-steps.md).

## Repo map

```
SKILL.md                     ← agents start here (routing + 9 invariants)
colab/run-qwen-image-t4.ipynb        ← the notebook (built, verified on a fresh T4)
colab/run-qwen-image-t4_output.ipynb ← the executed notebook from the verification run
colab/proof-served-through-tunnel.png ← image fetched through the public tunnel
samples/                     ← the two raw outputs from the measured combo run (embedded above)
clients/txt2img.py           ← stdlib-only client, works from any machine
workflows/t2i-api.json       ← the exact verified ComfyUI workflow graph
references/01-manual-steps.md    ← the human's (tiny) part, verbatim scripts to say
references/02-colab-cli-lane.md  ← agent lane: install, auth, exec, artifacts
references/03-api-from-anywhere.md ← the API surface + client + curl
references/04-tweaking.md    ← measured table, knobs, what does NOT help
references/05-troubleshooting.md ← 12 entries, all paid for in real runs
tools/build_notebook.py      ← regenerates the notebook (edit here, not in the ipynb)
scripts/verify_env.py        ← sanity-check a machine before driving the lane
```

## Status

- [x] Notebook verified cell-by-cell on a fresh free T4 (2026-09-21)
- [x] API URL proven reachable from outside Colab; image served through the tunnel
- [x] `txt2img.py` tested end-to-end through the tunnel
- Untested, honestly labeled: cfg 1.0, Q8_0 speed, 1024²-fp16, Lightning LoRAs

Model: [abenzerps/Qwen-Image-2.1-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF)
(base: Qwen/Qwen-Image-2.1, **Qwen Research License**). Sibling repo for LLMs:
[run-llm-on-free-gpu](https://github.com/KodeIsFun/run-llm-on-free-gpu).
