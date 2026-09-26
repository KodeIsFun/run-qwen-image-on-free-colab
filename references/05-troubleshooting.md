# 05 — Troubleshooting

Every entry below was hit for real during the runs behind this repo (or its
sibling lab). Ordered by how likely you are to meet it.

## 1. `RuntimeError: aimdo memory compile error` (KSampler, under `--force-fp16`)

ComfyUI's model compiler (`comfy_aimdo`) cannot compile the fp16 forward on
Turing. The model itself loads fine (`weight dtype torch.float16` in the
log). **Fix: launch with both flags:**

```bash
python main.py --listen 127.0.0.1 --port 8188 --force-fp16 --disable-comfy-compiler
```

`--disable-comfy-compiler` disables the compiler/CUDA-graph subfeature;
without fp16 you do not need it (but it is harmless and keeps one canonical
command line). Without `--force-fp16` at all, the model silently runs an
fp32 manual cast at ~6× slower (38 vs 6.2 s/step) — check the log line
`weight dtype ..., manual cast: ...` to know which mode you got.

## 2. `Unknown model architecture!` when loading the GGUF

You cloned the **city96** fork of ComfyUI-GGUF. Qwen-Image-2.1 GGUFs are
sd.cpp-flavored (`[arch:qwen_image21]`). Use the **leejet** fork:

```bash
git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF \
  ComfyUI/custom_nodes/ComfyUI-GGUF
```

(Its `UnetLoaderGGUFAdvanced` has no `weight_dtype` input — server flags are
the only fp16 path. The plain `UnetLoaderGGUF` is what the notebook uses.)

## 3. "Will the 9.35 GB text encoder OOM my 12.7 GB RAM VM?"

No — measured peak 75–78 %. ComfyUI stages it ("Model ... prepared for
dynamic VRAM loading. 8916MB Staged"). Do not download smaller encoders or
add `--lowvram` preemptively; that is solved downtime.

## 4. Notebook prints `WARNING: no GPU`

The runtime is CPU (default on colabs that were not switched). Menu
*Runtime → Change runtime type → T4 GPU → Save*, then *Runtime → Run all*
again. `nvidia-smi` must show a Tesla T4. If Colab refuses with a quota
message, the free GPU allowance is exhausted — retry tomorrow or use Pro;
do not run image gen on CPU (hours per image).

## 5. Port already in use / server "already running" on re-run

The notebook's launch cell is idempotent: if `/system_stats` already
responds it skips the boot (this happens when re-running cells on a live
session). To force a fresh server: `Runtime → Restart runtime` or kill the
process listening on 8188 first.

## 6. The tunnel URL stopped working

Quick tunnels are ephemeral by design: the URL changes every notebook run,
and the VM sleeps/reclaims when idle or when Colab's session cap hits.
Re-run the notebook, take the NEW `API URL:` line. If `cloudflared` fails to
start, check the cell printed a `https://...trycloudflare.com` line at all —
a corporate/proxy network can block quick tunnels; retry usually clears it.

## 7. `colab` CLI: auth errors (401/403)

OAuth consent expired or missing: `colab --auth=oauth2 sessions` and approve
the browser prompt again. There is no `gcloud`/ADC step for this path; docs
that say otherwise are stale. `colab whoami` shows the active identity.

## 8. `colab run` executed my notebook as garbage / died on `true`

`colab run` is for **.py scripts** — on an .ipynb it runs raw JSON as Python.
Notebooks: `colab new -s name --gpu T4` → `colab exec -s name -f nb.ipynb
--timeout 1500` → `colab stop -s name`. All CLI `--timeout` values are
**seconds, default 30**.

## 9. Generation never finishes (history stays `{}`)

One image is ~80–130 s at combo settings. If `/history/<id>` is still empty
after ~25 min the queue is wedged — fetch the ComfyUI log tail (notebook
keeps it at `/content/comfyui.log`), then restart the runtime. Known wedges:
the §1 compiler crash (fixed by the flags) and GPU-quota mid-session
eviction (rare; re-run).

## 10. OOM at higher resolutions / bigger quants

Attempt it anyway — **OOM is a result, record it** — then step down: fp16
before fp32, 768² before 1024², Q4_K_M before Q8_0. VRAM arithmetic: file_GB
× 1024 + ~1500 ≤ free MB. The fp32 default-cast is the classic silent OOM
source at 1024²+ (peak 91 % measured).

## 11. Download stalls / truncates weights

The notebook uses `wget -c --tries=3` and checks the target size before
skipping. If a file < 100 MB exists where a multi-GB file should be, delete
it and re-run the download cell. Hugging Face serves these repos via CDN;
~100 MB/s is normal on Colab (~15.6 GB total ≈ 3 min).

## 12. `FileNotFoundError: /content/ComfyUI` in an agent script

The script assumed a previous session's state. Every job must be
**fresh-VM safe**: clone/install/download with `if not exists` skips (see
the notebook's cells 2–3 for the pattern).

## 13. `download failed: .../qwen-image-2.1-Q4_K_M.gguf` (GitHub issue #1)

The notebook asserts every download is > 100 MB; this assertion firing on
the diffusion GGUF means Hugging Face returned a 404 body. Cause: abenzerps
restructured `Qwen-Image-2.1-GGUF` into `Qwen-Image-2.1-Uncensored-GGUF` and
dropped the base diffusion file. **Fixed 2026-09-26**: the notebook (and
every doc here) now pulls the GGUF from
[unsloth/Qwen-Image-2.1-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)
— the base model, which is also the distribution the turbo LoRA was
distilled on. TE + VAE still resolve at abenzerps' paths. If you are on an
old checkout: `git pull`, delete the partial file (`rm
/content/ComfyUI/models/diffusion_models/qwen-image-2.1-Q4_K_M.gguf`), and
re-run the download cell. `wget -c` cannot resume from a 404 husk — the husk
must go.

## 14. `ViggleTurboLora`/`ViggleTurboSigmas` missing from `/object_info`, or a turbo generation errors

The turbo graph needs Viggle's custom node: the notebook's cell 2 fetches
`viggle_turbo.py` from the
[Viggle repo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)
into `custom_nodes/`. Custom nodes load at **server boot** — after adding
the file, restart the runtime (or kill the 8188 process) and re-run from
cell 4. If the node loads but a turbo generation errors (the author calls
the ComfyUI port "vibe-coded", verified end-to-end only against diffusers),
fall back to the **baseline graph**: `txt2img.py --no-turbo` (12 steps, cfg
2.5, ~69 s warm). A stock `LoraLoaderModelOnly` merge is a last resort for
speed — quality caveat: the author measures ~70 % of the LoRA update lost on
bf16 merges, worse on quantized weights, and stock samplers cannot express
the turbo sigma schedule (approximation only).
