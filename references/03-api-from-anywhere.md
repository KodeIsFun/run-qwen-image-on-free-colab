# 03 — The API, consumable from anywhere

ComfyUI runs headless on the VM with its HTTP API on `127.0.0.1:8188`. The
notebook tunnels it through a **Cloudflare quick tunnel** (no account, no
token) and prints:

```
API URL: https://<random-words>.trycloudflare.com
```

The API surface used here is three endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/prompt` | POST | queue a workflow JSON; returns `{"prompt_id": ...}` |
| `/history/<prompt_id>` | GET | `{}` while running; on completion has `outputs` with image filenames and `status.status_str` = `success` or `error` |
| `/view?filename=X&subfolder=&type=output` | GET | the generated PNG bytes |

Generation is **asynchronous by design**: submit, poll, fetch.

## The easy way: the bundled client (stdlib only, any machine)

```bash
python3 clients/txt2img.py \
  --url https://<random-words>.trycloudflare.com \
  --prompt 'a neon shop sign that reads "FREE GPU LAB", rainy night' \
  --out my-image.png
```

Options: `--steps 12 --cfg 2.5 --width 768 --height 768 --seed 42
--negative ""` (defaults = the measured combo). Requires only Python 3 —
no pip installs. This client was tested end-to-end from a machine outside
Colab through the tunnel; the proof PNG ships in this repo at
`colab/proof-served-through-tunnel.png`.

## The curl way

```bash
# 1. submit (workflow JSON: see workflows/t2i-api.json for the full shape)
curl -s -X POST $URL/prompt -H 'Content-Type: application/json' -d '{
  "prompt": {...}, "client_id": "curl"}'          # -> {"prompt_id": "..."}

# 2. poll until outputs appear (prompt_id from step 1)
curl -s $URL/history/<prompt_id>

# 3. fetch the image named in outputs."9".images[0].filename
curl -s "$URL/view?filename=<name>&subfolder=&type=output" -o out.png
```

## Raw workflow JSON

`workflows/t2i-api.json` is the exact verified graph (Q4_K_M GGUF loader,
`qwen_image` CLIP loader, VAE, 768² latent, KSampler res_multistep/simple,
VAE decode, SaveImage). POST it wrapped as `{"prompt": <graph>,
"client_id": "anything"}`. Node IDs are strings; links are `["<node>", <output_index>]`.

## Operational truths

- **The URL is ephemeral.** Every notebook re-run prints a new one; Colab
  reclaims idle VMs. Design clients to take the URL as a parameter, never
  hard-coded. (A persistent URL needs a persistent host — out of scope for
  the free tier.)
- **One generation at a time** is the sane default on the T4; ComfyUI queues
  additional `/prompt` submissions rather than rejecting them, so a client
  can fire-and-poll multiple prompts safely (they just run serially).
- **Timeouts**: a combo-settings image takes ~80–130 s end to end (queue +
  text encode + 12 steps + VAE). Poll `/history` every ~3 s; give up after
  ~25 min (that is a stuck queue, not slowness — see troubleshooting §9).
- **Health check**: `GET /system_stats` → 200 with JSON means the server is
  alive. (There is no `/health`.)
- **Do not ship the tunnel URL anywhere public.** Anyone with it can queue
  jobs on the user's free VM.
