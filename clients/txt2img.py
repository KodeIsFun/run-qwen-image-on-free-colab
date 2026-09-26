#!/usr/bin/env python3
"""txt2img — generate an image on a remote ComfyUI (e.g. the free-T4 notebook
tunnel) using only the Python standard library. No pip installs, any OS.

Tested end-to-end from outside Colab through a trycloudflare tunnel.

Defaults to the measured turbo graph (Viggle LoRA applied server-side, 6
steps, CFG off, ~21 s warm at 768²). --no-turbo switches to the 12-step
no-LoRA baseline (~69 s warm) — useful if the server lacks the turbo custom
node, or for an apples-to-apples quality comparison.

Examples:
    python3 txt2img.py --url https://x-y-z.trycloudflare.com --prompt "a red fox in snow"
    python3 txt2img.py --url $URL --prompt "..." --no-turbo --out fox-baseline.png
    python3 txt2img.py --url $URL --prompt "..." --steps 8 --width 1024 --height 1024 --seed 7 --out fox.png
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

GGUF_FILE = "qwen-image-2.1-Q4_K_M.gguf"
TE_FILE = "qwen3vl_8b_int8_convrot.safetensors"
VAE_FILE = "qwen_image_2.1_vae_bf16.safetensors"
LORA_FILE = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
SIGMAS_6 = "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"
TIMEOUT_SUBMIT = 60
TIMEOUT_IMAGE = 1800  # generous: includes queue + cold model load (~5 min)


def build_workflow(prompt, steps, cfg, width, height, seed, turbo):
    """The exact graph the notebook runs.

    turbo=True: ViggleTurboLora (runtime hooks — never merged) + cfg fully
    off via BasicGuider + the resolution-shifted sigma schedule
    (workflows/t2i-turbo.json). turbo=False: the 12-step KSampler baseline
    (workflows/t2i-api.json).
    """
    wf = {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": GGUF_FILE}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TE_FILE, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": VAE_FILE}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
    }
    if turbo:
        wf["5"] = {"class_type": "ViggleTurboLora",
                   "inputs": {"model": ["1", 0], "lora_name": LORA_FILE,
                              "strength": 1.0}}
        wf["7"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
        wf["8"] = {"class_type": "KSamplerSelect",
                   "inputs": {"sampler_name": "euler"}}
        wf["9"] = {"class_type": "ViggleTurboSigmas",
                   "inputs": {"latent": ["6", 0], "nodes": SIGMAS_6}}
        wf["10"] = {"class_type": "BasicGuider",
                    "inputs": {"model": ["5", 0], "conditioning": ["4", 0]}}
        wf["11"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["7", 0], "guider": ["10", 0],
                               "sampler": ["8", 0], "sigmas": ["9", 0],
                               "latent_image": ["6", 0]}}
        wf["12"] = {"class_type": "VAEDecode",
                    "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}
        wf["13"] = {"class_type": "SaveImage",
                    "inputs": {"images": ["12", 0], "filename_prefix": "txt2img"}}
    else:
        wf["5"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"clip": ["2", 0], "text": ""}}
        wf["7"] = {"class_type": "KSampler",
                   "inputs": {"model": ["1", 0], "positive": ["4", 0],
                              "negative": ["5", 0], "latent_image": ["6", 0],
                              "seed": seed, "steps": steps, "cfg": cfg,
                              "sampler_name": "res_multistep",
                              "scheduler": "simple", "denoise": 1.0}}
        wf["8"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
        wf["9"] = {"class_type": "SaveImage",
                   "inputs": {"images": ["8", 0], "filename_prefix": "txt2img"}}
    return wf


def api(url, path, payload=None, timeout=60):
    req = urllib.request.Request(url.rstrip("/") + path, method="GET")
    if payload is not None:
        req = urllib.request.Request(url.rstrip("/") + path,
                                     data=json.dumps(payload).encode(),
                                     method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def api_retry(url, path, payload=None, timeout=60, tries=4):
    """api() with retries — quick tunnels occasionally reset a connection;
    one flaky reset must not kill a client that is willing to wait 25 min."""
    last = None
    for attempt in range(tries):
        try:
            return api(url, path, payload, timeout)
        except Exception as e:  # noqa: BLE001 — anything transient counts
            last = e
            time.sleep(min(2 ** attempt, 8))
    raise last


def main():
    p = argparse.ArgumentParser(description="txt2img on a remote ComfyUI")
    p.add_argument("--url", required=True, help="e.g. https://x-y.trycloudflare.com")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="",
                   help="ignored with the turbo graph (CFG is fully off there)")
    p.add_argument("--no-turbo", action="store_true",
                   help="12-step no-LoRA baseline graph instead of turbo")
    p.add_argument("--steps", type=int, default=None,
                   help="default: 6 (turbo, via its sigma schedule) / 12 (baseline)")
    p.add_argument("--cfg", type=float, default=None,
                   help="default: fixed by the graph (turbo has CFG off)")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="generated.png")
    a = p.parse_args()

    turbo = not a.no_turbo
    steps = a.steps if a.steps is not None else (6 if turbo else 12)
    cfg = a.cfg if a.cfg is not None else (1.0 if turbo else 2.5)

    url = a.url
    try:
        api_retry(url, "/system_stats", timeout=15)
        print(f"server alive: {url}")
    except Exception as e:
        sys.exit(f"server not reachable at {url} — is the notebook running? ({e})")

    wf = build_workflow(a.prompt, steps, cfg, a.width, a.height, a.seed, turbo)
    pid = api_retry(url, "/prompt", {"prompt": wf, "client_id": "txt2img.py"},
                    timeout=TIMEOUT_SUBMIT)["prompt_id"]
    mode = "turbo (6 steps, cfg off)" if turbo else f"baseline ({steps} steps, cfg {cfg})"
    print(f"queued {a.width}x{a.height}, {mode}, seed {a.seed} — polling...")

    t0 = time.time()
    while time.time() - t0 < TIMEOUT_IMAGE:
        time.sleep(3)
        try:
            hist = api_retry(url, f"/history/{pid}", timeout=20)
        except Exception:
            continue  # transient tunnel reset — the loop timeout bounds this
        if pid not in hist:
            continue
        entry = hist[pid]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            sys.exit(f"generation failed: {json.dumps(status.get('messages', []))[:400]}")
        imgs = [im for out in entry.get("outputs", {}).values() for im in out.get("images", [])]
        if imgs:
            im = imgs[0]
            q = urllib.parse.urlencode({"filename": im["filename"],
                                        "subfolder": im.get("subfolder", ""),
                                        "type": im.get("type", "output")})
            data = None
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(
                            f"{url.rstrip('/')}/view?{q}", timeout=120) as r:
                        data = r.read()
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(min(2 ** attempt, 8))
            with open(a.out, "wb") as f:
                f.write(data)
            print(f"done in {time.time() - t0:.0f}s -> {a.out}")
            return
    sys.exit("timed out waiting for the image (~25 min) — see troubleshooting §9")


if __name__ == "__main__":
    main()
