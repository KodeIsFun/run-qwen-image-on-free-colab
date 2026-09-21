#!/usr/bin/env python3
"""txt2img — generate an image on a remote ComfyUI (e.g. the free-T4 notebook
tunnel) using only the Python standard library. No pip installs, any OS.

Tested end-to-end from outside Colab through a trycloudflare tunnel.

Examples:
    python3 txt2img.py --url https://x-y-z.trycloudflare.com --prompt "a red fox in snow"
    python3 txt2img.py --url $URL --prompt "..." --steps 16 --width 1024 --height 1024 --seed 7 --out fox.png
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

# Defaults = the measured combo for a free T4 (see references/04-tweaking.md)
DEFAULTS = dict(steps=12, cfg=2.5, width=768, height=768, seed=42)
GGUF_FILE = "qwen-image-2.1-Q4_K_M.gguf"
TE_FILE = "qwen3vl_8b_int8_convrot.safetensors"
VAE_FILE = "qwen_image_2.1_vae_bf16.safetensors"
TIMEOUT_SUBMIT = 60
TIMEOUT_IMAGE = 1800  # generous: includes queue + cold model load (~5 min)


def build_workflow(prompt, negative, steps, cfg, width, height, seed):
    """The exact graph the notebook runs (workflows/t2i-api.json)."""
    return {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": GGUF_FILE}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TE_FILE, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": VAE_FILE}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": negative}},
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["6", 0],
                         "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "res_multistep", "scheduler": "simple",
                         "denoise": 1.0}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "txt2img"}},
    }


def api(url, path, payload=None, timeout=60):
    req = urllib.request.Request(url.rstrip("/") + path, method="GET")
    if payload is not None:
        req = urllib.request.Request(url.rstrip("/") + path,
                                     data=json.dumps(payload).encode(),
                                     method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    p = argparse.ArgumentParser(description="txt2img on a remote ComfyUI")
    p.add_argument("--url", required=True, help="e.g. https://x-y.trycloudflare.com")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="")
    p.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    p.add_argument("--cfg", type=float, default=DEFAULTS["cfg"])
    p.add_argument("--width", type=int, default=DEFAULTS["width"])
    p.add_argument("--height", type=int, default=DEFAULTS["height"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--out", default="generated.png")
    a = p.parse_args()

    url = a.url
    try:
        api(url, "/system_stats", timeout=15)
        print(f"server alive: {url}")
    except Exception as e:
        sys.exit(f"server not reachable at {url} — is the notebook running? ({e})")

    wf = build_workflow(a.prompt, a.negative, a.steps, a.cfg, a.width, a.height, a.seed)
    pid = api(url, "/prompt", {"prompt": wf, "client_id": "txt2img.py"},
              timeout=TIMEOUT_SUBMIT)["prompt_id"]
    print(f"queued {a.width}x{a.height}, {a.steps} steps, seed {a.seed} — polling...")

    t0 = time.time()
    while time.time() - t0 < TIMEOUT_IMAGE:
        time.sleep(3)
        hist = api(url, f"/history/{pid}", timeout=20)
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
            with urllib.request.urlopen(f"{url.rstrip('/')}/view?{q}",
                                        timeout=120) as r, open(a.out, "wb") as f:
                f.write(r.read())
            print(f"done in {time.time() - t0:.0f}s -> {a.out}")
            return
    sys.exit("timed out waiting for the image (~25 min) — see troubleshooting §9")


if __name__ == "__main__":
    main()
