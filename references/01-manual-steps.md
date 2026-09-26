# 01 — Manual steps (what the human does by hand)

Everything automatable is automated. The human's part is deliberately tiny,
and there are exactly two paths. As the agent, read the path that applies and
**say these sentences to the user** — do not assume they know the Colab UI.

## Path A — browser only (no tools, no CLI, works for anyone)

Total human effort: ~2 minutes of clicking, then ~10 minutes of waiting.

1. **Have a Google account.** That is the only prerequisite. Free tier gives
   a T4 GPU with usage limits that reset daily; a normal first run fits
   easily.
2. **Open the notebook in Colab** — click the badge in the README
   ("Open in Colab"), which is this URL:

   ```
   https://colab.research.google.com/github/KodeIsFun/run-qwen-image-on-free-colab/blob/main/colab/run-qwen-image-t4.ipynb
   ```

   (If the badge is ever broken, any Colab session can open it via
   *File → Open notebook → GitHub* tab, pasting
   `KodeIsFun/run-qwen-image-on-free-colab`.)
3. **Pick the GPU**: menu *Runtime → Change runtime type → T4 GPU → Save*.
   The notebook's first cell checks this and prints a warning if the runtime
   has no GPU — if it warns, redo this step.
4. **Run all**: menu *Runtime → Run all*, then approve the "not authored by
   Google" warning. Cells print `OK: ...` as they finish.
5. **Wait ~10 minutes.** Install ≈ 1 min, downloads ≈ 3–4 min (15.6 GB),
   model boot ≈ 2 min, first image ≈ 2 min (model loads; warm images after
   it run ~21 s), tunnel ≈ 15 s. The notebook shows progress.
6. **Copy the line that looks like**

   ```
   API URL: https://something-random-words.trycloudflare.com
   ```

   That is the public text-to-image API. Generate from anywhere:

   ```bash
   python3 clients/txt2img.py \
     --url https://something-random-words.trycloudflare.com \
     --prompt "a neon shop sign that reads \"FREE GPU LAB\", rainy night" \
     --out my-image.png
   ```

   (Or just use the notebook itself as a generator — it displays the sample
   image in-cell. The API is for when you want to generate from another
   machine or an app.)

### Tell the user these caveats up front

- The tunnel URL **changes every run** — re-run the notebook to get a fresh
  one. This is inherent to free tunnels, not a bug.
- The free VM **sleeps when idle** and is reclaimed eventually (Colab caps
  sessions). Anything long-lived needs the re-run, or a paid tier, or a
  Kaggle/persistent box. The notebook is designed so a re-run costs one click
  and ~10 minutes.
- The model is **Qwen Research License** — fine for experiments, research,
  and demos; check the license before a commercial product.

## Path B — the agent drives (one OAuth consent)

1. **Human installs the CLI** (or lets the agent do it):

   ```bash
   uv tool install google-colab-cli     # or: pipx install google-colab-cli
   ```

2. **Human approves OAuth once.** The agent runs `colab --auth=oauth2
   sessions`; a browser window opens; the human clicks through with their
   Google account. One time — the refresh token is cached in
   `~/.config/colab-cli/token.json`. Tell the human verbatim:

   > "A browser window will open asking you to sign in to Google and approve
   > the Colab CLI. Click Allow. Tell me when done."

   Nobody should ever need `gcloud` or `kaggle.json` for this path. If a
   doc says otherwise, it is stale.
3. **Everything else is the agent's job** — see
   [02-colab-cli-lane.md](02-colab-cli-lane.md). The human never needs to
   look at Colab again.

## What the human should NOT need to do (agent: do not ask them)

- No `kaggle.json`, no Hugging Face token (the model files are ungated), no
  Cloudflare/ngrok account (the tunnel is accountless), no ngrok authtoken,
  no CUDA/driver installs, no Docker, no GPU selection beyond the one menu
  click in Path A.
- No payment. If Colab says the free GPU quota is exhausted, the honest move
  is "retry tomorrow or use Colab Pro", not silently burning someone's card.
