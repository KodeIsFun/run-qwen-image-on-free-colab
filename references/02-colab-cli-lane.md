# 02 — Colab CLI lane (agent-driven)

Install once, auth once (see [01-manual-steps.md](01-manual-steps.md) Path B),
then everything here is headless. Binary is `colab` (package
`google-colab-cli`).

## Install

```bash
uv tool install google-colab-cli        # or: pipx install google-colab-cli
~/.local/bin/colab --version            # 0.6.0 measured
```

## Auth (needs the human once)

```bash
colab --auth=oauth2 sessions            # browser consent on first use
```

Stale docs mention `gcloud` / ADC — ignore them for this path. OAuth2 is the
flow that works; the token lands in `~/.config/colab-cli/token.json`.

## Run the notebook headlessly

`colab run` executes a **.py script** on a fresh VM and self-cleans.
Notebooks go through `colab new` + `colab exec -f` (exec's `--timeout` is
**seconds**, default 30 — always set it):

```bash
colab new --gpu T4 -s qwenimg
colab exec -s qwenimg -f colab/run-qwen-image-t4.ipynb --timeout 1500
colab stop -s qwenimg
colab sessions                          # must be empty
```

- `exec` streams cell output live — grep for the notebook's success markers:
  `OK: gpu`, `OK: install`, `OK: download`, `OK: launch`, `OK: generate`,
  `OK: tunnel`, and the `API URL:` line.
- If `colab new` 400s on the accelerator, the free tier has no T4 entitlement
  *right now*: retry later. An unrecognized `--gpu` value silently falls back
  to A100 (which then usually fails) — use `T4`.

## Long jobs: the detached-launch pattern

`colab exec` cells queue behind a running cell, so a second exec **while the
notebook runs** hangs until it finishes. For jobs longer than a couple of
minutes, run the payload as a background process and poll the log instead:

```bash
# 1. upload + launch detached (returns immediately)
colab upload -s qwenimg job.py /content/job.py
colab exec -s qwenimg <<'EOF'
import subprocess, sys
subprocess.Popen([sys.executable, "/content/job.py"],
                 stdout=open("/content/job.log", "w"),
                 stderr=subprocess.STDOUT,
                 start_new_session=True, cwd="/content")
print("launched")
EOF

# 2. poll every ~2 min
colab exec -s qwenimg <<'EOF'
t = open("/content/job.log").read()
print(t[-1500:])
print("DONE" if "=== DONE" in t else "RUNNING")
EOF
```

`start_new_session=True` is what makes the job survive kernel restarts.

## Artifacts: download BEFORE stop, verify every one

`colab stop` is terminal — the VM and `/content` die together. The full
artifact set of a run was once lost to a download loop that failed silently
while the stop ran anyway. The protocol:

```bash
colab download -s qwenimg /content/out/image.png ./image.png
ls -la ./image.png          # verify size > 0 — every single file
# ... only when ALL artifacts are verified locally:
colab stop -s qwenimg
colab sessions              # must be empty
```

Never bundle downloads and the stop into one command; never grep away a
download's error.

## Handout scripts: fresh-VM safe or broken

Any script uploaded to a VM must carry its own setup (clone/pip/download
with `if not os.path.exists(...)` skips). A script that assumes a previous
session's `/content` dies with `FileNotFoundError` on a fresh VM.

## Known breakage (fixed once, will recur on upgrades)

A `uv` upgrade once pulled `jupyter_kernel_client` 1.0.2, which renamed
`KernelClient` and crashed `colab` before allocating any VM:

```bash
uv pip install --python ~/.local/share/uv/tools/google-colab-cli/bin/python \
  'jupyter_kernel_client<1'
```

If `colab` dies with an AttributeError right after an upgrade, check this pin
first.
