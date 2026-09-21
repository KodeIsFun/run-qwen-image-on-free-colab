#!/usr/bin/env python3
"""Sanity-check a machine that wants to drive the Colab CLI lane.

Run BEFORE starting a session:
    python3 scripts/verify_env.py

Checks (each prints OK/FAIL with the fix):
  1. colab binary present and importable
  2. auth works (a live `colab sessions` call)
  3. no sessions already running (idle VMs burn compute units)
  4. the notebook file parses and its code cells compile
  5. the client script compiles on this Python
"""

import ast
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
fails = 0


def check(name, ok, fix=""):
    global fails
    print(f"{'OK  ' if ok else 'FAIL'} {name}" + (f" — {fix}" if (fix and not ok) else ""))
    if not ok:
        fails += 1


# 1. binary
colab = shutil.which("colab")
check("colab binary on PATH", colab is not None,
      "uv tool install google-colab-cli  (or pipx install google-colab-cli)")

# 2. auth: a real read-only call
if colab:
    r = subprocess.run([colab, "sessions"], capture_output=True, text=True, timeout=120)
    check("colab auth works (sessions listing)", r.returncode == 0,
          "run: colab --auth=oauth2 sessions  -> approve the browser prompt once")
    # 3. no orphan sessions
    body = "\n".join(l for l in (r.stdout or "").splitlines()
                     if not l.strip().startswith("[colab]"))
    check("no running Colab sessions", not body.strip(),
          f"colab stop -s <name> each of:\n{body}")
else:
    fails += 1

# 4. notebook parses + code cells compile
nb_path = ROOT / "colab" / "run-qwen-image-t4.ipynb"
try:
    nb = json.loads(nb_path.read_text())
    check(f"notebook parses ({len(nb['cells'])} cells)", nb.get("nbformat") == 4)
    bad = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        py = "\n".join("    pass" if l.lstrip().startswith(("!", "%")) else l
                       for l in c["source"])
        try:
            ast.parse(py)
        except SyntaxError as e:
            bad.append((i, str(e)))
    check("all notebook code cells compile", not bad, str(bad))
except Exception as e:
    check("notebook parses", False, str(e))

# 5. client compiles here
client = ROOT / "clients" / "txt2img.py"
try:
    ast.parse(client.read_text())
    check("clients/txt2img.py compiles on this python", True)
except SyntaxError as e:
    check("clients/txt2img.py compiles", False, str(e))

print(f"\n{'ALL GREEN' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
