#!/usr/bin/env python3
"""Consistency check: the turbo/baseline workflow graphs exist in THREE places
(workflows/*.json, the notebook's generate cell, clients/txt2img.py) and MUST
not drift apart. Run after any change:

    python3 tools/check_workflows.py

Graphs are compared node-by-node after ignoring the two fields that are
*supposed* to differ between copies (the sample prompt text and the
SaveImage filename prefix). Everything else — loaders, sizes, seed, sampler
wiring, sigma schedule — must be byte-identical.
"""

import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
IGNORED_INPUTS = {"filename_prefix", "text"}


def normalize(graph):
    out = {}
    for node_id, node in graph.items():
        inputs = {k: v for k, v in node["inputs"].items() if k not in IGNORED_INPUTS}
        out[node_id] = {"class_type": node["class_type"], "inputs": inputs}
    return out


def wf_from_notebook(nb_path, cell_marker):
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code" or cell_marker not in "".join(cell["source"]):
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "wf" for t in node.targets):
                return ast.literal_eval(node.value)
    raise SystemExit(f"no 'wf = ' assignment found in the {cell_marker!r} cell")


def wf_from_client(client_path, turbo):
    import importlib.util
    spec = importlib.util.spec_from_file_location("txt2img_client", client_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_workflow("PROMPT", steps=6 if turbo else 12,
                              cfg=1.0 if turbo else 2.5,
                              width=768, height=768, seed=42, turbo=turbo)


def main():
    fails = 0
    # turbo lives in three places (json + notebook cell + client);
    # the baseline is intentionally notebook-absent (fallback documented in
    # references/) so it is checked json vs client only.
    pairs = [
        ("turbo", ROOT / "workflows" / "t2i-turbo.json", "ViggleTurboLora", True),
        ("baseline", ROOT / "workflows" / "t2i-api.json", None, False),
    ]
    for name, json_path, nb_marker, turbo in pairs:
        canon = normalize({k: v for k, v in json.loads(json_path.read_text()).items()
                           if k != "_comment"})
        sources = [("client", wf_from_client(ROOT / "clients" / "txt2img.py", turbo))]
        if nb_marker is not None:
            sources.insert(0, ("notebook", wf_from_notebook(
                ROOT / "colab" / "run-qwen-image-t4.ipynb", nb_marker)))
        for label, other in sources:
            other = normalize(other)
            if canon == other:
                print(f"OK   {name}: json == {label}")
            else:
                fails += 1
                keys = set(canon) | set(other)
                for k in sorted(keys, key=str):
                    if canon.get(k) != other.get(k):
                        print(f"FAIL {name}: node {k} differs\n"
                              f"     json:    {canon.get(k)}\n"
                              f"     {label}: {other.get(k)}")
    # the turbo JSON must reference the same LoRA file the download cell fetches
    dl_src = (ROOT / "tools" / "build_notebook.py").read_text()
    lora = json.loads((ROOT / "workflows" / "t2i-turbo.json").read_text())["5"][
        "inputs"]["lora_name"]
    gguf = json.loads((ROOT / "workflows" / "t2i-turbo.json").read_text())["1"][
        "inputs"]["unet_name"]
    for fname, token in ((lora, "LORA"), (gguf, "GGUF")):
        if fname not in dl_src:
            fails += 1
            print(f"FAIL download cell: {token} file name {fname!r} not found in "
                  f"build_notebook.py CELL_DOWNLOAD")
        else:
            print(f"OK   download cell fetches the {token} the turbo graph loads "
                  f"({fname})")
    print(f"\n{'ALL GREEN' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
