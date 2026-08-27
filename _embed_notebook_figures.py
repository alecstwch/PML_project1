"""Inject saved PNGs into P1_notebook.ipynb cell outputs (nbclient-safe)."""

from __future__ import annotations

import base64
import os
import re

import nbformat

ROOT = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(ROOT, "P1_notebook.ipynb")
FIG_DIR = os.path.join(ROOT, "figures")


def _png_display(path: str):
    with open(path, "rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")
    return nbformat.v4.new_output(
        "display_data",
        data={
            "image/png": payload,
            "text/plain": ["<IPython.core.display.Image object>"],
        },
        metadata={"pml_figure": os.path.basename(path)},
    )


def _figure_names(source: str) -> list[str]:
    names: list[str] = []
    names.extend(re.findall(r'save_and_show\(\s*fig,\s*"([^"]+\.png)"', source))
    for name in re.findall(r'\bname\s*=\s*"([^"]+)"', source):
        names.extend(
            [
                f"svm_heatmap_{name}.png",
                f"cm_{name}.png",
                f"xgb_tune_{name}.png",
                f"cnn_search_{name}.png",
                f"curves_{name}.png",
            ]
        )
    for tag in re.findall(r'\.report\([^)]*?["\']([^"\']+)["\']\s*\)', source, flags=re.S):
        names.append(f"cm_{tag}.png")
        names.append(f"curves_{tag}.png")
    seen = set()
    out = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _strip_figure_repr(outputs: list) -> list:
    kept = []
    for item in outputs:
        if item.get("output_type") == "display_data":
            text = "".join(item.get("data", {}).get("text/plain", []))
            if text.startswith("<Figure size") and "image/png" not in item.get("data", {}):
                continue
        kept.append(item)
    return kept


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    n_added = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        paths = []
        for name in _figure_names(src):
            path = os.path.join(FIG_DIR, name)
            if os.path.isfile(path):
                paths.append(path)
        if not paths:
            continue
        outputs = list(cell.get("outputs", []))
        outputs = _strip_figure_repr(outputs)
        for path in paths:
            marker = os.path.basename(path)
            if any((item.get("metadata") or {}).get("pml_figure") == marker for item in outputs):
                continue
            disp = _png_display(path)
            outputs.append(disp)
            n_added += 1
        cell["outputs"] = outputs
    nbformat.write(nb, NB_PATH)
    print(f"Embedded {n_added} PNG outputs into {NB_PATH}")


if __name__ == "__main__":
    main()
