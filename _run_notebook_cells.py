"""Execute P1_notebook.ipynb cell by cell on the current kernel, saving after each cell."""

from __future__ import annotations

import os
import sys
import time
import traceback

# Agg has no window. Figures are embedded by save_and_show() via IPython.display.Image.
os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat
from nbclient.client import NotebookClient

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "P1_notebook.ipynb")
START_AT = int(os.environ.get("START_AT", "0"))
# Comma-separated notebook cell indices to run, e.g. ONLY_CELLS=2,4,12,14,21
ONLY_CELLS = {
    int(x) for x in os.environ.get("ONLY_CELLS", "").replace(" ", "").split(",") if x
}


def main() -> int:
    os.chdir(os.path.dirname(NB_PATH))
    nb = nbformat.read(NB_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name=os.environ.get("KERNEL_NAME", "pml-wsl-gpu"),
        resources={"metadata": {"path": os.path.dirname(NB_PATH)}},
    )
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    code_i = 0
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if ONLY_CELLS and i not in ONLY_CELLS:
                if cell.cell_type == "code":
                    code_i += 1
                print(f"[{i:02d}] skip (not in ONLY_CELLS)", flush=True)
                continue
            if i < START_AT:
                if cell.cell_type == "code":
                    code_i += 1
                continue
            label = cell.get("id", f"cell{i}")
            if cell.cell_type != "code":
                print(f"[{i:02d}] skip markdown {label}", flush=True)
                continue
            code_i += 1
            src = "".join(cell.source).strip().splitlines()
            preview = src[0][:80] if src else "(empty)"
            print(f"[{i:02d}] code {code_i}/{n_code} {label}: {preview}", flush=True)
            t0 = time.time()
            try:
                cell.outputs = []
                cell.execution_count = code_i
                client.execute_cell(cell, i, execution_count=code_i)
                dt = time.time() - t0
                print(f"     done in {dt:.1f}s", flush=True)
            except Exception as exc:
                dt = time.time() - t0
                print(f"     FAILED after {dt:.1f}s: {exc}", flush=True)
                traceback.print_exc()
                nbformat.write(nb, NB_PATH)
                return 1
            nbformat.write(nb, NB_PATH)
    print("All cells finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
