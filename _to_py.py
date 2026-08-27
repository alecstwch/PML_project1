"""Convert P1_notebook.ipynb to P1_notebook.py (no Jupyter magics)."""

import json
from pathlib import Path

nb = json.loads(Path("P1_notebook.ipynb").read_text(encoding="utf-8"))
out = ["# Converted from P1_notebook.ipynb. Keep p1_core.py in the same folder.", ""]
for i, cell in enumerate(nb["cells"]):
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    src = src.strip("\n")
    if cell["cell_type"] == "markdown":
        out.append('"""')
        out.append(src.replace('"""', "'''"))
        out.append('"""')
        out.append("")
    else:
        out.append(f"# --- cell {i} ---")
        out.append(src)
        out.append("")
Path("P1_notebook.py").write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote P1_notebook.py")
