# PML Project 1 — coffee leaf biotic stress

Supervised classification of BRACOL leaf photos by `predominant_stress`. Official models: **SVM-RBF** and **EfficientNet-B0**. Extras: ResNet-50, XGBoost, ensemble, residual Mixed rule.

GitHub: [https://github.com/alecstwch/PML_project1](https://github.com/alecstwch/PML_project1)

## Dataset (not in this repo)

Place the BRACOL leaf CSV and JPEGs next to this project:

```
PML/
  Project_1/          ← this repo
  coffee-datasets/coffee-datasets/leaf/
    dataset.csv
    images/{id}.jpg
```

Override with `--data-dir` if the leaf folder is elsewhere. **Do not commit or zip the JPEGs or CSV.**

## Run with the CLI (what the teacher can execute)

The course asks for a `.py` file, not a notebook. From `Project_1/`:

```bash
python P1_run.py --help
python P1_run.py                          # 6-class official pair: SVM-RBF + EfficientNet
python P1_run.py --models svm
python P1_run.py --drop-mixed --models resnet
python P1_run.py --models all --skip-xgb
python P1_run.py --data-dir /path/to/leaf
```

Default is the **6-class** official pair. `--drop-mixed` is the Esgario-style **5-class** protocol (drop Mixed, 1,685 images). Writes under `results/` (`cli_results.csv` for 6-class CLI, `final_results_nomix.csv` for 5-class).

Use the **WSL GPU** interpreter for CNNs (`/home/alecs/pml_venv/bin/python`, kernel name `pml-wsl-gpu`). The Windows `.venv` is TensorFlow CPU-only.

## Notebooks

| File | Protocol |
|---|---|
| `P1_notebook.ipynb` | Official **6-class** run (Mixed kept, 1,747 images) |
| `P1_no_mixed_notebook.ipynb` | Esgario-style **5-class** experiment (`drop_mixed=True`) |
| `P1_notebook.py` | Linear 6-class script (same flow as the official notebook) |

Re-run a notebook cell-by-cell on WSL:

```bash
KERNEL_NAME=pml-wsl-gpu python _run_notebook_cells.py
NB_PATH=P1_no_mixed_notebook.ipynb KERNEL_NAME=pml-wsl-gpu python _run_notebook_cells.py
```

## Tests

```bash
python -m pytest tests -q
```

## GitHub vs course zip

This GitHub repo may keep `representation/`, `evaluation/`, and `optimization/` with `p1_core.py` as a facade. The course zip does **not** allow subfolders of code: flatten those packages into `p1_core.py` later, put `P1_run.py` / `P1_notebook.py` next to it, and ship the PDF under `P1_{family}_{first}_{group}_doc/`. Never zip leaf JPEGs, `dataset.csv`, `models/*.keras`, or `features/*.npz`.
