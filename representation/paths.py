"""Project folders (Windows or WSL)."""

from __future__ import annotations

import os
from typing import Dict, Optional


def resolve_paths(base_dir: Optional[str] = None, data_dir: Optional[str] = None) -> Dict[str, str]:
    """Return data and output folders next to this project."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(pkg_dir)
    if base_dir is None:
        cwd = os.getcwd()
        if os.path.basename(cwd) == "Project_1":
            base_dir = cwd
        elif os.path.basename(project_root) == "Project_1":
            base_dir = project_root
        else:
            base_dir = os.path.join(cwd, "Project_1")
    if data_dir is None:
        data_dir = os.path.join(base_dir, "..", "coffee-datasets", "coffee-datasets", "leaf")
    paths = {
        "base_dir": os.path.abspath(base_dir),
        "data_dir": os.path.abspath(data_dir),
        "image_dir": os.path.abspath(os.path.join(data_dir, "images")),
        "csv_path": os.path.abspath(os.path.join(data_dir, "dataset.csv")),
        "bracol_dir": os.path.abspath(
            os.path.join(base_dir, "..", "BRACOL_REVIEWED_ANNOTATIONS", "BRACOL_REVIEWED")
        ),
        "figures_dir": os.path.abspath(os.path.join(base_dir, "figures")),
        "features_dir": os.path.abspath(os.path.join(base_dir, "features")),
        "results_dir": os.path.abspath(os.path.join(base_dir, "results")),
        "models_dir": os.path.abspath(os.path.join(base_dir, "models")),
    }
    for key in ("figures_dir", "features_dir", "results_dir", "models_dir"):
        os.makedirs(paths[key], exist_ok=True)
    return paths
