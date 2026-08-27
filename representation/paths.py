"""Project folders and extra JPEG recovery."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence

import pandas as pd


def resolve_paths(base_dir: Optional[str] = None) -> Dict[str, str]:
    """Return data and output folders next to this project (Windows or WSL)."""
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
    data_dir = os.path.join(base_dir, "..", "coffee-datasets", "coffee-datasets", "leaf")
    paths = {
        "base_dir": os.path.abspath(base_dir),
        "data_dir": os.path.abspath(data_dir),
        "image_dir": os.path.abspath(os.path.join(data_dir, "images")),
        "csv_path": os.path.abspath(os.path.join(data_dir, "dataset.csv")),
        "bracol_dir": os.path.abspath(
            os.path.join(base_dir, "..", "BRACOL_REVIEWED_ANNOTATIONS", "BRACOL_REVIEWED")
        ),
        "ninja_dir": os.path.abspath(os.path.join(base_dir, "..", "dataset-ninja")),
        "figures_dir": os.path.abspath(os.path.join(base_dir, "figures")),
        "features_dir": os.path.abspath(os.path.join(base_dir, "features")),
        "results_dir": os.path.abspath(os.path.join(base_dir, "results")),
        "models_dir": os.path.abspath(os.path.join(base_dir, "models")),
    }
    for key in ("figures_dir", "features_dir", "results_dir", "models_dir"):
        os.makedirs(paths[key], exist_ok=True)
    return paths


def _leaf_id_from_filename(name: str) -> Optional[int]:
    """Parse a CSV leaf id from `{id}.jpg` or a Roboflow `{id}_jpg.rf.*.jpg` name."""
    stem, ext = os.path.splitext(name)
    if ext.lower() not in {".jpg", ".jpeg", ".png"}:
        return None
    if stem.isdigit():
        return int(stem)
    match = re.match(r"^(\d+)_jpg\.rf\.", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def copy_missing_leaf_jpgs(csv_path: str, image_dir: str, extra_roots: Sequence[str]) -> int:
    """Copy `{id}.jpg` from extra folders into image_dir when the CSV row has no file yet.

    Accepts plain `{id}.jpg` (Dataset Ninja / Supervisely) and Roboflow exports
    named `{id}_jpg.rf.<hash>.jpg`. Does not overwrite a file that is already present.
    """
    import shutil

    df = pd.read_csv(csv_path)
    wanted = {int(i) for i in df["id"].tolist()}
    os.makedirs(image_dir, exist_ok=True)
    copied = 0
    for root in extra_roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for name in files:
                img_id = _leaf_id_from_filename(name)
                if img_id is None or img_id not in wanted:
                    continue
                dest = os.path.join(image_dir, f"{img_id}.jpg")
                if os.path.exists(dest):
                    continue
                src = os.path.join(dirpath, name)
                shutil.copy2(src, dest)
                copied += 1
    return copied


def extra_image_roots(paths: Dict[str, str]) -> List[str]:
    """Folders that may hold extra `{id}.jpg` files (Dataset Ninja / YOLO exports)."""
    base = paths["base_dir"]
    roots = [
        paths.get("ninja_dir", ""),
        os.path.expanduser("~/dataset-ninja"),
        os.path.abspath(os.path.join(base, "..", "BRACOL_REVIEWED_ANNOTATIONS")),
        os.path.abspath(os.path.join(base, "..", "BRACOL-ORIGINAL-ANNOTATIONS")),
    ]
    seen = set()
    out = []
    for root in roots:
        if not root:
            continue
        abs_root = os.path.abspath(root)
        if abs_root in seen:
            continue
        seen.add(abs_root)
        out.append(abs_root)
    return out
