"""Tiny synthetic leaf-dataset fixtures for unit tests. Never use real BRACOL files."""

import os
import sys
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_leaf_dir(tmp_path):
    """CSV + a few fake JPEGs (green leaf on white, labelled 0-5)."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    csv_path = tmp_path / "dataset.csv"

    rng = np.random.default_rng(0)
    rows = []
    n_per_class = 8
    idx = 1
    for label in range(6):
        for k in range(n_per_class):
            img = np.full((64, 64, 3), 255, dtype=np.uint8)
            colour = {
                0: (40, 160, 40),
                1: (20, 80, 20),
                2: (30, 90, 180),
                3: (40, 70, 110),
                4: (50, 50, 50),
                5: (70, 100, 40),
            }[label]
            cv2.ellipse(img, (32, 32), (22, 18), 0, 0, 360, colour, -1)
            noise = rng.integers(0, 12, size=img.shape, dtype=np.uint8)
            img = cv2.subtract(img, noise)
            path = image_dir / f"{idx}.jpg"
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(path, quality=95)
            miner = int(label in (1, 5))
            rust = int(label in (2, 5))
            phoma = int(label in (3, 5))
            cerc = int(label in (4, 5))
            rows.append((idx, label, miner, rust, phoma, cerc, 0 if label == 0 else 2))
            idx += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "predominant_stress", "miner", "rust", "phoma", "cercospora", "severity"])
        w.writerows(rows)

    return {"csv": str(csv_path), "images": str(image_dir), "root": str(tmp_path)}
