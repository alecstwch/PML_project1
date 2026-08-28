"""CSV + JPEG loading and the shared 70/15/15 split."""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from representation.constants import RANDOM_STATE


class LeafDataset:
    """Load the leaf CSV. Every `{id}.jpg` is assumed to exist and decode."""

    def __init__(self, csv_path: str, image_dir: str, drop_mixed: bool = False):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.drop_mixed = drop_mixed
        self.frame: Optional[pd.DataFrame] = None
        self.n_csv: int = 0

    def load_and_match(self) -> pd.DataFrame:
        """Load the CSV, attach image paths, optionally drop Mixed (class 5)."""
        df = pd.read_csv(self.csv_path)
        self.n_csv = len(df)
        df["image_path"] = df["id"].apply(lambda x: os.path.join(self.image_dir, f"{x}.jpg"))
        if self.drop_mixed:
            df = df[df["predominant_stress"] != 5].reset_index(drop=True)
        self.frame = df
        return df

    def class_counts(self) -> pd.Series:
        """Count of each predominant_stress label."""
        if self.frame is None:
            raise RuntimeError("Call load_and_match first.")
        return self.frame["predominant_stress"].value_counts().sort_index()

    def eda_summary(self) -> Dict[str, object]:
        """Small dict used by plots and by tests."""
        if self.frame is None:
            raise RuntimeError("Call load_and_match first.")
        counts = self.class_counts()
        return {
            "n_csv": self.n_csv,
            "n_kept": int(len(self.frame)),
            "drop_mixed": self.drop_mixed,
            "class_counts": {int(k): int(v) for k, v in counts.items()},
        }


class TrainValTestSplit:
    """Stratified 70 / 15 / 15 split. Indices are positions in dataset.frame."""

    def __init__(self, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
        self.train_idx = np.asarray(train_idx)
        self.val_idx = np.asarray(val_idx)
        self.test_idx = np.asarray(test_idx)

    @classmethod
    def make(
        cls,
        dataset: LeafDataset,
        seed: int = RANDOM_STATE,
        path: Optional[str] = None,
    ) -> "TrainValTestSplit":
        """Build a new split, or load one from path if that file already exists."""
        if path and os.path.exists(path):
            return cls.load(path)
        if dataset.frame is None:
            raise RuntimeError("Call load_and_match first.")
        y = dataset.frame["predominant_stress"].values
        idx = np.arange(len(dataset.frame))
        # Split test off first (15%), then val (15% of the original, i.e. 15/85 of the rest).
        train_val, test = train_test_split(idx, test_size=0.15, stratify=y, random_state=seed)
        try:
            train, val = train_test_split(
                train_val, test_size=0.15 / 0.85, stratify=y[train_val], random_state=seed
            )
        except ValueError:
            train, val = train_test_split(
                train_val, test_size=0.15 / 0.85, random_state=seed
            )
        split = cls(train, val, test)
        if path:
            split.save(path)
        return split

    def save(self, path: str) -> None:
        """Write train / val / test indices to an .npz file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, train_idx=self.train_idx, val_idx=self.val_idx, test_idx=self.test_idx)

    @classmethod
    def load(cls, path: str) -> "TrainValTestSplit":
        """Read a split saved by save()."""
        data = np.load(path)
        return cls(data["train_idx"], data["val_idx"], data["test_idx"])

    def sizes(self) -> Dict[str, int]:
        """Number of samples in each split."""
        return {
            "train": int(len(self.train_idx)),
            "val": int(len(self.val_idx)),
            "test": int(len(self.test_idx)),
        }

    def assert_disjoint(self) -> None:
        """Raise if any image index sits in two splits."""
        t, v, s = set(self.train_idx), set(self.val_idx), set(self.test_idx)
        if t & v or t & s or v & s:
            raise ValueError("Split indices overlap.")
