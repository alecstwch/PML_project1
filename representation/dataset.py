"""CSV + JPEG loading and the shared 70/15/15 split."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from representation.constants import RANDOM_STATE


def is_jpeg_valid(path: str) -> bool:
    """Return True if the file opens as a complete image."""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


class LeafDataset:
    """Load the leaf CSV and keep rows whose image file can be read."""

    def __init__(self, csv_path: str, image_dir: str, check_jpeg: bool = True):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.check_jpeg = check_jpeg
        self.frame: Optional[pd.DataFrame] = None
        self.n_csv: int = 0
        self.n_missing: int = 0
        self.n_corrupt: int = 0
        self.corrupt_ids: List[int] = []
        self.duplicate_groups: List[List[int]] = []

    def load_and_match(self) -> pd.DataFrame:
        """Load the CSV and drop rows whose image file is missing or corrupt."""
        df = pd.read_csv(self.csv_path)
        self.n_csv = len(df)
        df["image_path"] = df["id"].apply(lambda x: os.path.join(self.image_dir, f"{x}.jpg"))
        exists = df["image_path"].apply(os.path.exists)
        self.n_missing = int((~exists).sum())
        df = df[exists].reset_index(drop=True)

        if self.check_jpeg:
            valid = []
            self.corrupt_ids = []
            for _, row in df.iterrows():
                ok = is_jpeg_valid(row["image_path"])
                valid.append(ok)
                if not ok:
                    self.corrupt_ids.append(int(row["id"]))
            self.n_corrupt = len(self.corrupt_ids)
            df = df[pd.Series(valid)].reset_index(drop=True)
        else:
            self.n_corrupt = 0
            self.corrupt_ids = []

        self.frame = df
        return df

    def class_counts(self) -> pd.Series:
        """Count of each predominant_stress label."""
        if self.frame is None:
            raise RuntimeError("Call load_and_match first.")
        return self.frame["predominant_stress"].value_counts().sort_index()

    def find_duplicates(self, hash_size: int = 8) -> List[List[int]]:
        """Group image ids that share the same perceptual hash."""
        if self.frame is None:
            raise RuntimeError("Call load_and_match first.")
        try:
            import imagehash
        except ImportError:
            imagehash = None

        buckets: Dict[str, List[int]] = {}
        for _, row in self.frame.iterrows():
            key = self._hash_image(row["image_path"], hash_size, imagehash)
            buckets.setdefault(key, []).append(int(row["id"]))
        self.duplicate_groups = [ids for ids in buckets.values() if len(ids) > 1]
        return self.duplicate_groups

    @staticmethod
    def _hash_image(path: str, hash_size: int, imagehash_mod) -> str:
        """Perceptual hash of one file (imagehash if present, else a tiny average hash)."""
        img = Image.open(path).convert("RGB")
        if imagehash_mod is not None:
            return str(imagehash_mod.phash(img, hash_size=hash_size))
        small = img.resize((hash_size, hash_size), Image.Resampling.BILINEAR).convert("L")
        arr = np.asarray(small, dtype=np.float32)
        bits = arr > arr.mean()
        return "".join("1" if b else "0" for b in bits.ravel())

    def eda_summary(self) -> Dict[str, object]:
        """Small dict used by plots and by tests."""
        if self.frame is None:
            raise RuntimeError("Call load_and_match first.")
        counts = self.class_counts()
        return {
            "n_csv": self.n_csv,
            "n_kept": int(len(self.frame)),
            "n_missing": self.n_missing,
            "n_corrupt": self.n_corrupt,
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
        # Doing 70/30 then 50/50 can leave one sample of a rare class in the 30% pool, and
        # sklearn refuses to stratify that.
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
