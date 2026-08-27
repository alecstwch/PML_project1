"""
Shared classes for Project 1: coffee leaf disease classification.

Thin facade so the notebook and tests keep `from p1_core import ...`.
Implementation lives in representation/, evaluation/, and optimization/.
"""

from __future__ import annotations

from evaluation.metrics import classification_metrics, target_names_for
from evaluation.plots import save_and_show
from evaluation.rules import (
    ResidualMixedRule,
    apply_mixup,
    apply_residual_rule,
    ensemble_average,
    fit_disease_probe,
    load_bracol_labels,
)
from optimization.compute import fmt_hms, report_compute
from optimization.loop import LearnerLoop, class_weight_dict, sample_weights
from representation.constants import (
    BRACOL_CLASSES,
    BRACOL_TO_STRESS,
    HANDCRAFTED_SIZE,
    HOG_BLOCK,
    HOG_CELL,
    HOG_ORIENT,
    HSV_BINS,
    IMG_SIZE,
    LBP_POINTS,
    LBP_RADIUS,
    MASK_S_MAX,
    MASK_V_MIN,
    RANDOM_STATE,
    SEVERITY_NAMES,
    STRESS_NAMES,
)
from representation.dataset import LeafDataset, TrainValTestSplit, is_jpeg_valid
from representation.features import (
    FeatureExtractor,
    FeaturePack,
    FrozenDeepFeatures,
    HandcraftedFeatures,
    PCAFeatures,
    handcrafted_vector,
    hog_vector,
    hsv_histogram,
    l2_normalize,
    leaf_mask,
    lbp_histogram,
    read_bgr,
    scale_pack,
)
from representation.models_boost import XGBoostPipeline
from representation.models_cnn import (
    EfficientNetPipeline,
    ResNetPipeline,
    load_rgb_batch,
    load_rgb_batch_hsv_crop,
)
from representation.models_svm import SVMPipeline
from representation.paths import (
    _leaf_id_from_filename,
    copy_missing_leaf_jpgs,
    extra_image_roots,
    resolve_paths,
)

# Back-compat alias used internally before the split.
_sample_weights = sample_weights

__all__ = [
    "STRESS_NAMES",
    "SEVERITY_NAMES",
    "BRACOL_CLASSES",
    "BRACOL_TO_STRESS",
    "RANDOM_STATE",
    "IMG_SIZE",
    "HANDCRAFTED_SIZE",
    "HOG_CELL",
    "HOG_BLOCK",
    "HOG_ORIENT",
    "HSV_BINS",
    "LBP_POINTS",
    "LBP_RADIUS",
    "MASK_S_MAX",
    "MASK_V_MIN",
    "fmt_hms",
    "resolve_paths",
    "report_compute",
    "_leaf_id_from_filename",
    "copy_missing_leaf_jpgs",
    "extra_image_roots",
    "save_and_show",
    "is_jpeg_valid",
    "leaf_mask",
    "read_bgr",
    "l2_normalize",
    "classification_metrics",
    "target_names_for",
    "apply_mixup",
    "apply_residual_rule",
    "LeafDataset",
    "TrainValTestSplit",
    "FeaturePack",
    "FeatureExtractor",
    "hsv_histogram",
    "lbp_histogram",
    "hog_vector",
    "handcrafted_vector",
    "HandcraftedFeatures",
    "FrozenDeepFeatures",
    "PCAFeatures",
    "scale_pack",
    "LearnerLoop",
    "SVMPipeline",
    "XGBoostPipeline",
    "sample_weights",
    "_sample_weights",
    "class_weight_dict",
    "load_rgb_batch",
    "load_rgb_batch_hsv_crop",
    "EfficientNetPipeline",
    "ResNetPipeline",
    "ResidualMixedRule",
    "fit_disease_probe",
    "load_bracol_labels",
    "ensemble_average",
]
