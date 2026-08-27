"""Shared constants for labels, sizes, and handcrafted descriptors."""

STRESS_NAMES = {
    0: "Healthy",
    1: "Miner",
    2: "Rust",
    3: "Phoma",
    4: "Cercospora",
    5: "Mixed",
}

SEVERITY_NAMES = {
    0: "Healthy",
    1: "Very Low",
    2: "Low",
    3: "High",
    4: "Very High",
}

# BRACOL YOLO class ids are not the same as predominant_stress.
BRACOL_CLASSES = {0: "Cercospora", 1: "Miner", 2: "Phoma", 3: "Rust"}
BRACOL_TO_STRESS = {0: 4, 1: 1, 2: 3, 3: 2}

RANDOM_STATE = 42
IMG_SIZE = 224
HANDCRAFTED_SIZE = 224
HOG_CELL = (8, 8)
HOG_BLOCK = (2, 2)
HOG_ORIENT = 9
HSV_BINS = (8, 8, 8)
LBP_POINTS = 24
LBP_RADIUS = 3

# White paper: low saturation, high value. Leaf pixels are the rest.
MASK_S_MAX = 40
MASK_V_MIN = 180
