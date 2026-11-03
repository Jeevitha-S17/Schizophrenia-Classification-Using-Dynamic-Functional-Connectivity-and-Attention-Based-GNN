"""
Central configuration for the pseudo-dynamic functional-connectivity
pipeline. All "magic numbers" used across the pipeline live here so the
approximation choices are documented and easy to audit/tune.

Tested against: Python 3.9.0
"""

import os

# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------
DATASET_ROOT = "dataset"
TRAIN_DIR = os.path.join(DATASET_ROOT, "TRAIN")
TEST_DIR = os.path.join(DATASET_ROOT, "TEST")

# Folder name (as it appears on disk) -> integer label
# SZ = 1 (positive / schizophrenia), HC = 0 (negative / healthy control)
CLASS_FOLDERS = {
    "Schizophrenia": 1,
    "Healthy": 0,
}

CACHE_DIR = "cache_matrices"       # cached .npy numeric matrices
CHECKPOINT_DIR = "checkpoints"     # saved model weights + config.json

# ---------------------------------------------------------------------------
# Image -> matrix conversion
# ---------------------------------------------------------------------------
# Assumed number of ROIs (nodes). Every PNG is resized to NUM_ROIS x NUM_ROIS
# before being treated as a connectivity matrix. Change this if you know the
# atlas your PNGs were rendered from (e.g. AAL-116 -> 116, Schaefer-200 -> 200).
NUM_ROIS = 116

# If the image's average color saturation is below this threshold it is
# treated as grayscale (intensity-only) rather than colormap-encoded.
SATURATION_THRESHOLD = 0.12

# Colormap assumed for colored connectivity heatmaps, used to invert
# pixel color back to an approximate correlation value in [-1, 1].
# Must be a name registered in matplotlib.cm (e.g. "jet", "viridis", "coolwarm").
ASSUMED_COLORMAP = "jet"
COLORMAP_LUT_SIZE = 256  # resolution of the colormap lookup table

# ---------------------------------------------------------------------------
# Pseudo-dynamic ("bootstrap") window generation
# ---------------------------------------------------------------------------
# NOTE: This is an APPROXIMATION. There is no real time axis in the source
# PNGs. We simulate plausible temporal fluctuation by perturbing the static
# matrix N_WINDOWS times with small, reproducible Gaussian noise. This is
# NOT sliding-window DFC on a genuine BOLD time-series - see project report
# for the explicit caveat.
N_WINDOWS = 15
NOISE_STD_FRAC = 0.10     # noise std = NOISE_STD_FRAC * |edge value|
NOISE_MIN_STD = 0.01      # floor so near-zero edges still get *some* jitter
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
EDGE_TOP_K = 10  # keep the strongest EDGE_TOP_K edges (by |weight|) per node

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
HIDDEN_DIM = 64
NUM_GAT_HEADS = 4
GAT_LAYERS = 2
NUM_ATTENTION_LAYERS = 2   # transformer encoder layers over the window sequence
TRANSFORMER_HEADS = 4
DROPOUT = 0.3
NUM_CLASSES = 2

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
EPOCHS = 50
VAL_SPLIT = 0.15
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
EARLY_STOP_PATIENCE = 15
