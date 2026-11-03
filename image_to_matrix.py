"""
image_to_matrix.py

Converts a static connectivity-matrix PNG image into a numeric
[NUM_ROIS x NUM_ROIS] correlation-like matrix in the range [-1, 1].

Two extraction modes are supported:
  1. Colormap inversion - if the image looks like a colored heatmap
     (average saturation above SATURATION_THRESHOLD), pixel colors are
     matched against a lookup table built from ASSUMED_COLORMAP to
     recover an approximate correlation value per pixel.
  2. Grayscale fallback - if the image looks grayscale, pixel intensity
     is normalized directly to [-1, 1] and used as a proxy for
     correlation strength.

Either way, the result is symmetrized and the diagonal is zeroed, since
self-correlation is not meaningful for the graph we build downstream.

Tested against: Python 3.9.0
"""

import os
import hashlib
from typing import Optional

import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import config


def _build_colormap_lut(colormap_name: str, lut_size: int) -> np.ndarray:
    """
    Build a lookup table of RGB colors sampled evenly across a matplotlib
    colormap, alongside the scalar value in [-1, 1] each color corresponds
    to. Used for nearest-color matching during colormap inversion.

    Returns
    -------
    colors : np.ndarray of shape [lut_size, 3], values in [0, 255]
    values : np.ndarray of shape [lut_size], values in [-1, 1]
    """
    colormap = cm.get_cmap(colormap_name, lut_size)
    values = np.linspace(-1.0, 1.0, lut_size)
    colors = np.array([colormap(i)[:3] for i in range(lut_size)]) * 255.0
    return colors.astype(np.float32), values.astype(np.float32)


_LUT_COLORS, _LUT_VALUES = _build_colormap_lut(
    config.ASSUMED_COLORMAP, config.COLORMAP_LUT_SIZE
)


def _estimate_saturation(rgb_image: np.ndarray) -> float:
    """Average HSV saturation of an RGB image in [0, 1], used to decide
    whether the image is a colored heatmap or effectively grayscale."""
    hsv = mcolors.rgb_to_hsv(rgb_image.astype(np.float32) / 255.0)
    return float(np.mean(hsv[..., 1]))


def _invert_colormap(rgb_image: np.ndarray) -> np.ndarray:
    """
    Map every pixel's RGB color to the nearest color in the colormap LUT
    and return the corresponding scalar value. Vectorized via a distance
    matrix between all pixels and all LUT entries.
    """
    h, w, _ = rgb_image.shape
    pixels = rgb_image.reshape(-1, 3).astype(np.float32)  # [P, 3]

    # Squared Euclidean distance from every pixel to every LUT color.
    # [P, 1, 3] - [1, L, 3] -> [P, L, 3] -> sum -> [P, L]
    diffs = pixels[:, None, :] - _LUT_COLORS[None, :, :]
    dist_sq = np.sum(diffs ** 2, axis=-1)
    nearest_idx = np.argmin(dist_sq, axis=-1)  # [P]

    values = _LUT_VALUES[nearest_idx]
    return values.reshape(h, w)


def _grayscale_fallback(gray_image: np.ndarray) -> np.ndarray:
    """Normalize grayscale intensity (0-255) to a [-1, 1] proxy value."""
    normalized = gray_image.astype(np.float32) / 255.0  # [0, 1]
    return normalized * 2.0 - 1.0  # [-1, 1]


def load_png_as_matrix(
    image_path: str,
    num_rois: int = config.NUM_ROIS,
    use_cache: bool = True,
    cache_dir: str = config.CACHE_DIR,
) -> np.ndarray:
    """
    Load a connectivity-matrix PNG and convert it to a numeric
    [num_rois, num_rois] matrix in [-1, 1]. Symmetrized, diagonal zeroed.

    Results are cached to `cache_dir` as .npy files keyed by a hash of the
    absolute image path plus num_rois, so re-runs skip recomputation.
    """
    cache_path = None
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        key = hashlib.md5(f"{os.path.abspath(image_path)}_{num_rois}".encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{key}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((num_rois, num_rois), Image.BILINEAR)
        rgb = np.array(img)

    saturation = _estimate_saturation(rgb)
    if saturation >= config.SATURATION_THRESHOLD:
        matrix = _invert_colormap(rgb)
    else:
        gray = np.array(Image.fromarray(rgb).convert("L"))
        matrix = _grayscale_fallback(gray)

    # Symmetrize and remove self-connectivity.
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    matrix = matrix.astype(np.float32)

    if use_cache and cache_path is not None:
        np.save(cache_path, matrix)

    return matrix


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_dataset_matrices(
    class_dir_map: dict,
    num_rois: int = config.NUM_ROIS,
    use_cache: bool = True,
) -> "tuple[list, list, list]":
    """
    Walk a {folder_path: label} mapping and load every image inside each
    folder as a numeric matrix. Accepts .png/.jpg/.jpeg/.bmp/.tif(f),
    matched case-insensitively.

    Returns
    -------
    matrices : list of np.ndarray, one [num_rois, num_rois] matrix per file
    labels   : list of int labels, aligned with `matrices`
    filenames: list of str basenames, aligned with `matrices`
    """
    matrices, labels, filenames = [], [], []
    per_folder_counts = {}

    for folder_path, label in class_dir_map.items():
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Expected class folder not found: {folder_path}")

        found = [
            fname for fname in sorted(os.listdir(folder_path))
            if fname.lower().endswith(IMAGE_EXTENSIONS)
        ]
        per_folder_counts[folder_path] = len(found)

        for fname in found:
            full_path = os.path.join(folder_path, fname)
            matrix = load_png_as_matrix(full_path, num_rois=num_rois, use_cache=use_cache)
            matrices.append(matrix)
            labels.append(label)
            filenames.append(fname)

    print("Image counts per class folder:")
    for folder_path, count in per_folder_counts.items():
        print(f"  {folder_path}: {count} image(s)")

    empty_folders = [f for f, c in per_folder_counts.items() if c == 0]
    if empty_folders:
        raise RuntimeError(
            "The following class folder(s) contain zero readable images "
            f"(checked extensions: {IMAGE_EXTENSIONS}): {empty_folders}. "
            "Check the file extensions and folder paths - training cannot "
            "proceed with a missing class."
        )

    unique_labels = set(labels)
    if len(unique_labels) < 2:
        raise RuntimeError(
            f"Only {len(unique_labels)} class(es) found in the loaded data "
            f"({unique_labels}). Both Schizophrenia and Healthy folders must "
            "contain images. Check per-folder counts printed above."
        )

    return matrices, labels, filenames
