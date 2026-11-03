"""
pseudo_dynamic.py

*** APPROXIMATION NOTICE ***
Real sliding-window Dynamic Functional Connectivity (DFC) requires a BOLD
time-series per ROI, split into overlapping temporal windows, with a
Pearson correlation matrix computed per window. Our dataset only provides
one static, already-computed connectivity matrix per subject - there is no
time-series left to window over.

To still give the downstream Attention-GNN multiple "windows" per subject
(so it can learn a sequence-attention pattern analogous to real DFC), we
generate N_WINDOWS statistically perturbed versions of the single static
matrix via reproducible Gaussian edge-wise bootstrap resampling. This
simulates plausible small fluctuations around the observed connectivity,
but it is NOT measured temporal brain activity. Report this explicitly
as a methodological limitation.

Tested against: Python 3.9.0
"""

import numpy as np

import config


def generate_pseudo_windows(
    matrix: np.ndarray,
    n_windows: int = config.N_WINDOWS,
    noise_std_frac: float = config.NOISE_STD_FRAC,
    noise_min_std: float = config.NOISE_MIN_STD,
    seed: int = config.RANDOM_SEED,
    subject_salt: int = 0,
) -> np.ndarray:
    """
    Generate `n_windows` perturbed copies of a static connectivity matrix
    via edge-wise Gaussian bootstrap.

    For each window and each upper-triangular edge (i, j), a noise term is
    drawn from N(0, std) where std = max(noise_std_frac * |M_ij|, noise_min_std).
    The perturbed matrix is symmetrized, diagonal zeroed, and clipped to
    [-1, 1] to stay in a valid correlation range.

    Parameters
    ----------
    matrix : np.ndarray [R, R]
        Static connectivity matrix for one subject.
    n_windows : int
        Number of pseudo-windows to generate.
    noise_std_frac : float
        Fraction of |edge value| used as the Gaussian noise std.
    noise_min_std : float
        Minimum noise std, so near-zero edges still get some jitter.
    seed : int
        Base random seed for reproducibility.
    subject_salt : int
        An additional integer (e.g. a subject index) mixed into the seed
        so different subjects don't all receive an identical noise
        pattern relative to their own matrix.

    Returns
    -------
    np.ndarray of shape [n_windows, R, R]
    """
    rng = np.random.default_rng(seed + subject_salt)
    r = matrix.shape[0]
    upper_idx = np.triu_indices(r, k=1)

    windows = np.zeros((n_windows, r, r), dtype=np.float32)
    base_values = matrix[upper_idx]
    stds = np.maximum(noise_std_frac * np.abs(base_values), noise_min_std)

    for w in range(n_windows):
        noise = rng.normal(loc=0.0, scale=stds).astype(np.float32)
        perturbed_upper = np.clip(base_values + noise, -1.0, 1.0)

        perturbed = np.zeros((r, r), dtype=np.float32)
        perturbed[upper_idx] = perturbed_upper
        perturbed = perturbed + perturbed.T  # symmetrize
        np.fill_diagonal(perturbed, 0.0)

        windows[w] = perturbed

    return windows
