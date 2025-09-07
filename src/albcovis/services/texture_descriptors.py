from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image
from skimage import color, filters, feature, util
from scipy import ndimage as ndi

# ----------------------------- Entropy utility (base-2, normalized) -----------------------------

def shannon_entropy_norm(p: np.ndarray, nbins: int) -> float:
    """
    Shannon entropy with log base 2, normalized to [0,1].

    Args:
        p: 1D or ND array of nonnegative values representing either probabilities or counts.
           (Zeros are allowed; they contribute 0 to the sum.)
        nbins: The intended number of categories/bins.

    Returns:
        H_norm in [0,1] where 0 means perfectly concentrated (one bin has prob=1),
        and 1 means perfectly uniform over 'nbins' bins.
    """
    q = np.asarray(p, dtype=np.float64).ravel()
    s = q.sum()
    if s <= 0 or not np.isfinite(s):
        return 0.0

    # Convert counts -> probabilities (safe even if already probs)
    q = q / s

    # Compute H = -sum p log2 p (ignoring p==0 to avoid log2(0))
    mask = q > 0
    H = -(q[mask] * np.log2(q[mask])).sum()

    # Normalize by the maximum possible entropy for 'nbins' categories: log2(nbins)
    if nbins <= 1:
        return 0.0
    H_norm = H / np.log2(float(nbins))
    # Numerical safety
    return float(np.clip(H_norm, 0.0, 1.0))



# -------------------------------- Edge density & Orientation entropy -----------------------------------

def _auto_canny_thresholds(gray01: np.ndarray, low_scale: float = 0.66, high_scale: float = 1.33) -> Tuple[float, float]:
    """
    Compute adaptive Canny thresholds from the image using Otsu's threshold on gray.
    We scale Otsu up/down to get a stable pair of thresholds, and then clamp to [0,1].
    """
    # Otsu returns a threshold in the same range as the image (here [0,1]).
    otsu = filters.threshold_otsu(gray01)
    low = np.clip(otsu * low_scale, 0.0, 1.0)
    high = np.clip(otsu * high_scale, 0.0, 1.0)
    # Ensure low < high and give some minimum gap if needed
    if high <= low:
        # fallback: set a small gap around otsu
        low = max(0.0, otsu * 0.8)
        high = min(1.0, otsu * 1.2)
        if high <= low:
            # extreme corner case: use fixed thresholds
            low, high = 0.1, 0.3
    return float(low), float(high)

def canny_thresholds_otsu_on_gray(
    gray01: np.ndarray,
    *,
    ratio: float = 3.0,
    safety_floor: float = 1e-6,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Otsu on *raw grayscale intensities* (not gradients), with Canny-style ratio.

    Parameters
    ----------
    gray01 : float image in [0,1]
    ratio : target high:low ratio (e.g., 3.0 -> low = high/3)
    safety_floor : minimum positive threshold to avoid degeneracy

    Returns
    -------
    low, high, info
    """
    # Guard against pathological inputs
    if gray01.size == 0 or not np.isfinite(gray01).any():
        high = 0.02
        low  = high / max(ratio, 1.0)
        return float(low), float(high), {"fallback": 1.0, "t_otsu": None}

    # 1) Otsu on raw intensities (same units as gray01)
    t = float(filters.threshold_otsu(gray01))

    # 2) Map to hysteresis thresholds with Canny's guidance
    high = float(np.clip(t, safety_floor, 1.0))
    low  = float(max(high / max(ratio, 1.0), safety_floor))

    # Ensure ordering
    if low >= high:  # extremely low contrast + clamp edge case
        low  = max(high * 0.5, safety_floor)
        # keep at least a tiny gap
        high = float(min(1.0, max(high, low * 1.01)))

    info = {
        "t_otsu": t,
        "ratio_target": float(ratio),
        "ratio_effective": float(high / max(low, 1e-12)),
        "fallback": 0.0,
    }
    return low, high, info

def canny_thresholds_otsu_on_gradmag(
    gray01: np.ndarray,
    *,
    sigma: float = 1.0,
    ratio: float = 3.0,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute Canny thresholds by:
      1) Gaussian smoothing (sigma, skimage.filters.gaussian)
      2) Sobel gradients with scipy.ndimage.sobel -> gradient magnitude
      3) Otsu on the magnitude (strictly positive entries)
      4) low = high / ratio

    Returns (low, high, info) where thresholds are in the same units as gray01.
    """
    # 1) Smooth like skimage.canny does before its Sobel step
    smoothed = filters.gaussian(gray01, sigma=sigma, preserve_range=True)

    # 2) Sobel gradients and magnitude (match skimage.canny internals)
    gx = ndi.sobel(smoothed, axis=1)
    gy = ndi.sobel(smoothed, axis=0)
    mag = np.hypot(gx, gy)

    # 3) Otsu on strictly positive magnitudes to avoid the zero spike
    vals = mag[mag > 0]
    info: Dict[str, float] = {}
    if vals.size == 0:
        high = 0.02
        low = high / max(ratio, 1.0)
        info.update(fallback=1.0, t_otsu=None)
        return float(low), float(high), info

    t = float(filters.threshold_otsu(vals))
    high = max(t, 1e-6)
    low = max(high / max(ratio, 1.0), 1e-6)

    info.update(fallback=0.0, t_otsu=t, ratio_effective=high / max(low, 1e-12))
    return float(low), float(high), info


# (warning) import private function, can be risky and unstable when skimage gets updated (use on 0.25.2)
from skimage.feature._canny_cy import _nonmaximum_suppression_bilinear as _nms_bilinear

def canny_thresholds_otsu_on_nms(
    gray01: np.ndarray,
    *,
    sigma: float = 1.0,
    ratio: float = 3.0,
) -> Tuple[float, float, dict]:
    """
    Minimal-fidelity replica:
      - Gaussian smooth (skimage.filters.gaussian)
      - Sobel gx, gy (scipy.ndimage.sobel compatible with skimage.canny)
      - NMS via _nms_bilinear with a simple border-cleared mask
      - Otsu on thinned magnitudes, then low = high / ratio
    """
    # Smooth
    smoothed = filters.gaussian(gray01, sigma=sigma, preserve_range=True)

    # Sobel like skimage.canny (ndi.sobel on each axis)
    gx = ndi.sobel(smoothed, axis=1)
    gy = ndi.sobel(smoothed, axis=0)
    mag = np.hypot(gx, gy)

    # Simple eroded mask: ones except a 1-px cleared border (matches skimage idea)
    eroded_mask = np.ones_like(smoothed, dtype=bool)
    eroded_mask[:1, :] = False
    eroded_mask[-1:, :] = False
    eroded_mask[:, :1] = False
    eroded_mask[:, -1:] = False

    # NMS with zero low threshold to keep all thinned responses
    nms = _nms_bilinear(gx, gy, mag, eroded_mask, 0.0)

    # Otsu on positive NMS responses
    vals = nms[nms > 0]
    info = {"nms_nonzero_fraction": float(vals.size) / float(nms.size)}
    if vals.size == 0:
        high = 0.02
        low = high / max(ratio, 1.0)
        info["fallback"] = True
        info["t_otsu"] = None
        return float(low), float(high), info

    t = float(filters.threshold_otsu(vals))
    high = max(t, 1e-6)
    low = max(high / max(ratio, 1.0), 1e-6)
    info["t_otsu"] = t
    info["fallback"] = False
    info["ratio_effective"] = high / max(low, 1e-12)
    return float(low), float(high), info


def compute_canny_edges(gray01: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Compute a binary edge map using scikit-image's Canny detector. Thresholds are computed using Otsu on NMS gradient magnitude array.
    - gray01: grayscale float image in [0,1]
    - sigma: Gaussian smoothing for noise reduction before edge detection
    Returns a boolean array (H, W) where True means "edge pixel".
    """
    low, high, info = canny_thresholds_otsu_on_nms(gray01)
    edges = feature.canny(gray01, sigma=sigma, low_threshold=low, high_threshold=high)
    return edges


def compute_sobel_gradients(gray01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Sobel horizontal and vertical gradients and the gradient magnitude.
    Returns (gx, gy, mag) as float32 arrays.
    """
    gx = filters.sobel_h(gray01).astype(np.float32)
    gy = filters.sobel_v(gray01).astype(np.float32)
    mag = np.hypot(gx, gy).astype(np.float32)
    return gx, gy, mag

def compute_edge_density(edges: np.ndarray) -> float:
    """
    Fraction of pixels that are edges.
    edges: boolean array (H, W) from Canny.
    Returns a float in [0,1].
    """
    total = edges.size
    count = int(edges.sum())
    return count / float(total) if total > 0 else 0.0


def compute_orientation_entropy(gx: np.ndarray, gy: np.ndarray, mag: np.ndarray, nbins: int = 18) -> float:
    """
    Entropy of edge orientations, weighted by gradient magnitude.
    - We collapse angles to [0, pi) (orientation, not direction).
    - We build a magnitude-weighted histogram over 'nbins' bins.
    - Shannon entropy is normalized to [0,1] by dividing by log2(nbins).

    Returns: normalized entropy in [0,1], where:
      * 0 ~ all edges share the same orientation (very ordered)
      * 1 ~ orientations are spread uniformly (very diverse / cluttered)
    """
    # angle in [-pi, pi], convert to [0, pi) for orientation (ignoring sign/direction)
    angles = np.arctan2(gy, gx)  # [-pi, pi]
    orientations = np.mod(angles, np.pi)  # [0, pi)

    # We only care about locations with nontrivial gradients to avoid noise
    eps = 1e-8
    mask = mag > (mag.mean() + mag.std() * 0.3)  # keep stronger gradients
    if not np.any(mask):
        return 0.0

    weighted_hist, bin_edges = np.histogram(
        orientations[mask],
        bins=nbins,
        range=(0.0, np.pi),
        weights=mag[mask].astype(np.float64)  # weights must be float64 for np.histogram
    )

    # # Normalize to probabilities
    # p = weighted_hist / (weighted_hist.sum() + eps)

    return shannon_entropy_norm(weighted_hist, nbins=nbins)

# ---------------------------------- Pixel intensity entropy ------------------------------------------

def compute_entropy_gray(gray01: np.ndarray, nbins: int = 256) -> float:
    """
    Compute Shannon entropy of pixel intensities in a grayscale image.
    
    Args:
        gray01: Grayscale float image in [0,1].
        nbins: Number of histogram bins (default=256).
    
    Returns:
        Normalized entropy in [0,1].
    """
    # 1) Compute histogram
    hist, _ = np.histogram(gray01, bins=nbins, range=(0.0, 1.0))
    
    # # 2) Normalize to probabilities
    # p = hist.astype(np.float64) / hist.sum()
    
    # return Shannon entropy
    return shannon_entropy_norm(hist, nbins)

# ---------------------------------- GLCM features (Haralick descriptors) ------------------------------------------

# Helper to quantize grayscale to a fixed number of levels (for efficiency)
def _quantize_gray_levels(gray01: np.ndarray, levels: int = 32) -> np.ndarray:
    """
    Quantize a grayscale float image in [0,1] to integer levels in [0, levels-1].
    Returns uint8 (or uint16 if levels > 256) as required by graycomatrix.
    """
    gray01 = np.clip(gray01, 0.0, 1.0).astype(np.float32)
    # Map [0,1] -> {0, 1, ..., levels-1}
    q = np.floor(gray01 * levels).astype(np.int32)
    q[q == levels] = levels - 1  # handle edge-case where gray01==1.0
    # Choose dtype that fits 'levels'
    if levels <= 256:
        return q.astype(np.uint8)
    elif levels <= 65536:
        return q.astype(np.uint16)
    else:
        raise ValueError("levels too large for skimage.graycomatrix.")

def _glcm_entropy_norm2(glcm: np.ndarray, levels: int) -> float:
    """
    glcm: (levels, levels, D, A), with normed=True so each P sums to 1.
    Returns mean normalized entropy (base-2) across all (D,A).
    """
    D, A = glcm.shape[2], glcm.shape[3]
    ent = []
    nbins = levels * levels
    for d in range(D):
        for a in range(A):
            P = glcm[:, :, d, a]
            ent.append(shannon_entropy_norm(P, nbins=nbins))
    return float(np.mean(ent)) if ent else 0.0


# Main function to compute selected Haralick (GLCM) features
def compute_glcm_features(
    gray01: np.ndarray,
    levels: int = 32,
    distances=(1, 2, 4),
    angles=(0.0, np.pi/4, np.pi/2, 3*np.pi/4),
    symmetric: bool = True,
    normed: bool = True,
) -> dict:
    """
    Compute GLCM features and provide normalized [0,1] variants for
    contrast, correlation, and entropy (the rest are already in [0,1]).

    Args:
        gray01: Grayscale float image in [0,1].
        levels: Number of gray levels for quantization (typical: 16–64; default 32).
        distances: Pixel offsets for GLCM.
        angles: Directions (radians) for GLCM (0, 45, 90, 135 degrees).
        symmetric: Make GLCM symmetric (recommended).
        normed: Normalize GLCM to probabilities (recommended).

    Returns keys:
      - glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_entropy, glcm_contrast_norm, glcm_correlation_norm, glcm_entropy_norm
      - All returned metrics are returned in [0,1]
    """
    # 1) Quantize to discrete gray levels required by GLCM
    q = _quantize_gray_levels(gray01, levels=levels)

    # 2) Build the GLCM matriox: shape (levels, levels, len(distances), len(angles))
    glcm = feature.graycomatrix(
        q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=symmetric,
        normed=normed,
    )

    # Mean over all (distance, angle) combinations
    def _prop_mean(name: str) -> float:
        arr = feature.graycoprops(glcm, prop=name)  # shape (D, A)
        if name == "correlation":
            # correlation can be NaN for constant images (zero variance)
            arr = np.nan_to_num(arr, nan=1.0)
        return float(arr.mean())

    contrast      = _prop_mean("contrast")     # in [0, (L−1)²]
    homogeneity   = _prop_mean("homogeneity")  # already in [0,1]
    energy        = _prop_mean("energy")       # already in [0,1]
    correlation   = _prop_mean("correlation")  # in [-1,1]
    entropy_val   = _glcm_entropy_norm2(glcm, levels=levels) # already in [0,1]

    # Normalizations
    # Contrast: divide by (L-1)^2
    contrast_norm = contrast / float((levels - 1) ** 2)
    contrast_norm = float(np.clip(contrast_norm, 0.0, 1.0))

    # Correlation: map [-1,1] -> [0,1]
    corr_norm = (correlation + 1.0) / 2.0
    corr_norm = float(np.clip(corr_norm, 0.0, 1.0))

    return {
        "glcm_contrast": contrast_norm,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": corr_norm,
        "glcm_entropy": entropy_val,
    }

# ---------------------------------- Local Binary Patterns ------------------------------------------

def compute_lbp_features(
    gray01: np.ndarray,
    P: int = 8,
    R: float = 1.0,
    method: str = "uniform",
) -> dict:
    """
    Compute Local Binary Patterns (LBP) on a grayscale float image in [0,1]
    and return two histogram-based descriptors:
      - lbp_entropy: Shannon entropy of the LBP histogram, normalized to [0,1]
      - lbp_energy:  Sum of squared bin probabilities, in [0,1]

    Args:
        gray01: Grayscale image as float32/float64 in [0,1].
        P:     Number of circularly symmetric neighbor set points.
        R:     Radius of circle.
        method: 'uniform' (recommended), 'default', 'ror', etc. (skimage options)

    Returns:
        dict with:
          {
            "lbp_entropy": float in [0,1],
            "lbp_energy":  float in [0,1]
          }
    """
    # 1) Compute LBP code image
    lbp = feature.local_binary_pattern(gray01, P=P, R=R, method=method)

    # 2) Build normalized histogram over LBP codes
    n_bins = (P + 2) if method == "uniform" else int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=np.arange(n_bins + 1), range=(0, n_bins))
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        # Degenerate case (empty image). Return zeros.
        return {"lbp_entropy": 0.0, "lbp_energy": 0.0}
    p /= total

    # 3) LBP histogram entropy (normalized to [0,1])
    lbp_entropy = shannon_entropy_norm(hist, nbins=n_bins) 

    # 4) LBP histogram energy (sum of squares) in [0,1]
    lbp_energy = float((p * p).sum())

    return {
        "lbp_entropy": lbp_entropy,
        "lbp_energy": lbp_energy,
    }

# -------------------------------- Orchestration layer -----------------------------------
def extract_visual_complexity(gray01: np.ndarray):
    # Edge density & Orientation entropy
    edges = compute_canny_edges(gray01, sigma=0.75)
    edge_density = compute_edge_density(edges)
    gx, gy, mag = compute_sobel_gradients(gray01)
    orientation_entropy = compute_orientation_entropy(gx, gy, mag, nbins=18)

    # Pixel intensity entropy
    pixel_intensity_entropy = compute_entropy_gray(gray01)

    # GLCM features
    glcm_features = compute_glcm_features(gray01)
    glcm_contrast = glcm_features["glcm_contrast"]
    glcm_homogeneity = glcm_features["glcm_homogeneity"]
    glcm_energy = glcm_features["glcm_energy"]
    glcm_correlation = glcm_features["glcm_correlation"]
    glcm_entropy = glcm_features["glcm_entropy"]

    # LBP Features
    lbp_features = compute_lbp_features(gray01)
    lbp_entropy = lbp_features["lbp_entropy"]
    lbp_energy = lbp_features["lbp_energy"]

    # build group scores
    group_A = (pixel_intensity_entropy + glcm_entropy + lbp_entropy) / 3.0
    group_B = ((1.0 - glcm_homogeneity) + (1.0 - glcm_energy) + (1.0 - lbp_energy)) / 3.0
    group_C = (edge_density + orientation_entropy) / 2.0
    group_D = (glcm_contrast + (1.0 - glcm_correlation)) / 2.0

    # group weight
    wA, wB, wC, wD = 0.45, 0.30, 0.20, 0.05

    visual_complexity = (
        wA * group_A +
        wB * group_B +
        wC * group_C +
        wD * group_D
    )

    '''
    We group ten descriptors into four phenomena—entropy/busyness, non-uniformity (anti-smoothness),
    edge geometry, and contrast/predictability—based on their inter-correlations and conceptual overlap.
    We weight groups 0.45/0.30/0.20/0.05 and average equally within each group.
    This avoids double-counting correlated measures, keeps the index interpretable,
    and aligns with perceptual notions of visual complexity while remaining deterministic and dataset-independent.
    '''

    return {
        "texture_descriptors" : {
            "edge_density": edge_density,
            "orientation_entropy": orientation_entropy,
            
            "pixel_intensity_entropy": pixel_intensity_entropy,

            "glcm_contrast": glcm_contrast,
            "glcm_homogeneity": glcm_homogeneity,
            "glcm_energy": glcm_energy,
            "glcm_correlation": glcm_correlation,
            "glcm_entropy": glcm_entropy,

            "lbp_entropy": lbp_entropy,
            "lbp_energy": lbp_energy,
        },

        "visual_complexity": visual_complexity
    }