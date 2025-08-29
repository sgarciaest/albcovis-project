from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image
from skimage import color, filters, feature, util
from scipy import ndimage as ndi


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



def edge_density(edges: np.ndarray) -> float:
    """
    Fraction of pixels that are edges.
    edges: boolean array (H, W) from Canny.
    Returns a float in [0,1].
    """
    total = edges.size
    count = int(edges.sum())
    return count / float(total) if total > 0 else 0.0


def orientation_entropy_from_sobel(gx: np.ndarray, gy: np.ndarray, mag: np.ndarray, nbins: int = 18) -> float:
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
    # Normalize to probabilities
    p = weighted_hist / (weighted_hist.sum() + eps)
    # Shannon entropy (base 2)
    entropy = -(p[p > 0] * np.log2(p[p > 0])).sum()
    # Normalize to [0,1]
    entropy_norm = float(entropy / np.log2(nbins))
    return entropy_norm

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
    
    # 2) Normalize to probabilities
    p = hist.astype(np.float64) / hist.sum()
    
    # 3) Compute Shannon entropy (ignoring zero probabilities)
    p_nonzero = p[p > 0]
    H = -(p_nonzero * np.log2(p_nonzero)).sum()
    
    # 4) Normalize by maximum possible entropy (log2(nbins))
    H_norm = H / np.log2(nbins)
    
    return float(H_norm)
