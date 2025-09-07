# ------- Color Feature Based Dominant Color Extraction (CFDC) -------
# (For me it's more the prominent color extraction rather than dominant)
# (WARNING) This method needs:
# - Improve efficiency, faster code execution, use optimizations as in normal k-means
# - Understand intuitively each line

"""
Dominant color extraction (Chang & Mukai, IEEE Access 2022) — Python/scikit
Implements: bilateral filter (CIELAB) → K-means (large K) → RAG merge →
boundary cluster pruning → C/A/S feature computation → p = C+A+S →
weighted selection of 5 dominant colors.

Paper: "Color Feature Based Dominant Color Extraction" (Chang & Mukai, 2022)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage import color
from skimage.restoration import denoise_bilateral
from skimage.morphology import binary_erosion, disk
from skimage.util import img_as_float
from scipy.ndimage import gaussian_filter
from skimage.filters import gabor
from skimage import graph
from PIL import Image
from albcovis.utils.img import limit_image_size

# ------------------------- Utility helpers -------------------------

def _lab_to_unit(lab: np.ndarray) -> np.ndarray:
    """Scale CIELAB to [0,1] per channel for filtering."""
    L = lab[..., 0] / 100.0
    a = (lab[..., 1] + 128.0) / 255.0
    b = (lab[..., 2] + 128.0) / 255.0
    return np.stack([L, a, b], axis=-1)

def _unit_to_lab(unit: np.ndarray) -> np.ndarray:
    """Inverse of _lab_to_unit."""
    L = unit[..., 0] * 100.0
    a = unit[..., 1] * 255.0 - 128.0
    b = unit[..., 2] * 255.0 - 128.0
    return np.stack([L, a, b], axis=-1)

def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mn, mx = np.min(x), np.max(x)
    if mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def _deltaE76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    d = lab1 - lab2
    return float(np.sqrt(np.sum(d * d)))

def _hue_angle_ab(a: float, b: float) -> float:
    """Hue angle from a*, b* in radians, range [0, 2π)."""
    h = np.arctan2(b, a)
    if h < 0:
        h += 2 * np.pi
    return float(h)

def _circ_diff(h1: float, h2: float) -> float:
    """Smallest absolute difference between two angles."""
    d = abs(h1 - h2) % (2 * np.pi)
    return float(min(d, 2 * np.pi - d))

# Helpers to avoid buggy "<function _unique_dispatcher at 0x7f4ed4d92840> returned a result with an exception set"
def _as_c_int32(x):
    return np.asarray(x, dtype=np.int32, order="C")

def _as_c_float64(x):
    return np.asarray(x, dtype=np.float64, order="C")

def _safe_unique_int(x: np.ndarray) -> np.ndarray:
    """Robust unique for label arrays; avoids buggy NumPy dispatcher corner cases."""
    try:
        return np.unique(x)
    except Exception:
        lst = np.asarray(x, dtype=np.int64, order="C").ravel().tolist()
        return np.array(sorted(set(lst)), dtype=np.int64)


# ------------------------- Itti-like saliency (approx.) -------------------------

def itti_saliency_lab(lab: np.ndarray,
                      sigmas: Tuple[int, ...] = (1, 2, 4),
                      gabor_freq: float = 0.2) -> np.ndarray:
    """
    Simplified multi-scale saliency following Itti et al.:
    - center-surround differences on L*, a*, b* using Gaussian pyramids
    - orientation conspicuity via Gabor responses on L*
    Returns saliency map in [0,1].
    """
    L = lab[..., 0].astype(float)
    a = lab[..., 1].astype(float)
    b = lab[..., 2].astype(float)

    # Normalize L to [0,1] for gabor stability
    Ln = _normalize(L)

    sal = np.zeros(L.shape, dtype=float)

    for s in sigmas:
        # center-surround (DoG) per channel
        cL = gaussian_filter(L, s) - gaussian_filter(L, 2 * s)
        cA = gaussian_filter(a, s) - gaussian_filter(a, 2 * s)
        cB = gaussian_filter(b, s) - gaussian_filter(b, 2 * s)

        # orientation conspicuity from Gabor energy on L*
        ori = np.zeros_like(Ln)
        for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            real, imag = gabor(Ln, frequency=gabor_freq, theta=theta)
            ori += np.hypot(real, imag)

        cm = np.abs(cL) + np.abs(cA) + np.abs(cB) + _normalize(ori)
        sal += _normalize(cm)

    return _normalize(sal)

# ------------------------- Main extractor -------------------------

@dataclass
class ExtractorParams:
    k_init: int = 12                 # large K for initial candidates
    bilateral_sigma_spatial: float = 2.0
    bilateral_sigma_color: float = 0.08  # in [0,1] units after scaling
    rag_merge_thresh: float = 8.0    # ΔE76 threshold for merging adjacent regions
    erosion_iterations: int = 1
    n_final: int = 5

    # ((new))
    wC: float = 1.0   # contrast weight
    wS: float = 1.0   # saturation weight
    wA: float = 1.0   # area weight
    pick_contrast_second: bool = False  # pick highest C as the second guaranteed slot

    # efficiency params
    kmeans_max_samples: int = 200_000
    use_minibatch: bool = True       # MiniBatchKMeans for speed

class DominantColorExtractor:
    def __init__(self, params: ExtractorParams = ExtractorParams()):
        self.p = params

    # ---- Step 1: bilateral filter in CIELAB ----
    def _bilateral_lab(self, img_rgb: np.ndarray) -> np.ndarray:
        img_rgb = img_as_float(img_rgb)
        lab = color.rgb2lab(img_rgb)
        unit = _lab_to_unit(lab)
        den = denoise_bilateral(
            unit, sigma_color=self.p.bilateral_sigma_color,
            sigma_spatial=self.p.bilateral_sigma_spatial,
            channel_axis=-1
        )
        lab_f = _unit_to_lab(den)
        return lab_f

    # ---- Step 2: K-means in CIELAB (large K) ----
    # def _kmeans_lab(self, lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     H, W, _ = lab.shape
    #     X = lab.reshape(-1, 3)
    #     km = KMeans(n_clusters=self.p.k_init, n_init=10, random_state=0)
    #     labels = km.fit_predict(X)
    #     centers = km.cluster_centers_
    #     return labels.reshape(H, W), centers

    def _kmeans_lab(self, lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W, _ = lab.shape
        X = lab.reshape(-1, 3).astype(np.float32)

        # Sample for fitting
        n = X.shape[0]
        if self.p.kmeans_max_samples and n > self.p.kmeans_max_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, self.p.kmeans_max_samples, replace=False)
            X_fit = X[idx]
        else:
            X_fit = X

        if self.p.use_minibatch:
            km = MiniBatchKMeans(
                n_clusters=self.p.k_init, random_state=0, batch_size=4096, n_init="auto"
            )
            km.fit(X_fit)
            labels = km.predict(X)  # vectorized; still fast
            centers = km.cluster_centers_.astype(np.float32)
        else:
            km = KMeans(n_clusters=self.p.k_init, n_init=10, random_state=0)
            km.fit(X_fit)
            labels = km.predict(X)
            centers = km.cluster_centers_.astype(np.float32)

        return labels.reshape(H, W), centers


    # ---- Step 3: enforce image-space coherency (RAG merge) ----
    def _rag_merge(self, lab: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a RAG with mean colors and merge adjacent regions whose
        CIELAB distance is below rag_merge_thresh (approximation of 'graph cut').
        """
        lab    = _as_c_float64(lab)
        labels = _as_c_int32(labels)
        # RAG edge weights are mean-color distances (mode='distance')
        rag = graph.rag_mean_color(lab, labels, mode='distance')
        # cut_threshold keeps edges with weight >= thresh; we want to MERGE similar regions
        # so we set thresh to the desired max distance and then label the low-weight components.
        merged = graph.cut_threshold(labels, rag, self.p.rag_merge_thresh)
        merged = _as_c_int32(merged)

        # Compute new centers after merge:
        uniq = _safe_unique_int(merged)
        new_centers = np.empty((len(uniq), 3), dtype=np.float64)
        for i, lbl in enumerate(uniq):
            m = (merged == int(lbl))
            new_centers[i] = lab[m].mean(axis=0)
        return merged, new_centers

    # ---- Step 4: prune boundary 'mixing' clusters by erosion ----
    def _prune_boundary_clusters(self, labels: np.ndarray):
        labels = _as_c_int32(labels)
        kept: List[int] = []

        selem = disk(1)
        for lbl in _safe_unique_int(labels):
            lbl = int(lbl)
            mask = np.asarray(labels == lbl, dtype=bool, order="C")
            eroded = mask
            for _ in range(self.p.erosion_iterations):
                eroded = binary_erosion(eroded, selem)
            if eroded.any():
                kept.append(lbl)

        if not kept:
            vals, counts = np.unique(labels, return_counts=True)
            kept = [int(vals[int(np.argmax(counts))])]

        remap = {old: i for i, old in enumerate(kept)}
        new_labels = np.full(labels.shape, -1, dtype=np.int32)
        for old, i in remap.items():
            new_labels[labels == old] = i
        if (new_labels < 0).any():
            new_labels[new_labels < 0] = 0
        return new_labels, kept

    # ---- Step 5: compute color features C, A, S and integrative p ----
    def _compute_features(self, lab: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> Dict[str, np.ndarray]:
        sal = itti_saliency_lab(lab)  # C(i,j) ∈ [0,1]
        H, W = labels.shape
        K = int(labels.max()) + 1

        # Area Ak (raw counts), Contrast Ck (mean saliency), Saturation Sk (||[a*,b*]||)
        Ak_raw = np.zeros(K, dtype=float)
        Ck_raw = np.zeros(K, dtype=float)
        Sk_raw = np.zeros(K, dtype=float)

        for k in range(K):
            mask = (labels == k)
            n = mask.sum()
            Ak_raw[k] = n
            Ck_raw[k] = sal[mask].mean() if n > 0 else 0.0
            a, b = centers[k, 1], centers[k, 2]
            Sk_raw[k] = np.sqrt(a * a + b * b)

        # Normalize each feature by its max across candidates
        Ak = Ak_raw / (Ak_raw.max() + 1e-12)
        Ck = Ck_raw / (Ck_raw.max() + 1e-12)
        Sk = Sk_raw / (Sk_raw.max() + 1e-12)

        # pk = Ck + Ak + Sk  # Eq. (5)
        # ((new))
        pk = self.p.wC * Ck + self.p.wS * Sk + self.p.wA * Ak

        return dict(A=Ak, C=Ck, S=Sk, p=pk)

    # ---- Step 6: palette selection with weights w(d,k) ----
    def _select_palette(self, centers: np.ndarray, features: Dict[str, np.ndarray], n_final: int) -> List[int]:
        K = centers.shape[0]
        remaining = list(range(K))
        selected: List[int] = []

        # ((new))
        wA, wC, wS = self.p.wA, self.p.wC, self.p.wS

        # Precompute hue angles and pairwise dispersions for σ_h, σ_c
        hues = np.array([_hue_angle_ab(c[1], c[2]) for c in centers])
        # For σ_h, σ_c, use differences across all pairs
        if K >= 2:
            dh_all = []
            dc_all = []
            for i in range(K):
                for j in range(i + 1, K):
                    dh_all.append(_circ_diff(hues[i], hues[j]))
                    dc_all.append(_deltaE76(centers[i], centers[j]))
            sigma_h = np.std(dh_all) + 1e-12
            sigma_c = np.std(dc_all) + 1e-12
        else:
            sigma_h = sigma_c = 1.0

        A = features["A"].copy()
        C = features["C"].copy()
        S = features["S"].copy()
        p = features["p"].copy()

        # 1) pick highest S
        first = remaining[int(np.argmax(S[remaining]))]
        selected.append(first)
        remaining.remove(first)

        # After each selection, reweight remaining features
        def reweight(d_idx: int):
            nonlocal A, C, S, p
            hd = hues[d_idx]
            cd = centers[d_idx]
            for k in remaining:
                dh = _circ_diff(hd, hues[k])
                dc = _deltaE76(cd, centers[k])
                # Paper Eqs. (7)-(9); we use squared diffs inside Gaussian
                w1 = 1.0 - np.exp(-(dh * dh) / (2.0 * sigma_h * sigma_h))
                w2 = 1.0 - np.exp(-(dc * dc) / (2.0 * sigma_c * sigma_c))
                w = w1 * w2
                A[k] *= w
                C[k] *= w
                S[k] *= w
                # p[k] = A[k] + C[k] + S[k]
                # ((new))
                p[k] = (wA * A[k]) + (wC * C[k]) + (wS * S[k])

        reweight(first)

        # # 2) pick largest area
        # if remaining:
        #     second = remaining[int(np.argmax(A[remaining]))]
        #     selected.append(second)
        #     remaining.remove(second)
        #     reweight(second)

        # ((new))
        # 2) pick second: contrast OR area depending on flag (CHANGED)
        if remaining:
            if getattr(self.p, "pick_contrast_second", False):
                second = remaining[int(np.argmax(C[remaining]))]   # highest contrast
            else:
                second = remaining[int(np.argmax(A[remaining]))]   # largest area (original behavior)
            selected.append(second)
            remaining.remove(second)
            reweight(second)

        # 3) pick by highest p until n_final
        while remaining and len(selected) < n_final:
            k_next = remaining[int(np.argmax(p[remaining]))]
            selected.append(k_next)
            remaining.remove(k_next)
            reweight(k_next)

        return selected

    # ---- Orchestrator ----
    def extract(self, rgb01) -> Dict[str, np.ndarray]:
        # ---- Step 1: bilateral filter in Lab ----
        lab_f = self._bilateral_lab(rgb01)

        # ---- Step 2: K-means in Lab ----
        labels_km, centers_km = self._kmeans_lab(lab_f)

        # Normalize dtypes/layouts before graph ops
        lab_f = _as_c_float64(lab_f)
        labels_km = _as_c_int32(labels_km)

        # ---- Step 3: RAG merge (image-space coherency) ----
        labels_merged, centers_merged = self._rag_merge(lab_f, labels_km)
        labels_merged = _as_c_int32(labels_merged)
        centers_merged = _as_c_float64(centers_merged)

        # Remap merged labels to consecutive ids [0..K-1] robustly
        uniq = _safe_unique_int(labels_merged).astype(labels_merged.dtype, copy=False)
        labels_merged = np.searchsorted(uniq, labels_merged)  # fast, contiguous ids
        labels_merged = _as_c_int32(labels_merged)

        # Recompute centers after remap
        centers_merged = np.array(
            [lab_f[labels_merged == i].mean(axis=0) for i in range(len(uniq))],
            dtype=np.float64
        )

        # ---- Step 4: prune boundary/mixing clusters via erosion ----
        labels_pruned, kept_ids = self._prune_boundary_clusters(labels_merged)
        labels_pruned = _as_c_int32(labels_pruned)

        # Defensive: eliminate any residual negatives (shouldn't happen)
        if (labels_pruned < 0).any():
            labels_pruned[labels_pruned < 0] = 0

        # ---- Step 5: features over candidates ----
        final_ids = _safe_unique_int(labels_pruned).astype(int).tolist()
        centers_final = np.array(
            [lab_f[labels_pruned == i].mean(axis=0) for i in final_ids],
            dtype=np.float64
        )

        feats = self._compute_features(lab_f, labels_pruned, centers_final)

        # ---- Step 6: palette selection ----
        sel_idx_local = self._select_palette(centers_final, feats, n_final=self.p.n_final)
        palette_lab = centers_final[sel_idx_local]

        # Normalize weights from integrative feature p
        p_selected = feats["p"][sel_idx_local].astype(float)
        weights = p_selected / (p_selected.sum() + 1e-12)
        order = np.argsort(-weights)
        palette_lab = palette_lab[order]
        weights = weights[order]

        # LAB -> RGB01 -> RGB255 -> HEX
        palette_rgb01 = np.clip(
            color.lab2rgb(palette_lab.reshape(1, -1, 3)).reshape(-1, 3),
            0, 1
        )
        palette_rgb255 = np.clip(np.rint(palette_rgb01 * 255), 0, 255).astype(int)

        # Build HEX strings
        centers_hex = [
            "#" + "".join(f"{int(v):02X}" for v in row) for row in palette_rgb255
        ]
        # Round LAB
        palette_lab_round = np.round(palette_lab.astype(float), 3)

        # Standardized output
        return [
            {
                "hex": centers_hex[i],
                "rgb": palette_rgb255[i].tolist(),
                "lab": palette_lab_round[i].tolist(),
                "weight": round(float(weights[i]), 3),
            }
            for i in range(len(centers_hex))
        ]

