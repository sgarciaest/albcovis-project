from __future__ import annotations

from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage import color  # rgb2lab, lab2rgb
from albcovis.utils.img import limit_image_size

# ---- Dominant colors extraction with k-means and Lab color space ----

def dominant_colors_kmeans(path: str, k=5, sample=400_000, seed=42):
    # Load and downsample if needed
    img = Image.open(path).convert("RGB") # ensure rgb color space
    img = limit_image_size(img)

    # sRGB [0,1] -> Lab (D65) using scikit-image
    rgb = np.asarray(img, dtype=np.float32) / 255.0 # H×W×3 shape array (3d)
    lab = color.rgb2lab(rgb).reshape(-1, 3).astype(np.float32) # Nx3 shape array (2d, pixel list, each pixel a row, features the Lab color space)

    # Optional random sampling
    if sample and lab.shape[0] > sample:
        idx = np.random.default_rng(seed).choice(lab.shape[0], sample, replace=False)
        lab_sample = lab[idx]
    else:
        lab_sample = lab

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init="auto")
    # Fit on a (possibly) subset of pixels 
    kmeans.fit(lab_sample)
    # Assign computed centroids to all pixels
    labels = kmeans.predict(lab)
    # How many pixels fell into each cluster, and compute weithts accordingly
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    weights = counts / counts.sum()

    # Convert centers back to sRGB
    centers_lab = kmeans.cluster_centers_.astype(np.float32) # (k, 3) shape
    # Convert to rgb via a temporary leading axis, then drop it. Because lab2rgb expect HxWx3
    # Lab -> RGB in float; skimage uses float math, so we use float64 then back float32
    centers_rgb = color.lab2rgb(centers_lab[np.newaxis, :, :].astype(np.float64))[0] # (k, 3), in [0,1]
    centers_rgb255 = np.clip(np.rint(centers_rgb * 255), 0, 255).astype(int)  # (k, 3), in ints
    # Hex from rgb
    centers_hex = ["#" + "".join(f"{int(round(c*255)):02X}" for c in row) for row in centers_rgb]

    order = np.argsort(-weights)
    # drop zero-weight clusters (very rare, but better)
    order = [i for i in order if weights[i] > 0.0]

    # return [(centers_hex[i], float(weights[i])) for i in order]
    return [
        {
            "hex": centers_hex[i],
            "rgb": centers_rgb255[i].tolist(),                          # [R, G, B] in 0–255 ints
            "lab": np.round(centers_lab[i].astype(float), 3).tolist(),  # [L*, a*, b*] rounded
            "weight": round(float(weights[i]), 3)
        }
        for i in order
    ]

# ------ Dominant colors extraction with MMCQ through colorthief ------
from colorthief import ColorThief
import numpy as np
from skimage import color  # rgb2lab

def dominant_colors_colorthief(path: str, color_count=5, quality=5):
    ct = ColorThief(path)
    colors_rgb = ct.get_palette(color_count=color_count, quality=quality) or []
    if not colors_rgb:
        return []

    # Convert to ndarray for vectorized processing
    rgb_arr = np.asarray(colors_rgb, dtype=np.float32)                 # (k, 3) in 0–255
    rgb01 = rgb_arr / 255.0                                            # (k, 3) in 0–1

    # sRGB -> Lab using scikit-image; add a leading axis because rgb2lab expects HxWx3-like shapes
    lab = color.rgb2lab(rgb01[np.newaxis, :, :])[0].astype(np.float32) # (k, 3)
    lab_round = np.round(lab.astype(float), 3)

    # Build hex strings
    rgb_int = rgb_arr.astype(int)
    hex_list = [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in rgb_int]

    # Assemble output to match k-means function schema
    out = []
    for hx, (r, g, b), (L, a, b2) in zip(hex_list, rgb_int.tolist(), lab_round.tolist()):
        out.append({
            "hex": hx,
            "rgb": [int(r), int(g), int(b)],
            "lab": [float(L), float(a), float(b2)],
            "weight": None  # no weights from ColorThief
        })
    return out

# ---------------------------------------- CFDC ----------------------------------------
from albcovis.services.cfdc import DominantColorExtractor, ExtractorParams

def prominent_colors_cfdc(path: str, n_final=5):
    extractor_ = DominantColorExtractor(ExtractorParams(
        k_init=20,
        bilateral_sigma_spatial=1.25,
        bilateral_sigma_color=0.06,
        rag_merge_thresh=6.0,
        erosion_iterations=1,
        n_final=n_final,
        wC = 0.9,
        wS = 1.1,
        wA = 0.8,
        pick_contrast_second=False
    ))
    out = extractor_.extract(path)
    return out

# --------------------------------- Orchestrator Layer ---------------------------------
def extract_colors(path: str):
    dominant_colors = dominant_colors_kmeans(path)
    prominent_colors = prominent_colors_cfdc(path)
    return {
        "dominant_colors": dominant_colors,
        "prominent_colors": prominent_colors
    }

# ------------------------------------ Plot Utility ------------------------------------
import math
from typing import Iterable, Sequence, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from colorsys import rgb_to_hsv

def plot_color_swatch(
    colors: Iterable[dict],
    orientation: str = "horizontal",
    sort_by: Optional[str] = None,           # {"weight","hue"} or None
    show_labels: bool = False,
    label_fields: Sequence[str] = ("hex",),
    show_percent: bool = False,
    normalize_weights: bool = False,
    gap: float = 0.02,                        # fraction of axis length between tiles
    tile_aspect: float = 2.0,                 # width/height if horizontal; height/width if vertical
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot a color swatch from extracted colors. Expects items with keys:
    - "hex": str like "#RRGGBB"
    - "rgb": [R,G,B] ints in 0..255
    - "lab": [L*, a*, b*] (unused for plotting)
    - "weight": float or None (prominence or area, depending on method)
    """
    colors = list(colors)
    n = len(colors)
    if n == 0:
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 1))
        ax.axis("off")
        return ax

    # --- Get sRGB in 0..1, weights (missing -> 0) ---
    rgb01 = []
    weights = []
    for c in colors:
        if "rgb" in c and c["rgb"] is not None:
            r, g, b = c["rgb"]
            rgb01.append([r/255.0, g/255.0, b/255.0])
        elif "hex" in c and isinstance(c["hex"], str) and len(c["hex"]) == 7:
            hx = c["hex"]
            r = int(hx[1:3], 16) / 255.0
            g = int(hx[3:5], 16) / 255.0
            b = int(hx[5:7], 16) / 255.0
            rgb01.append([r, g, b])
        else:
            rgb01.append([0.5, 0.5, 0.5])  # fallback gray

        w = c.get("weight", 0.0)
        try:
            w = float(0.0 if w is None else w)
        except Exception:
            w = 0.0
        weights.append(w)

    rgb01 = np.asarray(rgb01, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # --- Sorting ---
    order = np.arange(n)
    if sort_by == "weight":
        order = np.argsort(-weights)
    elif sort_by == "hue":
        # Convert to HSV hue on sRGB (0..1)
        hues = np.array([rgb_to_hsv(*rgb01[i]) [0] for i in range(n)], dtype=float)
        order = np.argsort(hues)  # ascending hue

    rgb01 = rgb01[order]
    weights = weights[order]
    colors = [colors[i] for i in order]

    # --- Tile sizes (proportions) ---
    any_w = np.any(weights > 0)
    if normalize_weights and any_w:
        props = weights / (weights.sum() + 1e-12)
    else:
        props = np.full(n, 1.0 / n, dtype=float)

    # account for total gap space (n-1 gaps)
    total_gap = gap * max(n - 1, 0)
    usable = max(1.0 - total_gap, 1e-9)
    sizes = props * usable  # each tile length along the main axis

    # --- Figure / axes ---
    if ax is None:
        if figsize is None:
            # Aesthetic defaults based on orientation, tile_aspect, and count
            if orientation == "horizontal":
                # height = 1 unit; width scales with tile_aspect * n (bounded)
                W = max(2.5, min(12.0, tile_aspect * max(n, 3) * 0.45))
                H = max(0.8, W / max(tile_aspect * max(n, 3), 3 * tile_aspect))
                figsize = (W, H)
            else:
                H = max(2.5, min(12.0, tile_aspect * max(n, 3) * 0.45))
                W = max(0.8, H / max(tile_aspect * max(n, 3), 3 * tile_aspect))
                figsize = (W, H)
        _, ax = plt.subplots(figsize=figsize)
    else:
        if figsize is not None:
            ax.figure.set_size_inches(*figsize, forward=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Draw tiles ---
    # coordinate convention: we fill [0,1] in the main axis with gaps between tiles
    def text_color_for_bg(rgb):
        # simple luma heuristic in sRGB space
        r, g, b = rgb
        luma = 0.2126*r + 0.7152*g + 0.0722*b
        return (0,0,0) if luma > 0.6 else (1,1,1)

    cursor = 0.0
    for i in range(n):
        if orientation == "horizontal":
            x0 = cursor
            w = sizes[i]
            y0 = 0.0
            h = 1.0
            rect = Rectangle((x0, y0), w, h, facecolor=rgb01[i], edgecolor="none")
            ax.add_patch(rect)

            if show_labels:
                parts = []
                for fld in label_fields:
                    val = colors[i].get(fld, None)
                    if val is None:
                        continue
                    parts.append(str(val))
                if show_percent and normalize_weights and any_w:
                    parts.append(f"{props[i]*100:.1f}%")
                if parts:
                    ax.text(
                        x0 + w/2, 0.5, "\n".join(parts),
                        ha="center", va="center",
                        fontsize=10, color=text_color_for_bg(rgb01[i]),
                        family="DejaVu Sans", clip_on=False
                    )

            cursor = x0 + w + (gap if i < n-1 else 0.0)

        else:  # vertical
            y0 = cursor
            h = sizes[i]
            x0 = 0.0
            w = 1.0
            rect = Rectangle((x0, y0), w, h, facecolor=rgb01[i], edgecolor="none")
            ax.add_patch(rect)

            if show_labels:
                parts = []
                for fld in label_fields:
                    val = colors[i].get(fld, None)
                    if val is None:
                        continue
                    parts.append(str(val))
                if show_percent and normalize_weights and any_w:
                    parts.append(f"{props[i]*100:.1f}%")
                if parts:
                    ax.text(
                        0.5, y0 + h/2, "\n".join(parts),
                        ha="center", va="center",
                        fontsize=10, color=text_color_for_bg(rgb01[i]),
                        family="DejaVu Sans", clip_on=False
                    )

            cursor = y0 + h + (gap if i < n-1 else 0.0)

    return ax
