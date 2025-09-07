import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from albcovis.services.color_extraction import plot_color_swatch
from albcovis.services.cfdc import  DominantColorExtractor, ExtractorParams
from albcovis.utils.img import limit_image_size

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SRC_DIR = Path("data/source/images")
OUT_DIR = Path("data/processed/cfdc_params")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_COLORS = 5
METHOD_LABELS = [
    "Initial Default",
    "Prominent+", "Prominent++", "Prominent+++",
    "Prominent++ +A-C A2nd", "Prominent+++ +A-C A2nd",
    "Prominent+++ eq weights A2nd",
    "Prominent++ +A-C A2nd -C", "Prominent+++ +A-C A2nd -C"]

# First param config, default
extractor_1 = DominantColorExtractor(ExtractorParams(
    k_init=12,
    bilateral_sigma_spatial=2.0,
    bilateral_sigma_color=0.08,
    rag_merge_thresh=8.0,
    erosion_iterations=1,
    n_final=N_COLORS,
    wC = 1.0,
    wS = 1.0,
    wA = 1.0,
    pick_contrast_second=False
))

# Start playing with weights, lower area importance. Increment k_init to start with a larger palette of candidates
extractor_2 = DominantColorExtractor(ExtractorParams(
    k_init=16,
    bilateral_sigma_spatial=1.5,
    bilateral_sigma_color=0.06,
    rag_merge_thresh=6.0,
    erosion_iterations=1,
    n_final=N_COLORS,
    wC = 1.2,
    wS = 1.1,
    wA = 0.7,
    pick_contrast_second=True
))
extractor_3 = DominantColorExtractor(ExtractorParams(
    k_init=18,
    bilateral_sigma_spatial=1.25,
    bilateral_sigma_color=0.06,
    rag_merge_thresh=6.0,
    erosion_iterations=1,
    n_final=N_COLORS,
    wC = 1.2,
    wS = 1.1,
    wA = 0.7,
    pick_contrast_second=True
))
extractor_4 = DominantColorExtractor(ExtractorParams(
    k_init=20,
    bilateral_sigma_spatial=1.0,
    bilateral_sigma_color=0.04,
    rag_merge_thresh=4.0,
    erosion_iterations=1,
    n_final=N_COLORS,
    wC = 1.2,
    wS = 1.1,
    wA = 0.7,
    pick_contrast_second=True
))

# Adjust weights to rest importance to area and contrast. Don't pick contrast as second
extractor_5 = DominantColorExtractor(ExtractorParams(
    k_init=18,
    bilateral_sigma_spatial=1.25,
    bilateral_sigma_color=0.06,
    rag_merge_thresh=6.0,
    erosion_iterations=1,
    n_final=N_COLORS,
    wC = 1.1,
    wS = 1.1,
    wA = 0.8,
    pick_contrast_second=False
))
extractor_6 = DominantColorExtractor(ExtractorParams(
    k_init=20,
    bilateral_sigma_spatial=1.0,
    bilateral_sigma_color=0.04,
    rag_merge_thresh=4.0,
    erosion_iterations=1,
    n_final=N_COLORS,
    wC = 1.1,
    wS = 1.1,
    wA = 0.8,
    pick_contrast_second=False
))

# Return to equal weights, increment bilateral_sigma_spatial bilateral_sigma_color rag_merge_thresh
extractor_7 = DominantColorExtractor(ExtractorParams(
    k_init=20,
    bilateral_sigma_spatial=1.25,
    bilateral_sigma_color=0.06,
    rag_merge_thresh=6.0,
    erosion_iterations=1,
    n_final=N_COLORS,
    wC = 1.0,
    wS = 1.0,
    wA = 1.0,
    pick_contrast_second=False
))

# Readjust weights, try less importance to contrast
extractor_8 = DominantColorExtractor(ExtractorParams(
    k_init=18,
    bilateral_sigma_spatial=1.25,
    bilateral_sigma_color=0.06,
    rag_merge_thresh=6.0,
    erosion_iterations=1,
    n_final=N_COLORS,
    wC = 0.9,
    wS = 1.1,
    wA = 0.8,
    pick_contrast_second=False
))
extractor_9 = DominantColorExtractor(ExtractorParams(
    k_init=20,
    bilateral_sigma_spatial=1.25,
    bilateral_sigma_color=0.06,
    rag_merge_thresh=6.0,
    erosion_iterations=1,
    n_final=N_COLORS,
    wC = 0.9,
    wS = 1.1,
    wA = 0.8,
    pick_contrast_second=False
))



# ---------------------------------------------------------------------
# Helper: build one figure with image on the left and 3 swatch rows on right
# ---------------------------------------------------------------------
def build_and_save_panel(img_path):
    # Load + limit size (utility)
    img = Image.open(img_path).convert("RGB")
    img = limit_image_size(img)
    img_np = np.asarray(img, dtype=np.uint8)

    out_1 = extractor_1.extract(str(img_path))
    out_2 = extractor_2.extract(str(img_path))
    out_3 = extractor_3.extract(str(img_path))
    out_4 = extractor_4.extract(str(img_path))
    out_5 = extractor_5.extract(str(img_path))
    out_6 = extractor_6.extract(str(img_path))
    out_7 = extractor_7.extract(str(img_path))
    out_8 = extractor_8.extract(str(img_path))
    out_9 = extractor_9.extract(str(img_path))

    # --- Matplotlib layout ---
    # Two columns: image (left), palettes (right) in 3 rows
    fig = plt.figure(figsize=(10, 6), dpi=120)  # feel free to tweak
    gs = fig.add_gridspec(nrows=9, ncols=2, width_ratios=[1.2, 1.0], height_ratios=[0.75]*9, wspace=0.2, hspace=0.6)

    # Left: the image spans all rows
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img_np)
    ax_img.set_axis_off()
    ax_img.set_title(img_path.name, fontsize=11, pad=6)

    # Right: three rows of swatches, each labeled
    methods_data = [out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9]
    for row, (label, colors) in enumerate(zip(METHOD_LABELS, methods_data)):
        ax = fig.add_subplot(gs[row, 1])
        # Order by hue for comparability
        plot_color_swatch(
            colors,
            orientation="horizontal",
            sort_by="weight",
            show_labels=False,
            normalize_weights=False,  # equal-width tiles so order is the only varying factor
            gap=0.02,
            tile_aspect=1.0,
            ax=ax,
            show_percent=True
        )
        ax.set_title(label, fontsize=10, pad=4)

    # Save and close
    out_file = OUT_DIR / f"{img_path.stem}__cfdc_palettes.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
import traceback
if __name__ == "__main__":

    # Collect images (sorted)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    files = sorted([p for p in SRC_DIR.iterdir() if p.suffix.lower() in exts])

    if not files:
        print(f"No images found in: {SRC_DIR.resolve()}")
    else:
        print(f"Found {len(files)} images in: {SRC_DIR.resolve()}")

    for i, path in enumerate(files, start=1):
        try:
            print(f"[{i}/{len(files)}] Processing {path.name} ... ", end="", flush=True)
            out_path = build_and_save_panel(path)
            print(f"saved → {out_path.relative_to(OUT_DIR.parent)}")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
