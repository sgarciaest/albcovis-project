import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from albcovis.services.color_extraction import dominant_colors_kmeans, dominant_colors_colorthief, prominent_colors_cfdc, plot_color_swatch
# from albcovis.services.cfdc import  DominantColorExtractor, ExtractorParams
from albcovis.utils.img import limit_image_size

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SRC_DIR = Path("data/source/images")
OUT_DIR = Path("data/processed/dominant_color_extraction_methods")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_COLORS = 5
METHOD_LABELS = ["K-Means (Lab)", "MMCQ (ColorThief)", "CFDC (Chang & Mukai)"]

# ---------------------------------------------------------------------
# Helper: build one figure with image on the left and 3 swatch rows on right
# ---------------------------------------------------------------------
def build_and_save_panel(img_path):
    # Load + limit size
    img = Image.open(path).convert("RGB") # ensure rgb color space
    img = limit_image_size(img)
    rgb01 = pil_to_numpy01(img)

    # Run methods (each returns list of dicts with standardized keys)
    km = dominant_colors_kmeans(rgb01, k=N_COLORS)
    ct = dominant_colors_colorthief(str(img_path), color_count=N_COLORS)
    cfdc = prominent_colors_cfdc(rgb01)  
    # --- Matplotlib layout ---
    # Two columns: image (left), palettes (right) in 3 rows
    fig = plt.figure(figsize=(10, 6), dpi=120)  # feel free to tweak
    gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[1.2, 1.0], height_ratios=[1, 1, 1], wspace=0.15, hspace=0.3)

    # Left: the image spans all rows
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(rgb01)
    ax_img.set_axis_off()
    ax_img.set_title(img_path.name, fontsize=11, pad=6)

    # Right: three rows of swatches, each labeled
    methods_data = [km, ct, cfdc]
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
            tile_aspect=4.0,
            ax=ax
        )
        ax.set_title(label, fontsize=10, pad=4)

    # Save and close
    out_file = OUT_DIR / f"{img_path.stem}_palettes_hue.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
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
