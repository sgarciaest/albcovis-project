from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from deepface import DeepFace
import numpy as np
import time
import pandas as pd
from albcovis.utils.img import limit_image_size


SRC_DIR = Path("data/source/images")
OUT_DIR = Path("data/processed/face_detection_backends")
OUT_DIR.mkdir(parents=True, exist_ok=True)

backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'yunet',
    'centerface',
    'yolov8',
    'yolov11s',
    'yolov11n',
    'yolov11m',
    # 'fastmtcnn',
    # 'mediapipe',
]

def build_and_save_panel(path):
    times = {}

    img = Image.open(path).convert("RGB")
    img = limit_image_size(img)

    n = len(backends)
    n_cols = (n + 1) // 2  # split into 2 rows
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 10), squeeze=False)


    for j, b in enumerate(backends):
        row, col = divmod(j, n_cols)
        ax = axes[row, col]

        try:
            start = time.time()
            faces = DeepFace.extract_faces(str(path), detector_backend=b, enforce_detection=False)
            elapsed = time.time() - start
        except Exception as e:
            faces = []
            elapsed = None
            print(f"Error with {b} backend: {type(e).__name__}: {e}")

        times[b] = elapsed

        ax.imshow(img)
        ax.axis("off")

        # Draw faces if detected
        for f in faces:
            fa = f["facial_area"]
            x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
            rect = patches.Rectangle((x, y), w, h, linewidth=1.3, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            conf = f.get("confidence")
            if conf is not None:
                ax.text(x, y - 5, f"{conf:.2f}", color="limegreen", fontsize=8, weight="bold")

        # Title with backend + time
        if elapsed is not None:
            ax.set_title(f"{b}\n{elapsed:.2f}s", fontsize=10)
        else:
            ax.set_title(f"{b}\nError", fontsize=10)

    out_file = OUT_DIR / "images" / f"{path.stem}_face_detection_backends.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file, times

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
import traceback
if __name__ == "__main__":
    times_aggregated = {}

    # Collect images (sorted)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    files = sorted([p for p in SRC_DIR.iterdir() if p.suffix.lower() in exts])
    files = files[0:3]

    if not files:
        print(f"No images found in: {SRC_DIR.resolve()}")
    else:
        print(f"Found {len(files)} images in: {SRC_DIR.resolve()}")

    for i, path in enumerate(files, start=1):
        try:
            print(f"[{i}/{len(files)}] Processing {path.name} ... ", end="", flush=True)
            out_path, times = build_and_save_panel(path)
            times_aggregated[path] = times
            print(f"saved → {out_path.relative_to(OUT_DIR.parent)}")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()

    times_aggregated_df = pd.DataFrame.from_dict(times_aggregated, orient="index")
    print("Execution time statistics for each backend:")
    print(times_aggregated_df.describe())
    times_aggregated_df.to_csv(OUT_DIR / "metrics_summary.csv")
