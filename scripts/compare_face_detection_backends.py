import json
import time
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from deepface import DeepFace
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ------------------------------- CONFIG --------------------------------
SRC_DIR = Path("data/source/images")
GT_COCO = Path("data/source/dev_set_annotations_via_cocoformat.json")
OUT_DIR = Path("data/processed/face_detection_backends")
IMG_OUT_DIR = OUT_DIR / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)

backends = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface',
    'yunet', 'centerface', 'yolov8', 'yolov11s', 'yolov11n', 'yolov11m'
]
exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# ------------------------- UTILS --------------------------------------
def build_filename_to_id_map(coco_json_path):
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
    return {
        img["file_name"]: int(img["file_name"].split("-")[0])
        for img in coco_data["images"]
    }

def compute_iou(box1, box2):
    eps = 1e-12
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Optional guards
    if w1 < 0 or h1 < 0 or w2 < 0 or h2 < 0:
        raise ValueError("w and h must be non-negative")

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    return inter_area / (union + eps) if union > 0 else 0.0


# ----------------- DEEPFACE DETECTION PHASE ---------------------------
def generate_predictions_for_backend(images_dir, coco_gt_json, output_json_path, backend):
    print(f"\n🔍 Running DeepFace detection for backend: {backend}")
    filename_to_id = build_filename_to_id_map(coco_gt_json)
    predictions = []

    image_paths = sorted([p for p in Path(images_dir).glob("*") if p.suffix.lower() in exts])
    gt = COCO(str(coco_gt_json))

    times = []
    for img_path in tqdm(image_paths, desc=f"[{backend}] Detecting"):
        fname = img_path.name
        if fname not in filename_to_id:
            continue
        image_id = filename_to_id[fname]

        try:
            start = time.time()
            detections = DeepFace.extract_faces(str(img_path), detector_backend=backend, enforce_detection=False)
            elapsed = time.time() - start
        except Exception:
            detections = []
            elapsed = None
        if elapsed:
            times.append(elapsed)

        # Load GT boxes for IoU computation
        gt_anns = gt.loadAnns(gt.getAnnIds(imgIds=image_id))
        gt_boxes = [ann["bbox"] for ann in gt_anns if ann["category_id"] == 1]

        for det in detections:
            fa = det.get("facial_area", {})
            x, y, w, h = fa.get("x"), fa.get("y"), fa.get("w"), fa.get("h")
            score = det.get("confidence", 1.0)
            iou = max([compute_iou((x, y, w, h), gt_box) for gt_box in gt_boxes], default=0.0)

            if None not in (x, y, w, h):
                predictions.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "score": score,
                    "iou": iou
                })

    with open(output_json_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"✅ Saved {len(predictions)} detections to {output_json_path.name}")
    return np.mean(times), output_json_path

# ---------------------- METRIC EVALUATION -----------------------------
def run_coco_eval(ground_truth_json, category_id, prediction_json):
    coco_gt = COCO(ground_truth_json)
    coco_dt = coco_gt.loadRes(prediction_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.catIds = [category_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = coco_eval.stats.tolist()
    return metrics  # list of 12 COCO metrics

# --------------------- VISUALIZATION PANEL ----------------------------
def build_and_save_panel(path, all_detections):
    from pycocotools.coco import COCO

    img = Image.open(path).convert("RGB")
    image_id = int(path.name.split("-")[0])

    n = len(backends) + 1  # +1 for ground truth
    n_cols = (n + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), squeeze=False)

    # Load GT boxes once
    gt = COCO(str(GT_COCO))
    gt_anns = gt.loadAnns(gt.getAnnIds(imgIds=image_id))
    gt_boxes = [ann["bbox"] for ann in gt_anns if ann["category_id"] == 1]

    # ------------------ Ground Truth Panel ------------------
    ax = axes[0, 0]
    ax.imshow(img)
    ax.axis("off")
    for x, y, w, h in gt_boxes:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="blue", facecolor="none")
        ax.add_patch(rect)
    ax.set_title("ground truth", fontsize=10)

    # ------------------ Detection Panels -------------------
    for j, backend in enumerate(backends, start=1):  # start from subplot 1
        row, col = divmod(j, n_cols)
        ax = axes[row, col]
        ax.imshow(img)
        ax.axis("off")

        dets = [d for d in all_detections[backend] if d["image_id"] == image_id]

        # Plot ground truth in thin blue (reference)
        for x, y, w, h in gt_boxes:
            rect = patches.Rectangle((x, y), w, h, linewidth=0.9, edgecolor="blue", facecolor="none", linestyle="--")
            ax.add_patch(rect)

        # Plot detections if any
        if dets:
            for d in dets:
                x, y, w, h = d["bbox"]
                score = d.get("score", 1.0)
                iou = d.get("iou", 0.0)
                rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor="red", facecolor="none")
                ax.add_patch(rect)
                ax.text(x, y - 5, f"{score:.2f} | IoU {iou:.2f}", color="limegreen", fontsize=7, weight="bold")

        ax.set_title(f"{backend}", fontsize=10)

    out_file = IMG_OUT_DIR / f"{path.stem}_face_detection_backends.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file



# ------------------------- MAIN SCRIPT -------------------------------
if __name__ == "__main__":
    print("🚀 Starting full pipeline...")
    results = []
    all_detections = {}

    # Run detection or load existing
    for backend in backends:
        json_path = OUT_DIR / f"detections_{backend}.json"
        if not json_path.exists():
            mean_time, _ = generate_predictions_for_backend(SRC_DIR, GT_COCO, json_path, backend)
        else:
            print(f"📁 Found cached detections: {json_path.name}")
            mean_time = None

        with open(json_path) as f:
            all_detections[backend] = json.load(f)

        metrics = run_coco_eval(GT_COCO, category_id=1, prediction_json=str(json_path))
        results.append([backend, mean_time] + metrics)

    # Save metrics summary
    columns = ["backend", "mean_time"] + [
        "AP@[IoU=0.50:0.95]",
        "AP@0.50", "AP@0.75",
        "AP_small", "AP_medium", "AP_large",
        "AR@1", "AR@10", "AR@100",
        "AR_small", "AR_medium", "AR_large"
    ]
    df = pd.DataFrame(results, columns=columns)
    print("\n📊 Evaluation Summary:")
    print(df.round(3))
    df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    # Build panels
    print("\n🖼️ Generating visual panels...")
    files = sorted([p for p in SRC_DIR.iterdir() if p.suffix.lower() in exts])
    for path in tqdm(files, desc="Building panels"):
        try:
            build_and_save_panel(path, all_detections)
        except Exception as e:
            print(f"Error on {path.name}: {e}")
            traceback.print_exc()

    print("✅ All done.")

