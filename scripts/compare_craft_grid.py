import json
import time
import itertools
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from shapely.geometry import Polygon
from craft_text_detector import load_craftnet_model, load_refinenet_model, get_prediction
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ------------------------------- CONFIG --------------------------------
SRC_DIR = Path("data/source/images")
GT_COCO = Path("data/source/dev_set_annotations_via_cocoformat.json")
OUT_DIR = Path("data/processed/craft_grid")
IMG_OUT_DIR = OUT_DIR / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_ID = 2  # category_id for text
MIN_IOU_THRESHOLD = 0.1
exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Param grid
param_grid = list(itertools.product(
    [0.9, 0.7, 0.5, 0.3],    # text_threshold
    [0.6, 0.4, 0.2],         # link_threshold
    [0.4],                   # low_text
    [1280, 800, 500]         # long size
))
param_names = [
    f"text={tt}_link={lt}_low={ltx}_size={ls}"
    for tt, lt, ltx, ls in param_grid
]

# ------------------------- UTILS --------------------------------------

def build_filename_to_id_map(coco_json_path):
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
    return {
        img["file_name"]: int(img["file_name"].split("-")[0])
        for img in coco_data["images"]
    }

def polygon_iou(poly1, poly2):
    try:
        p1 = Polygon(poly1).buffer(0)
        p2 = Polygon(poly2).buffer(0)
        if not p1.is_valid or not p2.is_valid:
            return 0.0
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        return inter / union if union > 0 else 0.0
    except:
        return 0.0

def polygon_to_bbox(poly):
    xs, ys = zip(*poly)
    x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    return [x, y, w, h]

def extract_ap(coco_eval, iou_threshold, category_id=0):
    iou_thrs = coco_eval.params.iouThrs
    area_idx = coco_eval.params.areaRngLbl.index("all")
    maxdet_idx = coco_eval.params.maxDets.index(100)

    try:
        iou_idx = np.where(np.isclose(iou_thrs, iou_threshold))[0][0]
        cat_idx = coco_eval.params.catIds.index(category_id)
    except (IndexError, ValueError):
        return -1.0

    precisions = coco_eval.eval['precision']
    values = precisions[iou_idx, :, cat_idx, area_idx, maxdet_idx]
    values = values[values > -1]
    return np.mean(values) if values.size > 0 else -1.0

def extract_ar(coco_eval, iou_threshold, category_id=0):
    iou_thrs = coco_eval.params.iouThrs
    area_idx = coco_eval.params.areaRngLbl.index("all")
    maxdet_idx = coco_eval.params.maxDets.index(100)

    try:
        iou_idx = np.where(np.isclose(iou_thrs, iou_threshold))[0][0]
        cat_idx = coco_eval.params.catIds.index(category_id)
    except (IndexError, ValueError):
        return -1.0

    recalls = coco_eval.eval['recall']
    value = recalls[iou_idx, cat_idx, area_idx, maxdet_idx]
    return value if value > -1 else -1.0



# ----------------- CRAFT DETECTION PHASE ---------------------------

def generate_predictions_for_params(images_dir, coco_gt_json, output_json_path,
                                    text_threshold, link_threshold, low_text, long_size=500):
    print(f"\n🔍 Running CRAFT text detection: text={text_threshold}, link={link_threshold}, low={low_text}")
    filename_to_id = build_filename_to_id_map(coco_gt_json)
    predictions = []

    image_paths = sorted([p for p in Path(images_dir).glob("*") if p.suffix.lower() in exts])
    gt = COCO(str(coco_gt_json))

    craft_net = load_craftnet_model(cuda=False)
    refine_net = load_refinenet_model(cuda=False)

    times = []

    for img_path in tqdm(image_paths, desc=f"[CRAFT]"):
        fname = img_path.name
        if fname not in filename_to_id:
            continue
        image_id = filename_to_id[fname]

        image = np.array(Image.open(img_path).convert("RGB"))

        try:
            start = time.time()
            prediction_result = get_prediction(
                image=image,
                craft_net=craft_net,
                refine_net=refine_net,
                text_threshold=text_threshold,
                link_threshold=link_threshold,
                low_text=low_text,
                long_size=long_size,
                cuda=False
            )
            elapsed = time.time() - start
            times.append(elapsed)
        except Exception:
            continue

        detected_polys = prediction_result.get("polys", [])

        gt_anns = gt.loadAnns(gt.getAnnIds(imgIds=image_id, catIds=[CATEGORY_ID]))
        gt_polys = [ann["segmentation"][0] for ann in gt_anns]
        gt_polys = [[(gt_poly[i], gt_poly[i + 1]) for i in range(0, len(gt_poly), 2)] for gt_poly in gt_polys]

        for poly in detected_polys:
            poly = poly.tolist()
            bbox = polygon_to_bbox(poly)
            iou = max([polygon_iou(poly, gt_poly) for gt_poly in gt_polys], default=0.0)

            predictions.append({
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                "bbox": bbox,
                "segmentation": [list(np.array(poly).flatten())],
                "score": 1.0,
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
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')  # or 'bbox' if you're using boxes
    coco_eval.params.iouThrs = np.linspace(0.1, 0.95, 18)  # includes 0.10, 0.30, 0.50
    coco_eval.params.catIds = [category_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap_all = coco_eval.stats[0]  # AP@[0.10:0.95]
    ap_010 = extract_ap(coco_eval, 0.10, category_id)
    ap_030 = extract_ap(coco_eval, 0.30, category_id)
    ap_050 = extract_ap(coco_eval, 0.50, category_id)

    ar_all = coco_eval.stats[8]  # AR@[0.10:0.95] (from updated iouThrs)
    ar_010 = extract_ar(coco_eval, 0.10, category_id)
    ar_030 = extract_ar(coco_eval, 0.30, category_id)
    ar_050 = extract_ar(coco_eval, 0.50, category_id)

    return [ap_all, ap_010, ap_030, ap_050, ar_all, ar_010, ar_030, ar_050]


# --------------------- VISUALIZATION PANEL ----------------------------
def build_and_save_panel(path, all_detections):
    from pycocotools.coco import COCO

    img = Image.open(path).convert("RGB")
    image_id = int(path.name.split("-")[0])

    # n = len(all_detections) + 1
    # n_cols = 4
    # n_rows = (n + n_cols - 1) // n_cols
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    n = len(all_detections) + 1
    n_cols = 6
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    gt = COCO(str(GT_COCO))
    gt_anns = gt.loadAnns(gt.getAnnIds(imgIds=image_id, catIds=[CATEGORY_ID]))
    gt_polys = [ann["segmentation"][0] for ann in gt_anns]
    gt_polys = [[(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)] for poly in gt_polys]

    # Ground truth panel
    ax = axes[0][0]
    ax.imshow(img)
    ax.axis("off")
    for poly in gt_polys:
        patch = patches.Polygon(poly, linewidth=1.5, edgecolor="blue", facecolor="none")
        ax.add_patch(patch)
    ax.set_title("ground truth")

    # Detection panels
    for j, (param_name, dets) in enumerate(all_detections.items(), start=1):
        row, col = divmod(j, n_cols)
        ax = axes[row][col]
        ax.imshow(img)
        ax.axis("off")

        for poly_det in dets:
            if poly_det["image_id"] != image_id:
                continue
            poly = [(poly_det["segmentation"][0][i], poly_det["segmentation"][0][i + 1])
                    for i in range(0, len(poly_det["segmentation"][0]), 2)]
            iou = poly_det.get("iou", 0.0)
            patch = patches.Polygon(poly, linewidth=1.3, edgecolor="red", facecolor="none")
            ax.add_patch(patch)
            ax.text(poly[0][0], poly[0][1] - 5, f"IoU {iou:.2f}", color="lime", fontsize=8)

        # Also draw GT for reference
        for poly in gt_polys:
            patch = patches.Polygon(poly, linewidth=0.8, edgecolor="blue", linestyle="--", facecolor="none")
            ax.add_patch(patch)

        ax.set_title(param_name, fontsize=8)

    out_file = IMG_OUT_DIR / f"{path.stem}_craft_grid.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file

# ------------------------- MAIN SCRIPT -------------------------------
if __name__ == "__main__":
    print("🚀 Starting CRAFT grid test...")
    results = []
    all_detections = {}

    for (tt, lt, ltx, ls), param_name in zip(param_grid, param_names):
        json_path = OUT_DIR / f"detections_{param_name}.json"
        if not json_path.exists():
            mean_time, _ = generate_predictions_for_params(
                SRC_DIR, GT_COCO, json_path,
                text_threshold=tt,
                link_threshold=lt,
                low_text=ltx,
                long_size=ls
            )
        else:
            print(f"📁 Found cached detections: {json_path.name}")
            mean_time = None

        with open(json_path) as f:
            all_detections[param_name] = json.load(f)

        metrics = run_coco_eval(GT_COCO, CATEGORY_ID, str(json_path))
        results.append([param_name, mean_time] + metrics)

    # Save metrics summary
    columns = ["config", "mean_time"] + [
        "AP@[IoU=0.10:0.95]",
        "AP@0.10", "AP@0.30", "AP@0.50",
        "AR@[IoU=0.10:0.95]",
        "AR@0.10", "AR@0.30", "AR@0.50"
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
