from craft_text_detector import load_craftnet_model, load_refinenet_model, get_prediction
from PIL import Image
from typing import List, Dict
from statistics import mean
import numpy as np

craft_net = load_craftnet_model(cuda=False)
refine_net = load_refinenet_model(cuda=False)

def detect_text(path: str) -> Dict:
    # Ensure the path is a string and not Path object
    path = str(path)

    img = Image.open(path).convert("RGB")
    img_nparray = np.array(img)
    w, h = img.size
    img_area = w * h

    prediction_result = get_prediction(
        image=img_nparray,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.5,
        link_threshold=0.1,
        low_text=0.4,
        long_size=500,
        cuda=False
    )


    boxes = prediction_result.get("boxes", [])
    texts: List[Dict] = []

    for box in boxes:
        # Compute bounding box from polygon
        x_coords = box[:, 0]
        y_coords = box[:, 1]
        x_min, y_min = np.min(x_coords), np.min(y_coords)
        x_max, y_max = np.max(x_coords), np.max(y_coords)
        width = x_max - x_min
        height = y_max - y_min
        area = width * height

        text_info = {
            "bbox": [float(x_min), float(y_min), float(width), float(height)],
            "area": float(area),
            "relative_size": float(area / img_area),
        }
        texts.append(text_info)

    # Aggregate stats
    out = {
        "n_texts": len(texts),
        "mean_area": mean([t["area"] for t in texts]) if texts else 0.0,
        "largest_text": max(texts, key=lambda t: t["area"], default=None),
        "average_relative_size": sum([t["area"] for t in texts]) / img_area if texts else 0.0,
        "texts": texts
    }

    return out
