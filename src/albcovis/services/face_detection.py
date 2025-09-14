from deepface import DeepFace
from PIL import Image
from typing import List, Dict
from statistics import mean
import numpy as np

def detect_faces(path: str) -> Dict:
    # Ensure the path is a string and not Path object
    path = str(path)

    img = Image.open(path)
    w, h = img.size
    img_area = w * h
    
    faces_raw = DeepFace.extract_faces(path, detector_backend="yolov11n", enforce_detection=False)
    
    keys_to_remove = ["left_eye", "right_eye"]

    faces: List[Dict] = []
    tolerance = 2  # allow for slight off-by-one bbox sizes
    for d in faces_raw:
        fa = {k: v for k, v in d["facial_area"].items() if k not in keys_to_remove} # Don't return the array of pixels
        x, y, w_box, h_box = float(fa["x"]), float(fa["y"]), float(fa["w"]), float(fa["h"])
        conf = d.get("confidence", None)

        # Robust "fake face" detection
        is_conf_zero = (conf is None) or (conf == 0.0)
        is_full_image = (
            abs(x) <= tolerance and abs(y) <= tolerance
            and abs(w_box - w) <= tolerance
            and abs(h_box - h) <= tolerance
        )

        if is_conf_zero and is_full_image:
            continue  # skip fake face

        area = w_box * h_box
        face_info = {
            "bbox": [x, y, w_box, h_box],
            "area": float(area),
            "relative_size": float(area / img_area),
            "confidence": conf,
        }
        faces.append(face_info)

    # If no valid faces were found → return an "empty summary"
    if not faces:
        return {
            "n_faces": 0,
            "mean_area": 0.0,
            "average_relative_size": 0.0,
            "largest_face": None,
            "highest_confidence_face": None,
            "faces": []
        }

    # Aggregated metrics
    out = {
        "n_faces": len(faces),
        "mean_area": mean([f["area"] for f in faces]) if faces else 0.0,
        "average_relative_size": sum([f["area"] for f in faces]) / img_area if faces else 0.0,
        "largest_face": max(faces, key=lambda f: f["area"], default=None),
        "highest_confidence_face": max(faces, key=lambda f: f["confidence"], default=None),
        "faces": faces
    }

    return out
