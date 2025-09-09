from deepface import DeepFace
from PIL import Image
from typing import List, Dict
from statistics import mean

def detect_faces(path: str) -> Dict:
    # Ensure the path is a string and not Path object
    path = str(path)

    img = Image.open(path)
    w, h = img.size
    img_area = w * h
    
    faces_raw = DeepFace.extract_faces(path, detector_backend="yolov11n", enforce_detection=False)
    
    keys_to_remove = ["left_eye", "right_eye"]

    faces: List[Dict] = []
    for d in faces_raw:
        fa = {k: v for k, v in d["facial_area"].items() if k not in keys_to_remove} # Don't return the array of pixels
        x, y, w_box, h_box = fa["x"], fa["y"], fa["w"], fa["h"]
        area = w_box * h_box
        face_info = {
            "bbox": [x, y, w_box, h_box],
            "segmentation": [[x, y, x + w_box, y, x + w_box, y + h_box, x, y + h_box]],
            "area": area,
            "relative_size": area / img_area,
            "confidence": d.get("confidence", 1.0),
        }
        faces.append(face_info)

    # Aggregated metrics
    out = {
        "n_faces": len(faces),
        "mean_area": mean([f["area"] for f in faces]),
        "largest_face": max(faces, key=lambda f: f["area"], default=None),
        "highest_confidence_face": max(faces, key=lambda f: f["confidence"], default=None),
        "average_relative_size": sum([f["area"] for f in faces]) / img_area,
    }

    out["faces"] = faces

    return out
