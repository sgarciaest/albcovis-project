from deepface import DeepFace
from PIL import Image
from typing import List, Dict

def detect_faces(path: str) -> Dict:
    # Ensure the path is a string and not Path object
    path = str(path)

    img = Image.open(path)
    w, h = img.size
    img_area = w * h
    
    faces = DeepFace.extract_faces(path, detector_backend="yolov11n", enforce_detection=False)
    
    # Don't return the array of pixels, only bboxes and confidence score

    keys_to_remove = ["left_eye", "right_eye"]
    faces = [
        {
            "bbox": list({k: v for k, v in d["facial_area"].items() if k not in keys_to_remove}.values()),
            "segmentation": [[d["facial_area"]["x"], d["facial_area"]["y"], d["facial_area"]["x"] + d["facial_area"]["w"], d["facial_area"]["y"], d["facial_area"]["x"] + d["facial_area"]["w"], d["facial_area"]["y"] + d["facial_area"]["h"], d["facial_area"]["x"], d["facial_area"]["y"] + d["facial_area"]["h"]]]
            "area": d["facial_area"]["w"] * d["facial_area"]["h"],
            "relative_size": (d["facial_area"]["w"] * d["facial_area"]["h"]) / img_area
            "confidence": d["confidence"]
        }
        for d in faces
    ]

    aggregated_faces_stats = {
        "n_faces": len(faces)
        "mean_area": mean([f["area"] for f in faces])
        "largest_face": XXXXX,
        "highest_confidence_face" XXXXXX,
        "average_relative_size": sum([f["area"] for f in faces]) / img_area
        "face_density": XXXXXXXXX
    }



    n_faces = len(faces)

    return faces
