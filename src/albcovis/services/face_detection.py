from deepface import DeepFace

def dectect_faces(path: str):
    # Ensure the path is a string and not Path object
    path = str(path)
    faces = DeepFace.extract_faces(path, detector_backend="yolov11n", enforce_detection=False)
    # Don't return the array of pixels, only bboxes and confidence score
    faces_no_image = [
        {"facial_area": d["facial_area"], "confidence": d["confidence"]}
        for d in faces
    ]
    return faces_no_image
