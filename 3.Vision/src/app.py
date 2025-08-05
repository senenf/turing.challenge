import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import requests
import uuid
import os
import subprocess
import concurrent.futures

# Constants
IMAGE_DIR = "downloads"
RESULTS_DIR = "results"
DETECTION_TIMEOUT = 30  # seconds
YOLO_DETECT_SCRIPT = "utils/yolov9/detect.py"
YOLO_WEIGHTS = "utils/yolov9/yolov9-e-converted.pt"

# Prepare folders
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI()

# COCO class mapping (subset of interest)
COCO_CLASSES = {
    0: "person",
    2: "car",
    # 3: "motorcycle",
    # 5: "bus",
    # 7: "truck"
}

# Response models
class BBox(BaseModel):
    x_center: float
    y_center: float
    width: float
    height: float

class DetectedObject(BaseModel):
    class_: str
    bbox: BBox

class DetectionResult(BaseModel):
    message: str
    detections: dict[str, list[DetectedObject]] | None = None


def run_detection(image_path: str, result_name: str) -> list[dict]:
    output_dir = os.path.join(RESULTS_DIR, result_name)
    labels_dir = os.path.join(output_dir, "labels")
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    label_file = os.path.join(labels_dir, f"{image_name}.txt")

    command = [
        "python", YOLO_DETECT_SCRIPT,
        "--weights", YOLO_WEIGHTS,
        "--source", image_path,
        "--save-txt",
        "--project", RESULTS_DIR,
        "--name", result_name,
        "--exist-ok",
        "--conf", "0.2"
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("YOLOv9 stdout:", result.stdout)
        print("YOLOv9 stderr:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Detection error output:", e.stderr)
        raise RuntimeError(f"Detection failed: {e.stderr}")

    if not os.path.exists(label_file):
        print("No label file found, returning empty results.")
        return []

    detections = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])

            label = COCO_CLASSES.get(class_id)
            if label:
                detections.append({
                    "class_": label,
                    "bbox": {
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height
                    }
                })

    return detections

# Endpoints
@app.get("/detection", response_model=DetectionResult)
def detect_from_url(image_url: str = Query(..., description="Public image URL to detect objects")):
    try:
        # Download image
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from URL")

        image_id = str(uuid.uuid4())
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        result_name = image_id  # used for YOLO's output folders

        with open(image_path, 'wb') as f:
            f.write(response.content)

        # Run detection with timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_detection, image_path, result_name)
            try:
                detections = future.result(timeout=DETECTION_TIMEOUT)
            except concurrent.futures.TimeoutError:
                return DetectionResult(
                    message="The image is being processed and has failed to respond within the time limit.",
                    detections=None
                )

        return DetectionResult(message="Detection completed", detections={"objects": detections})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
