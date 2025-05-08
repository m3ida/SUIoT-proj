import os
import json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# CONFIGURATION
IMAGE_DIR = "./results"  # Path to your images
OUTPUT_JSON = "label_studio_preannotations.json"
MODEL_PATH = "digits_detection.pt"  # Change to your model or custom weights
LABEL_STUDIO_IMAGE_PREFIX = "/data/local-files/?d=home/almeida/repos/UMa/SUIoT/project/results/"  # Adjust if needed

# Load YOLO model
model = YOLO(MODEL_PATH)

# Collect all images
image_paths = list(Path(IMAGE_DIR).rglob("*.jpg")) + list(Path(IMAGE_DIR).rglob("*.png"))

results_json = []

for image_path in image_paths:
    image = Image.open(image_path)
    width, height = image.size

    result = model.predict(source=str(image_path), conf=0.25, save=False, verbose=False)[0]
    annotations = []

    for box in result.boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id]

        xyxy = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = xyxy

        # Normalize bbox to percentages for Label Studio
        x = (x1 / width) * 100
        y = (y1 / height) * 100
        w = ((x2 - x1) / width) * 100
        h = ((y2 - y1) / height) * 100

        annotations.append({
            "value": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "rectanglelabels": [label]
            },
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels"
        })

    results_json.append({
        "data": {
            "image": LABEL_STUDIO_IMAGE_PREFIX + str(image_path).split("/")[-1]
        },
        "predictions": [{
            "model_version": MODEL_PATH,
            "result": annotations
        }]
    })

# Write to JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(results_json, f, indent=2)

print(f"✅ Saved preannotations to {OUTPUT_JSON}")
