"""
=============================================================
  AV Perception Lab — Flask Web Server
  Endpoints:
    GET  /              → serves index.html
    POST /predict       → upload image → GTSRB inference
    GET  /model-info    → model metadata + class list
    GET  /dataset-info  → dataset statistics
    GET  /sample-images → random test samples (if available)
=============================================================
"""

import os
import io
import json
import base64
import random
from pathlib import Path
from functools import lru_cache

from flask import (
    Flask, request, jsonify,
    send_from_directory, abort
)
from PIL import Image
import numpy as np

# ── Try loading TF model ──────────────────────────────────
try:
    import tensorflow as tf
    from gtsrb_model import (
        predict_image, build_gtsrb_model,
        GTSRB_CLASSES, NUM_CLASSES,
        IMG_WIDTH, IMG_HEIGHT,
    )
    TF_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] TensorFlow not available: {e}")
    TF_AVAILABLE = False


# ── App setup ────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH_V2 = BASE_DIR / "checkpoints" / "gtsrb_model_v2.h5"
MODEL_PATH_V1 = BASE_DIR / "checkpoints" / "gtsrb_model.h5"
MODEL_PATH    = MODEL_PATH_V2 if MODEL_PATH_V2.exists() else MODEL_PATH_V1
DATA_DIR   = BASE_DIR / "data" / "GTSRB"

app = Flask(__name__, static_folder=str(BASE_DIR))


# ── Lazy-load the model once ─────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        if not TF_AVAILABLE:
            return None
        if MODEL_PATH.exists():
            print(f"[INFO] Loading model from {MODEL_PATH}")
            import tensorflow as tf
            _model = tf.keras.models.load_model(str(MODEL_PATH))
        else:
            print("[WARN] No trained model found. Using untrained model for demo.")
            _model = build_gtsrb_model()
    return _model


# ── Routes ───────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: multipart/form-data with field 'image'
    Returns: JSON with top-5 predictions
    """
    if not TF_AVAILABLE:
        return jsonify({
            "error": "TensorFlow not installed. Run: pip install tensorflow",
            "demo": True,
            "predictions": _demo_prediction()
        }), 200

    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".gif", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not open image: {e}"}), 400

    # Thumbnail for response
    thumb = image.copy()
    thumb.thumbnail((120, 120))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    thumb_b64 = base64.b64encode(buf.getvalue()).decode()

    model = get_model()
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "demo": True,
            "predictions": _demo_prediction()
        }), 200

    result = predict_image(model, image)

    return jsonify({
        "demo": not MODEL_PATH.exists(),
        "thumbnail": f"data:image/png;base64,{thumb_b64}",
        "class_id":   result["class_id"],
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "top5":       result["top5"],
    })


@app.route("/model-info")
def model_info():
    """Return model metadata and class labels."""
    if not TF_AVAILABLE:
        classes = [{"id": k, "name": v} for k, v in _mock_classes().items()]
        return jsonify({"model": "demo", "num_classes": 43, "classes": classes})

    model = get_model()
    trained = MODEL_PATH.exists()
    classes = [{"id": k, "name": v} for k, v in GTSRB_CLASSES.items()]

    info = {
        "trained":     trained,
        "model_path":  str(MODEL_PATH) if trained else None,
        "num_classes": NUM_CLASSES,
        "input_shape": [IMG_HEIGHT, IMG_WIDTH, 3],
        "classes":     classes,
    }

    if model and trained:
        try:
            info["total_params"] = int(model.count_params())
        except Exception:
            pass

    return jsonify(info)


@app.route("/dataset-info")
def dataset_info():
    """Count GTSRB training images per class if dataset is present."""
    if not DATA_DIR.exists():
        return jsonify({
            "available": False,
            "message": f"Dataset not found at {DATA_DIR}",
            "hint": "Download GTSRB from https://benchmark.ini.rub.de/gtsrb_news.html"
        })

    train_dir = DATA_DIR / "Train"
    class_counts = {}
    total = 0
    for class_id in range(NUM_CLASSES if TF_AVAILABLE else 43):
        class_dir = train_dir / f"{class_id:05d}"
        if class_dir.exists():
            n = len(list(class_dir.glob("*.ppm")))
            class_counts[class_id] = n
            total += n

    return jsonify({
        "available":    True,
        "total_images": total,
        "class_counts": class_counts,
    })


@app.route("/sample-images")
def sample_images():
    """Return up to N base64-encoded random test images (with labels)."""
    n = int(request.args.get("n", 6))
    test_csv = DATA_DIR / "Test.csv"

    if not test_csv.exists():
        return jsonify({"available": False, "samples": []})

    # Read all test rows
    import csv as _csv
    rows = []
    with open(test_csv, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            img_path = DATA_DIR / row["Path"]
            if img_path.exists():
                rows.append({"path": img_path, "class_id": int(row["ClassId"])})

    chosen = random.sample(rows, min(n, len(rows)))
    samples = []
    for item in chosen:
        try:
            img = Image.open(item["path"]).convert("RGB")
            img.thumbnail((120, 120))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            samples.append({
                "image":      f"data:image/png;base64,{b64}",
                "class_id":   item["class_id"],
                "class_name": (GTSRB_CLASSES if TF_AVAILABLE else _mock_classes()).get(
                    item["class_id"], f"Class {item['class_id']}"
                ),
            })
        except Exception:
            continue

    return jsonify({"available": True, "samples": samples})


# ── Helpers ──────────────────────────────────────────────

def _demo_prediction():
    """Return a plausible-looking demo result when TF isn't available."""
    class_id = random.randint(0, 42)
    classes  = _mock_classes()
    return {
        "class_id":   class_id,
        "class_name": classes.get(class_id, f"Class {class_id}"),
        "confidence": round(random.uniform(0.60, 0.99), 4),
        "top5": [
            {"class_id": (class_id + i) % 43,
             "class_name": classes.get((class_id + i) % 43, ""),
             "confidence": round(random.uniform(0.01, 0.20), 4)}
            for i in range(5)
        ],
    }


def _mock_classes():
    return {
        0: "Speed limit (20km/h)",  1: "Speed limit (30km/h)",
        2: "Speed limit (50km/h)",  3: "Speed limit (60km/h)",
        4: "Speed limit (70km/h)",  5: "Speed limit (80km/h)",
        6: "End of speed limit (80km/h)",
        7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
        9: "No passing",           10: "No passing veh over 3.5 tons",
        11: "Right-of-way at intersection",
        12: "Priority road",       13: "Yield",
        14: "Stop",                15: "No vehicles",
        16: "Veh > 3.5 tons prohibited",
        17: "No entry",            18: "General caution",
        19: "Dangerous curve left",20: "Dangerous curve right",
        21: "Double curve",        22: "Bumpy road",
        23: "Slippery road",       24: "Road narrows on the right",
        25: "Road work",           26: "Traffic signals",
        27: "Pedestrians",         28: "Children crossing",
        29: "Bicycles crossing",   30: "Beware of ice/snow",
        31: "Wild animals crossing",
        32: "End speed + passing limits",
        33: "Turn right ahead",    34: "Turn left ahead",
        35: "Ahead only",          36: "Go straight or right",
        37: "Go straight or left", 38: "Keep right",
        39: "Keep left",           40: "Roundabout mandatory",
        41: "End of no passing",   42: "End no passing veh > 3.5 tons",
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  AV Perception Lab — Flask Server")
    print(f"  TF available : {TF_AVAILABLE}")
    print(f"  Model path   : {MODEL_PATH}")
    print(f"  Dataset dir  : {DATA_DIR}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=False)
