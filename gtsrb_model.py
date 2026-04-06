"""
=============================================================
  AV Perception Lab — GTSRB CNN (TensorFlow/Keras)
  Architecture matching the uploaded notebook:
    gtsrb-cnn-98-test-accuracy.ipynb
  
  Model: Custom CNN (no pretrained backbone)
  Input: 30×30×3 RGB 
  Output: 43-class softmax (all GTSRB signs)
  Target accuracy: ~98%
=============================================================
"""

import os
import csv
import random
import numpy as np
from pathlib import Path
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
#  GTSRB class labels — all 43 classes
# ─────────────────────────────────────────────
GTSRB_CLASSES = {
    0:  "Speed limit (20km/h)",
    1:  "Speed limit (30km/h)",
    2:  "Speed limit (50km/h)",
    3:  "Speed limit (60km/h)",
    4:  "Speed limit (70km/h)",
    5:  "Speed limit (80km/h)",
    6:  "End of speed limit (80km/h)",
    7:  "Speed limit (100km/h)",
    8:  "Speed limit (120km/h)",
    9:  "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5 tons",
}

NUM_CLASSES = 43
IMG_HEIGHT  = 30
IMG_WIDTH   = 30
CHANNELS    = 3

# ─────────────────────────────────────────────
#  Model Architecture (matches notebook)
# ─────────────────────────────────────────────
def build_gtsrb_model(num_classes=NUM_CLASSES):
    """
    CNN architecture from the Kaggle notebook achieving ~98% accuracy.
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (5, 5), activation='relu',
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.Conv2D(32, (5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ], name="GTSRB_CNN")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────
#  Data loading helpers
# ─────────────────────────────────────────────
def load_gtsrb_data(data_dir: str):
    """
    Load GTSRB dataset from directory.
    Expected layout:
        data_dir/
          Train/
            00000/ ← class 0
              *.ppm
            ...
          Test.csv
          Test/
            *.ppm
    
    Returns:
        X_train, y_train, X_test, y_test (numpy arrays)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "Train"
    test_csv  = data_dir / "Test.csv"

    print(f"[GTSRB] Loading training data from {train_dir}...")
    X_train, y_train = [], []
    for class_id in range(NUM_CLASSES):
        # Support both plain (0,1,2...) and zero-padded (00000,...) folder names
        class_path = train_dir / str(class_id)
        if not class_path.exists():
            class_path = train_dir / f"{class_id:05d}"
        if not class_path.exists():
            continue
        imgs = sorted(list(class_path.glob("*.png")) + list(class_path.glob("*.ppm")))
        for img_path in imgs:
            img = Image.open(img_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
            X_train.append(np.array(img))
            y_train.append(class_id)

    X_train = np.array(X_train, dtype=np.float32) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    print(f"[GTSRB] Training samples: {len(X_train)}")

    print(f"[GTSRB] Loading test data from {test_csv}...")
    X_test, y_test = [], []
    if test_csv.exists():
        with open(test_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try relative path first, then just filename in Test/
                img_path = data_dir / row["Path"]
                if not img_path.exists():
                    img_path = data_dir / "Test" / Path(row["Path"]).name
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
                    X_test.append(np.array(img))
                    y_test.append(int(row["ClassId"]))
    
    X_test = np.array(X_test, dtype=np.float32) / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    print(f"[GTSRB] Test samples: {len(X_test)}")

    return X_train, y_train, X_test, y_test


def train_model(data_dir: str, save_path: str = "checkpoints/gtsrb_model.h5",
                epochs: int = 15, batch_size: int = 64):
    """Train the GTSRB CNN and save to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    X_train, y_train, X_test, y_test = load_gtsrb_data(data_dir)

    model = build_gtsrb_model()
    model.summary()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest"
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # Final eval
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Final test accuracy: {acc:.4f}  loss: {loss:.4f}")
    return model, history


# ─────────────────────────────────────────────
#  Inference on a single image
# ─────────────────────────────────────────────
from PIL import ImageEnhance, ImageFilter

def _preprocess_for_gtsrb(image: Image.Image) -> Image.Image:
    """
    Smart preprocessing for real-world traffic sign photos.
    GTSRB model was trained on 30x30 tightly-cropped sign images.
    This function handles larger photos with backgrounds.
    """
    img = image.convert("RGB")
    w, h = img.size

    # 1. Square center-crop (keeps the sign which is usually centered)
    side = min(w, h)
    left  = (w - side) // 2
    top   = (h - side) // 2
    img   = img.crop((left, top, left + side, top + side))

    # 2. Contrast + sharpness boost to match GTSRB quality
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = ImageEnhance.Color(img).enhance(1.2)

    # 3. Resize to model input
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    return img


def _img_to_arr(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.float32) / 255.0


def predict_image(model_or_path, image: Image.Image):
    """
    Robust inference with test-time augmentation (TTA).
    Auto-detects input size from the model (works with v1 30x30 and v2 48x48).
    """
    if isinstance(model_or_path, (str, Path)):
        model = keras.models.load_model(str(model_or_path))
    else:
        model = model_or_path

    # Auto-detect input size from model
    input_shape = model.input_shape  # e.g. (None, 48, 48, 3)
    size = input_shape[1]            # 30 or 48

    img = image.convert("RGB")
    w, h = img.size

    # ─── Build TTA crops ────────────────────────────────────
    crops = []

    # Helper: resize a PIL image to model input size
    def rsz(im): return im.resize((size, size), Image.LANCZOS)

    # 1. Direct Resize (Best for clean clipart)
    crops.append(rsz(img))

    # 2. Smart center-crop square (Best for photos)
    side = min(w, h)
    cx   = img.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
    crops.append(rsz(cx))

    # 3. Inner 85% crop (Removes white borders from clipart)
    margin_x, margin_y = int(w * 0.075), int(h * 0.075)
    inner = img.crop((margin_x, margin_y, w - margin_x, h - margin_y))
    crops.append(rsz(inner))

    # 4. Slightly Brightened Center
    bright = ImageEnhance.Brightness(cx).enhance(1.2)
    crops.append(rsz(bright))

    # ─── Batch predict + average ────────────────────────────
    batch = np.stack([_img_to_arr(c) for c in crops], axis=0)
    all_probs = model.predict(batch, verbose=0)   # (6, 43)
    probs = all_probs.mean(axis=0)                # averaged (43,)

    class_id   = int(np.argmax(probs))
    confidence = float(probs[class_id])
    class_name = GTSRB_CLASSES.get(class_id, f"Class {class_id}")

    top5 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]
    top5_results = [
        {"class_id": int(c),
         "class_name": GTSRB_CLASSES.get(int(c), f"Class {c}"),
         "confidence": float(p)}
        for c, p in top5
    ]

    return {
        "class_id":   class_id,
        "class_name": class_name,
        "confidence": confidence,
        "top5":       top5_results,
    }


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/GTSRB"
    train_model(data_dir)
