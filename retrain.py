"""
Retrain GTSRB model from scratch with 48x48 input
(larger input = better sign feature discrimination)
Target: 98%+ accuracy, especially for similar signs like 50/100km/h
"""
import os, csv, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NUM_CLASSES = 43
IMG_SIZE    = 48          # bigger than 30 → model can distinguish 50 vs 100

DATASET = "/Users/saatwiksaxena/.cache/kagglehub/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/versions/1"
SAVE_PATH = "checkpoints/gtsrb_model_v2.h5"

GTSRB_CLASSES = {
    0:"Speed limit (20km/h)",1:"Speed limit (30km/h)",2:"Speed limit (50km/h)",
    3:"Speed limit (60km/h)",4:"Speed limit (70km/h)",5:"Speed limit (80km/h)",
    6:"End of speed limit (80km/h)",7:"Speed limit (100km/h)",8:"Speed limit (120km/h)",
    9:"No passing",10:"No passing veh over 3.5 tons",11:"Right-of-way at intersection",
    12:"Priority road",13:"Yield",14:"Stop",15:"No vehicles",16:"Veh > 3.5 tons prohibited",
    17:"No entry",18:"General caution",19:"Dangerous curve left",20:"Dangerous curve right",
    21:"Double curve",22:"Bumpy road",23:"Slippery road",24:"Road narrows on the right",
    25:"Road work",26:"Traffic signals",27:"Pedestrians",28:"Children crossing",
    29:"Bicycles crossing",30:"Beware of ice/snow",31:"Wild animals crossing",
    32:"End speed + passing limits",33:"Turn right ahead",34:"Turn left ahead",
    35:"Ahead only",36:"Go straight or right",37:"Go straight or left",38:"Keep right",
    39:"Keep left",40:"Roundabout mandatory",41:"End of no passing",
    42:"End no passing veh > 3.5 tons"
}

# ── Better CNN for 48×48 input ────────────────────────────
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Block 1
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.2),

        # Block 3
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ], name="GTSRB_CNN_v2")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Load data ─────────────────────────────────────────────
def load_data():
    data_dir = Path(DATASET)
    train_dir = data_dir / "Train"
    test_csv  = data_dir / "Test.csv"

    print("[GTSRB] Loading training images at 48×48...")
    X_train, y_train = [], []
    for class_id in range(NUM_CLASSES):
        class_path = train_dir / str(class_id)
        if not class_path.exists():
            class_path = train_dir / f"{class_id:05d}"
        if not class_path.exists():
            continue
        imgs = list(class_path.glob("*.png")) + list(class_path.glob("*.ppm"))
        for img_path in sorted(imgs):
            img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            X_train.append(np.array(img))
            y_train.append(class_id)

    X_train = np.array(X_train, dtype=np.float32) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    print(f"  Train: {len(X_train)} images")

    print("[GTSRB] Loading test images...")
    X_test, y_test = [], []
    with open(test_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = data_dir / row["Path"]
            if not img_path.exists():
                img_path = data_dir / "Test" / Path(row["Path"]).name
            if img_path.exists():
                img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                X_test.append(np.array(img))
                y_test.append(int(row["ClassId"]))

    X_test = np.array(X_test, dtype=np.float32) / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    print(f"  Test:  {len(X_test)} images")
    return X_train, y_train, X_test, y_test


# ── Train ─────────────────────────────────────────────────
def train():
    os.makedirs("checkpoints", exist_ok=True)
    X_train, y_train, X_test, y_test = load_data()

    model = build_model()
    model.summary()

    datagen = ImageDataGenerator(
        rotation_range=12,
        zoom_range=0.15,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.12,
        brightness_range=[0.7, 1.3],
        horizontal_flip=False,
        fill_mode="nearest"
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            SAVE_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1
        ),
    ]

    print("\n🚀 Training with 48×48 input (better sign discrimination)...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=25,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Final test accuracy: {acc*100:.2f}%  loss: {loss:.4f}")

    # Update app to use v2 model
    print(f"\nSaved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()
