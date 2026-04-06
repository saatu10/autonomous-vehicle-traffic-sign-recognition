the site is live at 
https://elaine-hawthorny-jaquelyn.ngrok-free.dev/
# Autonomous Vehicle Traffic Sign Recognition 🚦🚗

An AI-powered image classification system designed to assist autonomous vehicles by recognizing and classifying traffic signs accurately in real-time. This project utilizes a custom Convolutional Neural Network (CNN) built with TensorFlow to classify images from the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset into 43 distinct categories.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)

## ✨ Features

- **Custom Neural Network:** A fast, multi-layer CNN built with Keras specifically tuned for quick inference (~10MB footprint).
- **Test-Time Augmentation (TTA):** Robust preprocessing pipelined built into the inference engine handles noisy real-world vehicle camera feeds, mitigating issues like poor contrast, glare, and scaling artifacts.
- **Automated Data Fetching:** Seamlessly downloads and processes the massive GTSRB dataset automatically via `kagglehub`.
- **Interactive Web UI:** A premium, dark-themed Flask frontend that allows users to drag-and-drop live images and view the top-5 predicted probabilities.

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/saatu10/autonomous-vehicle-traffic-sign-recognition.git
   cd autonomous-vehicle-traffic-sign-recognition
   ```

2. **Install dependencies**
   Make sure you have Python 3.8+ installed. Run:
   ```bash
   pip install tensorflow flask pillow kagglehub numpy
   ```

3. **Run the Web Interface**
   To launch the Flask backend and the interactive visualizer:
   ```bash
   python app.py
   ```
   *The server will start at `http://localhost:5050`.*

---

## 🧠 Training the Model

If you wish to train the model from scratch on your own machine:

1. **Start the Training Script:**
   ```bash
   python retrain.py
   ```
2. **Details:** The training script automatically downloads the GTSRB dataset, resizes files to 48x48 dynamically, and applies aggressive data augmentation (rotations, zoom, shear) to artificially expand the 50,000+ training pool. Checkpoints are automatically saved to the `checkpoints/` directory.

## 📊 Dataset: GTSRB
The **German Traffic Sign Recognition Benchmark** dataset features:
- **43 Classes** ranging from Speed Limits (20km/h - 120km/h) to Yield, Stop, and Pedestrian Crossing signs.
- **50,000+ Images** spanning varying lighting conditions, occlusions, and physical damage.
- The model in this repository achieves **96%+ Validation Accuracy** on the testing sets.
