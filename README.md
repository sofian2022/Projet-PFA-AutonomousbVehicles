<!-- Project Banner -->
<p align="center">
  <img src="Traffic signs Detection And Recognition/images/project.jpg" alt="Project Banner" width="70%"/>
</p>

# 🚗 Autonomous Vehicle Project

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/sofian2022/Projet_PFA-Autonomous_Vehicles)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Welcome to the Autonomous Vehicle Project! This repository contains a modular, multi-component system for smart vehicle perception and control, developed as part of a PFA (Projet de Fin d'Année). It integrates state-of-the-art computer vision and deep learning for real-time object detection, license plate recognition, and traffic sign classification, with hardware control for a physical prototype.

---

## ✨ Features

| 🚦  | **Traffic Sign Recognition**   | Classifies traffic signs using a CNN trained on the GTSRB dataset, with a user-friendly GUI.             |
| --- | ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| 🔍  | **Real-time Object Detection** | Detects objects (cars, pedestrians, cyclists) using a pre-trained MobileNet SSD model.                   |
| 🏷️  | **License Plate Recognition**  | Detects license plates with YOLOv8, recognizes characters with EasyOCR, and checks status in a database. |
| 🛠️  | **Hardware Integration**       | Scripts for controlling camera, motors, servos, ultrasonic sensors, and LEDs.                            |

---

## 🗂️ Project Structure

```
.
├── LIcense Plate Number Detection/
│   ├── check_vehicle_status.py
│   ├── license_plate_recognition.py
│   ├── add_sample_vehicles.py
│   ├── db_connector.py
│   ├── Test_WebCam.py
│   └── license_plate_detector.pt
│
├── Traffic signs Detection And Recognition/
│   ├── Train.ipynb
│   ├── Test_images/
│   ├── Test/
│   │   ├── rpiCam.py
│   │   ├── interface.py
│   │   └── cameraPC.py
│   ├── model/
│   │   ├── traffic_classifiernew.h5
│   │   └── model_trained_epoch30.p
│   └── images/
│       ├── project.jpg
│       ├── predictions.png
│       └── GUI.jpg
│
├── Object Detection/
│   ├── detect.py
│   ├── MobileNetSSD_deploy.prototxt.txt
│   └── MobileNetSSD_deploy.caffemodel
│
├── Materiels_Test/
│   ├── servoMotor.py
│   ├── leds.py
│   ├── ultrasonic.py
│   ├── motor.py
│   └── Camera.py
│
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

- **Python 3**
- **PyTorch** (YOLOv8 for license plate detection)
- **TensorFlow/Keras** (traffic sign recognition)
- **Caffe** (object detection model)
- **OpenCV** (image/video processing)
- **EasyOCR** (license plate OCR)
- **Pillow** (GUI image handling)
- **MongoDB** (vehicle/license plate database)
- **Tkinter** (traffic sign classifier GUI)
- **NumPy**, **Matplotlib**, **ultralytics**, **requests**

---

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sofian2022/Projet_PFA-Autonomous_Vehicles.git
   cd Projet_PFA-Autonomous_Vehicles
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Database Setup (for License Plate Recognition):**
   - Ensure MongoDB is running.
   - Update connection details in `LIcense Plate Number Detection/db_connector.py` if needed.
   - Populate the database with `add_sample_vehicles.py` if desired.

---

## 🖥️ Usage

- **Object Detection:**

  ```bash
  cd Object Detection
  python detect.py
  ```

  Starts real-time object detection using your webcam.

- **License Plate Recognition:**
  Main script: `license_plate_recognition.py` (see script for usage details; can process images or video streams).

- **Traffic Sign Classifier GUI:**

  ```bash
  cd "Traffic signs Detection And Recognition/Test"
  python interface.py
  ```

  Opens a GUI for uploading and classifying traffic sign images.

- **Hardware Tests:**
  Scripts in `Materiels_Test/` are for Raspberry Pi or similar hardware with the appropriate components.

---

## 📸 Screenshots

<p align="center">
  <img src="Traffic signs Detection And Recognition/images/predictions.png" alt="Traffic Sign Predictions" width="45%"/>
  <img src="Traffic signs Detection And Recognition/images/GUI.jpg" alt="Traffic Sign GUI" width="45%"/>
</p>

---

## 📄 License

This project is open-source. Please credit the original authors if you use or modify this code.
