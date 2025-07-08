# 🚗 Car Accident Detection System

A real-time accident detection app using **YOLOv8** and **Streamlit**. This system analyzes traffic footage (images, videos, or live feed) to detect potential accidents using computer vision and deep learning.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![OpenCV](https://img.shields.io/badge/opencv-v4.8+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🚀 Features

- 🔍 Real-time accident detection with YOLOv8  
- 🎥 Supports images, video files, and webcam input  
- 📊 Detection dashboard with confidence scores  
- 🚨 Alert levels based on severity  
- 🧠 Fast + accurate analysis with OpenCV + PyTorch

---
## 📸 Sample Output

<img src="./assets/screenshots/Screenshot%202025-07-08%20185232.png" width="600"/>

---

## 🛠️ Tech Stack

- **Object Detection**: YOLOv8 (Ultralytics)  
- **Frontend**: Streamlit  
- **Vision Processing**: OpenCV  
- **Backend**: Python, PyTorch  
- **Data**: NumPy, Pandas

---

## 🔧 Quick Start

```bash
git clone https://github.com/KaranMishra22/AccidentDetection.git
cd car-accident-detection

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📸 Input Modes
Image Upload – Detect accidents in still images

Video Upload – Frame-by-frame accident detection

Live Camera – Real-time detection with webcam

## 🎯 How It Works
YOLOv8 detects vehicles in traffic footage

Proximity + behavior analysis identifies crash patterns

System scores severity and triggers alerts

## 📬 Contact

**Karan N.**  
📧 karann23cb@psnacet.edu.in  
🌐 [GitHub](https://github.com/KaranMishra22)
