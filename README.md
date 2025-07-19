# 🚗 License Plate Detection with YOLOv8x + PaddleOCR

This project is a real-time **license plate detection system** using **YOLOv8x** (Ultralytics) and **PaddleOCR**, with a web-based UI for processing images, videos, and live webcam input.

---

## 📸 Demo Result

<img src="https://github.com/LePhuocThai2502/DetectLicensePlateYOLOv8_OCR/assets/your_screenshot_id/image1.png" alt="Image recognition results" width="500"/>

> ✅ Detected plate: **51F-64665**

---

## 🌐 Web Interface

Upload image, detect license plate, and get results visually:

<img src="https://github.com/LePhuocThai2502/DetectLicensePlateYOLOv8_OCR/assets/your_screenshot_id/interface.png" alt="Web interface" width="700"/>

---

## 📁 Project Structure

``` bash
DetectLicensePlateYOLOv8_OCR/
├── static/uploads/ # Uploaded images
├── webcam_recognition.py # Main recognition script
├── plates.db # SQLite database
├── requirements.txt # Python dependencies
├── .gitignore
```

---

## 📥 Download YOLOv8x Model

Due to GitHub's 100MB file size limit, the model file `yolov8x_finetuned.pt` is hosted externally.

➡️ **Download from Hugging Face:**

🔗 [Click to Download yolov8x_finetuned.pt](https://huggingface.co/LePhuocThai003/YoloV8x_finetuned/resolve/main/yolov8x_finetuned.pt)

> 📂 After downloading, place the file in the root directory of the project.

---

## ▶️ How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```
2. **Run the app:**:

```bash
python webcam_recognition.py
```

---
🔧 Tech Stack
-  YOLOv8x (from Ultralytics).
-  PaddleOCR.
-  OpenCV.
-  SQLite3.
-  Flask (for web UI).

📃 License
MIT License © LePhuocThai2502


---










