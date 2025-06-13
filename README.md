# Raspberry Pi Face Attendance System

This project is a **Python + OpenCV + Flask**–based face attendance system optimized for **Raspberry Pi 4** using the **Picamera2** module. It detects and recognizes faces in real-time, logs attendance, and exports daily/weekly/monthly reports to CSV.

---

## Features

- Real-time face detection and recognition using OpenCV
- Uses Pi Camera (via `picamera2`) for lightweight performance
- Web-based interface (Flask) with:
  - Add, Erase, Fix person
  - Take attendance
  - View/download attendance reports (daily, weekly, monthly)
- CSV-based storage for labels and detections
- Works headless or with HDMI monitor
- Smart auto-timeouts and clean shutdown
- Live preview with rectangle and person’s name overlay

---

## Project Structure# face-attendance-system
face-attendance-system/
├── app.py # Flask web app

├── test_faces_full_workflow.py # Core face recognition logic (Raspberry Pi optimized)

├── utils.py # Helper functions (report generation, label handling)

├── faces/ # Collected face images

├── labels.csv # Label-to-name mappings

├── detections.csv # Attendance log

├── trainer.yml # Trained face recognizer model

├── templates/

│ └── index.html # Main UI

├── static/

│ └── styles.css # Simple CSS

└── README.md # (This file)


## Getting Started

### Prerequisites

- Raspberry Pi OS (Bullseye recommended)
- Raspberry Pi Camera Module (or USB webcam)
- Python 3.9+
- Dependencies:

```bash
sudo apt update
sudo apt install python3-opencv python3-pip libatlas-base-dev
pip3 install flask pandas opencv-contrib-python-headless picamera2

Enable camera:

bash
sudo raspi-config  # Interface Options → Camera → Enable

Usage
## Add a Person
bash
python3 test_faces_full_workflow.py --mode add --name "john"
Takes 20 face samples and trains the model.

## Train (only if you add manually to faces/)
bash
python3 test_faces_full_workflow.py --mode train
## Recognize & Take Attendance
bash
python3 test_faces_full_workflow.py --mode recognize
Will auto-exit after 3 mins or 60s idle

Draws rectangle and name above face

Logs data to detections.csv
## Start Web Interface
bash
python3 app.py
Visit http://<your-pi-ip>:5000 in your browser.

## Reports
Exportable from UI or CLI:

Daily
Weekly
Monthly
Saved as .csv in the same folder

✅ To-Do / Improvements
 Upload attendance to Google Sheets
 Add admin login and roles
 Touchscreen UI mode
 Add LED/beep feedback on recognition

👤Author
Built and customized by [phatnomenal].
📜 License
feel free to use my project
