import cv2
import os
import numpy as np
import pandas as pd
import csv
import argparse
from datetime import datetime

# Get absolute path to the script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, 'faces')
LABELS_CSV = os.path.join(BASE_DIR, 'labels.csv')
DETECTIONS_CSV = os.path.join(BASE_DIR, 'detections.csv')
TRAINER_YML = os.path.join(BASE_DIR, 'trainer.yml')

# Helper: Load or update label-name mapping
def load_labels():
    labels = {}
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    labels[int(row[0])] = row[1]
    return labels

def save_label(label, name):
    labels = load_labels()
    if label not in labels:
        with open(LABELS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([label, name])

# Step 1: Collect face images
def collect_faces(name=None, num_samples=20):
    if name is None:
        name = input("Enter the person's name: ").strip()
    else:
        name = name.strip()

    os.makedirs(FACES_DIR, exist_ok=True)
    # Assign label: next available integer
    labels = load_labels()
    if name in labels.values():
        label = [k for k, v in labels.items() if v == name][0]
    else:
        label = max(labels.keys(), default=0) + 1
        save_label(label, name)
    
    if name in labels.values():
        label = [k for k, v in labels.items() if v == name][0]
    else:
        label = max(labels.keys(), default=0) + 1
        save_label(label, name)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera!")
        return
    print("Camera opened successfully.")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    print("Press 's' to save a face, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera!")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_img = None
        if len(faces) > 0:
            print(f"Detected {len(faces)} face(s)")
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
        else:
            print("No face detected in this frame.")
        cv2.imshow('Collect Faces', frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            if face_img is not None:
                cv2.imwrite(os.path.join(FACES_DIR, f"{label}_{count}.jpg"), face_img)
                print(f"Saved {os.path.join(FACES_DIR, f'{label}_{count}.jpg')} for {name}")
                count += 1
                if count >= num_samples:
                    print("Collected required number of samples.")
                    break
            else:
                print("No face to save! Please ensure your face is visible to the camera.")
        elif key == ord('q'):
            print("Quitting face collection.")
            break
    cap.release()
    cv2.destroyAllWindows()

# Step 2: Train recognizer
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    if not os.path.exists(FACES_DIR) or len(os.listdir(FACES_DIR)) == 0:
        print("No training data found!")
        return False
    for filename in os.listdir(FACES_DIR):
        if filename.endswith('.jpg'):
            img_path = os.path.join(FACES_DIR, filename)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                label = int(filename.split('_')[0])
                faces.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    if len(faces) == 0:
        print("No valid face images found for training!")
        return False
    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_YML)
    print(f"Training complete with {len(faces)} images.")
    return True

# Step 3: Real-time recognition and logging
def recognize():
    if not os.path.exists(TRAINER_YML):
        print("No trained model found! Please train the recognizer first.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_YML)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    label_map = load_labels()
    if not face_cascade.empty():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera!")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame!")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))
                try:
                    label, confidence = recognizer.predict(roi)
                    name = label_map.get(label, f"Unknown ({label})")
                    color = (255, 0, 0) if confidence < 100 else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{name} ({confidence:.0f}%)", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    # Log detection
                    with open(DETECTIONS_CSV, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now().isoformat(), name, confidence])
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Failed to load face cascade classifier!")

# Camera preview before collecting faces
def camera_preview():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera!")
        return False
    print("Camera preview started. Press 's' to start collecting faces, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera!")
            break
        cv2.imshow('Camera Preview', frame)
        key = cv2.waitKey(1)
        if key != -1:
            print(f"Key pressed: {key}")  # Debug: see what key code is received
        if key in [ord('s'), ord('S')]:
            print("Starting face collection...")
            cap.release()
            cv2.destroyAllWindows()
            return True
        elif key in [ord('q'), ord('Q')]:
            print("Quitting.")
            cap.release()
            cv2.destroyAllWindows()
            return False

# --- Main workflow ---
"""if __name__ == "__main__":
    if camera_preview():
        print("Step 1: Collecting faces")
        collect_faces()
        print("\nStep 2: Training recognizer")
        if train_recognizer():
            print("\nStep 3: Starting real-time recognition")
            recognize() """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['add', 'train', 'recognize'], required=True)
    parser.add_argument('--name', help='Name of person (used only in add mode)')
    args = parser.parse_args()

    if args.mode == 'add':
        if not args.name:
            print("Error: --name is required in add mode")
        else:
            if camera_preview():
                collect_faces(args.name)
                train_recognizer()
    elif args.mode == 'train':
        train_recognizer()
    elif args.mode == 'recognize':
        recognize()

