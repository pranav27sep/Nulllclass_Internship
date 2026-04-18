Smart Attendance System with Emotion Detection

Problem Statement

Traditional attendance systems are time-consuming, prone to proxy attendance, and lack automation. There is a need for an intelligent system that can:

Automatically detect and recognize students
Mark attendance accurately in real-time
Prevent proxy attendance
Capture additional contextual information such as student emotions

This project proposes a face recognition-based attendance system integrated with emotion detection, operating within a defined time window.

Dataset

The system uses a custom dataset generated during runtime:

Student face images captured via webcam
Face embeddings stored in a serialized database (student_database.pkl)
Emotion detection uses pretrained deep learning models (MobileNetV2-based) trained on standard facial emotion datasets
No external static dataset is required, making the system dynamic and adaptable.

Methodology

The system follows a real-time pipeline:

Face Detection
Uses OpenCV Haar Cascade to detect faces from live video feed
Face Recognition
Extracts facial embeddings using a ResNet18-based PyTorch model
Matches embeddings with stored student database using distance threshold
Emotion Detection
Uses a MobileNetV2-based CNN model
Classifies emotions into 7 categories:
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
Attendance Marking
Marks student as Present if recognized
Stores timestamp, emotion, and confidence score
Time Constraint
Attendance is allowed only within a fixed window (e.g., 9:30–10:00 AM)
Output Storage
Saves attendance in:
Excel (.xlsx)
CSV (.csv)

Data Preprocessing
Face images resized:
112×112 for face recognition
48×48 (grayscale) for emotion detection
Normalization applied using PyTorch transforms
Conversion:
BGR → RGB (for embedding model)
BGR → Grayscale (for emotion model)

Feature Engineering
Face Embeddings (128-dimensional vector)
Generated using ResNet18 backbone
Normalized for better similarity comparison
Distance-Based Matching
Euclidean distance used to identify closest match
Emotion Feature
Adds contextual information to attendance record
Confidence Score
Derived from embedding distance (1 - distance)

Model Selection
1. Face Recognition Model
ResNet18 (Pretrained)
Modified to output 128-dimensional embeddings
Chosen for:
Good balance between accuracy and speed
Lightweight for real-time applications
2. Emotion Detection Model
MobileNetV2 (Pretrained)
Modified for 7-class classification
Chosen for:
Efficiency on real-time inference
Strong performance on image classification tasks

Results
Successfully detects and recognizes faces in real-time
Accurately marks attendance with:
Name
Student ID
Time
Emotion
Confidence score
Prevents duplicate entries
Automatically marks absentees
Generates well-formatted Excel reports with:
Color-coded Present/Absent status
Features
Live camera feed
Student registration via face capture
Emotion detection integration
Time-restricted attendance marking
GUI using Tkinter
Export to Excel & CSV
Automatic absent marking

Installation
pip install torch torchvision opencv-python Pillow openpyxl facenet-pytorch deepface

Usage
python task1_attendance_system.py

Steps:
Start camera
Register students
Capture & mark attendance
Save records

Conclusion

This system demonstrates how computer vision + deep learning can automate attendance efficiently while enhancing it with emotion analytics. It is scalable, real-time, and reduces manual effort significantly.