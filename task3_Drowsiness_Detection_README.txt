Drowsiness Detection System with Age Prediction

Problem Statement

Driver and passenger drowsiness is a major cause of road accidents and safety hazards. Manual monitoring is inefficient and not scalable. There is a need for an intelligent system that can:

Detect multiple people in real-time
Identify whether a person is awake or sleeping
Provide alerts when drowsiness is detected
Estimate additional attributes like age for better monitoring

This project proposes a deep learning-based drowsiness detection system using computer vision techniques.

Dataset

No custom dataset required
Uses:
Pretrained MobileNetV2 model for age prediction
Custom CNN model for eye state detection (Open/Closed)
Face and eye detection handled using OpenCV Haar Cascades

Methodology

The system follows a real-time detection pipeline:

Input Source
Image
Video
Live webcam
Face Detection
Detects faces using Haar Cascade classifier
Eye Detection
Detects eyes within each face region
Eye State Analysis
Determines whether eyes are open or closed using:
CNN model (EyeStateModel)
Eye Aspect Ratio (EAR) fallback method
Drowsiness Detection
If eyes remain closed for consecutive frames → classified as sleeping
Age Prediction
Uses MobileNetV2-based regression model
Predicts approximate age (0–100 years)
Visualization & Alerts
Sleeping persons highlighted in 🔴 Red
Awake persons highlighted in 🟢 Green
Popup alert for detected sleeping individuals

Data Preprocessing

Face images resized to 128×128 for age prediction
Eye regions resized to 48×48 grayscale for eye state detection
Image normalization applied using PyTorch transforms
Conversion:
BGR → RGB (for age model)
BGR → Grayscale (for eye model)

Feature Engineering

Eye Aspect Ratio (EAR)
Used as a fallback to detect eye closure
Temporal Feature
Consecutive frame tracking to confirm drowsiness
Face Region Extraction
Focuses on relevant regions (face & eyes)
Age Feature
Adds contextual information to detected individuals
Frame Skipping
Processes every 4th frame for performance optimization

Model Selection

1. Age Prediction Model
MobileNetV2 (Pretrained)
Modified for regression (single output neuron)
Chosen for:
Lightweight architecture
Efficient real-time inference
2. Eye State Detection Model
Custom CNN
Binary classification:
Open (0)
Closed (1)
Chosen for:
Simplicity
Fast processing
3. Haar Cascade Classifiers
Used for:
Face detection
Eye detection
Chosen for:
Speed and low computational cost

Results

Successfully detects multiple people in real-time
Accurately identifies:
Sleeping individuals
Awake individuals
Provides:
Age estimation
Real-time statistics
Alert system for drowsiness
Efficient performance due to:
Frame skipping
Lightweight models
Features
Image input support
Video input support
Live webcam detection
Drowsiness alert system
Multi-person detection
Age prediction
Real-time statistics dashboard
User-friendly GUI (Tkinter)

Installation
pip install torch torchvision opencv-python Pillow

Usage
python task3_drowsiness_detection.py

Steps:

Open image/video or start webcam
System detects faces and eye states
Identifies sleeping individuals
Displays results with alerts

Conclusion

This project demonstrates how computer vision and deep learning can be used to enhance safety systems. By combining eye state detection, temporal analysis, and age prediction, the system provides an effective solution for detecting drowsiness in real-time environments.