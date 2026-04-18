Nationality Detection System with Emotion, Age & Dress Analysis

Problem Statement

Understanding demographic and visual attributes from facial images has applications in analytics, human-computer interaction, and surveillance systems. However, extracting multiple attributes such as nationality, emotion, age, and dress color manually is inefficient.

This project aims to build an intelligent system that:

Predicts nationality from facial images
Detects emotion for all individuals
Provides additional insights based on nationality:
Indian → Age + Dress Color + Emotion
American → Age + Emotion
African → Dress Color + Emotion
Others → Nationality + Emotion

Dataset

No custom dataset explicitly provided
Uses pretrained deep learning models:
ResNet18 (Nationality classification)
MobileNetV2 (Emotion detection)
EfficientNet-B0 (Age prediction)
Face detection handled using OpenCV Haar Cascade

Methodology

The system follows a multi-stage computer vision pipeline:

Input
Image upload
Webcam capture
Face Detection
Detects faces using Haar Cascade classifier
Feature Extraction
Extracts facial regions for further analysis
Nationality Prediction
Uses ResNet18 model
Outputs probability distribution across 8 classes
Emotion Detection
Uses MobileNetV2 model
Classifies into 7 emotions
Age Prediction
Uses EfficientNet-based regression model
Dress Color Detection
Extracts body region below face
Uses HSV color space for dominant color detection
Conditional Output Logic
Output varies depending on predicted nationality

Data Preprocessing

Face images resized to 224×224 for nationality & age models
Emotion images resized to 48×48 grayscale
Normalization applied using standard ImageNet values
Color conversions:
BGR → RGB (for deep learning models)
BGR → HSV (for dress color detection)

Feature Engineering

Facial Region Extraction
Focuses only on relevant regions
Probability Scores
Softmax used for nationality & emotion confidence
HSV Color Segmentation
Used to detect dominant dress color
Conditional Feature Output
Dynamic feature selection based on nationality

Model Selection

1. Nationality Model
ResNet18 (Pretrained)
Modified final layer for multi-class classification
Chosen for:
Strong feature extraction capability
Good balance of speed and accuracy
2. Emotion Model
MobileNetV2
Modified for grayscale input
Chosen for:
Lightweight and efficient
Suitable for real-time inference
3. Age Model
EfficientNet-B0
Regression output scaled to age range
Chosen for:
High performance in image-based regression tasks
4. Haar Cascade
Used for face detection
Chosen for:
Fast and lightweight performance

Results

Successfully detects faces and predicts:
Nationality
Emotion
Age
Dress color
Provides dynamic outputs based on nationality rules
Displays results in a clean GUI interface
Handles multiple faces in a single image
Features
Image upload support
Webcam capture
Nationality prediction
Emotion detection
Age estimation
Dress color detection
Multi-person analysis
Interactive GUI (Tkinter)

Installation
pip install torch torchvision opencv-python Pillow

Usage
python task4_nationality_detection.py

Steps:
Upload an image or capture via webcam
Click Analyze
View results in the panel

Conclusion

This project demonstrates a multi-task deep learning system that integrates classification, regression, and image processing techniques. By combining multiple models, it provides a comprehensive analysis of facial attributes in a single pipeline.