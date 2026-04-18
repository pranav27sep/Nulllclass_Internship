Sign Language Detection System (ASL Recognition)

Problem Statement

Communication barriers between hearing-impaired individuals and others can make everyday interactions difficult. There is a need for an intelligent system that can:

Recognize hand gestures in real-time
Translate sign language into readable text
Support both alphabets and common words
Operate efficiently in real-time environments

This project aims to develop a deep learning-based sign language recognition system that detects hand gestures and predicts corresponding signs.

Dataset

No explicit dataset provided in code (assumes pretrained/fine-tuned model)
Supports 45 classes:
A–Z alphabets (26 classes)
Common words (19 classes):
Hello, Yes, No, Thanks, Sorry, Help, Stop, Good, Bad, Love, Eat, Water, Home, Work, Go, Come, Wait, More, Less

Methodology

The system follows a real-time computer vision pipeline:

Input
Image upload
Live webcam feed
Hand Detection
Uses HSV-based skin color segmentation
Extracts Region of Interest (ROI) containing the hand
Preprocessing
ROI resized to model input size
Normalization applied
Sign Prediction
Uses deep learning model to classify gestures
Outputs predicted sign with confidence score
Visualization
Displays bounding box around hand
Shows predicted sign and confidence
Time Constraint
System operates only between 6 PM – 10 PM (configurable)

Data Preprocessing

Input images resized to 224×224
Normalization using ImageNet mean and standard deviation
Color conversion:
BGR → RGB (for model input)
BGR → HSV (for hand detection)
Noise reduction:
Gaussian blur
Morphological operations (dilation)

Feature Engineering

Hand Region Extraction
Focuses on gesture-specific region
Skin Color Segmentation
Detects hand using HSV thresholds
Confidence Scores
Softmax probabilities for prediction confidence
Temporal Tracking
Maintains history of detected signs
Time-based Feature
System activation restricted to specific hours

Model Selection

EfficientNet-B0
Pretrained deep learning model
Modified classifier for 45-class classification

Why EfficientNet-B0?

High accuracy with fewer parameters
Efficient for real-time applications
Strong performance in image classification

Results

Successfully detects hand gestures in:
Images
Real-time webcam
Accurately predicts:
Alphabets (A–Z)
Common words
Displays:
Predicted sign
Confidence score
Detection history
Performs efficiently with real-time responsiveness
Features
Image upload support
Real-time webcam detection
Sign recognition (A–Z + words)
Time-restricted operation (6 PM–10 PM)
Confidence score display
Detection history tracking
Interactive GUI (Tkinter)

Installation
pip install torch torchvision opencv-python Pillow mediapipe

Usage
python task5_sign_language_detection.py

Steps:
Upload image or start webcam
Show hand gesture
View predicted sign and confidence
Check detection history

Conclusion

This project demonstrates how deep learning and computer vision can be used to bridge communication gaps through sign language recognition. By combining EfficientNet with image preprocessing techniques, the system delivers accurate and real-time gesture predictions.