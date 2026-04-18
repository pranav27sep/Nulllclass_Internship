Animal Detection & Classification System

Problem Statement

Monitoring animals in images and videos is important for applications such as wildlife surveillance, safety systems, and environmental monitoring. However, manually identifying animals and distinguishing between dangerous (carnivores) and non-dangerous (herbivores) species is inefficient and error-prone.

This project aims to build a real-time animal detection system that:

Detects multiple animals in images, videos, or live webcam feed
Classifies them into categories
Highlights carnivores (dangerous) and herbivores (non-dangerous)
Provides alerts when carnivores are detected

Dataset

Uses the COCO (Common Objects in Context) dataset via pretrained models
No custom dataset training required
Detectable animals include:
Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe
Classification:
Carnivores: Cat, Dog, Bear
Herbivores/Others: Horse, Sheep, Cow, Elephant, Zebra, Giraffe, Bird

Methodology

The system follows a deep learning-based object detection pipeline:

Input Source
Image upload
Video file
Live webcam
Object Detection
Uses Faster R-CNN with ResNet50 + FPN backbone
Pretrained on COCO dataset
Filtering
Filters detections to include only animal classes
Applies confidence threshold (≥ 0.45)
Classification
Identifies detected animal labels
Categorizes into:
Carnivore (Red bounding box 🔴)
Herbivore/Other (Blue bounding box 🔵)
Visualization
Draws bounding boxes with labels and confidence scores
Displays detection statistics
Alert System
Shows popup alert if carnivores are detected

Data Preprocessing
Input frames converted:
BGR → RGB for model compatibility
Images converted to tensors using torchvision.transforms
Normalization handled internally by pretrained model

Feature Engineering
Bounding Box Features
Coordinates extracted for localization
Confidence Scores
Used to filter weak detections
Custom Animal Mapping
COCO class IDs mapped to animal labels
Diet-Based Classification
Manual mapping:
Carnivore vs Herbivore
Frame Skipping Optimization
Processes every 3rd frame in videos for performance

Model Selection
Faster R-CNN (ResNet50 + FPN)
Pretrained on COCO dataset
Chosen because:
High accuracy in object detection
Robust for multiple object detection
Suitable for real-time applications with optimization
Why not YOLO?
Faster R-CNN provides:
Better accuracy for academic/research projects
More stable detection for smaller datasets

Results
Successfully detects multiple animals in:
Images
Videos
Live webcam
Clearly distinguishes:
🔴 Carnivores (Red boxes)
🔵 Herbivores/Others (Blue boxes)
Displays:
Total animals detected
Number of carnivores
Number of herbivores
Generates real-time alerts for carnivores
Features
Image detection
Video detection
Live webcam support
Carnivore alert system
Real-time detection statistics
Detection list display
User-friendly GUI (Tkinter)

Installation
pip install torch torchvision opencv-python Pillow

Usage
python task2_animal_detection.py

Steps:
Open image/video or use webcam
System detects animals automatically
View results with bounding boxes and stats
Receive alerts for carnivores

Conclusion

This project demonstrates how deep learning-based object detection can be applied to real-world problems like animal monitoring and safety systems. By combining Faster R-CNN with custom classification logic, the system delivers accurate, real-time results with meaningful insights.