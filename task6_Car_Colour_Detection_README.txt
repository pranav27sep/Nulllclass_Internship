Car Colour Detection & Traffic Monitoring System

Problem Statement

Traffic monitoring systems require accurate detection of vehicles and pedestrians for better management and safety. Identifying specific vehicle attributes, such as color, can help in:

Traffic analysis and surveillance
Identifying suspicious vehicles
Improving smart city infrastructure

This project aims to build an intelligent system that:

Detects vehicles and people at traffic signals
Identifies car colors
Highlights blue cars (flagged vehicles)
Displays real-time traffic statistics

Dataset

Uses COCO (Common Objects in Context) dataset via pretrained models
No custom dataset required

Detectable objects:

Vehicles:
Car, Bus, Truck, Motorcycle, Bicycle
People

Methodology

The system follows a deep learning-based object detection pipeline:

Input Source
Image
Video
Live webcam
Object Detection
Uses Faster R-CNN (ResNet50 + FPN) pretrained on COCO
Object Filtering
Filters detections into:
Vehicles
People
Car Color Detection
Extracts vehicle bounding box region
Converts to HSV color space
Applies predefined color ranges
Classification
Identifies dominant color of vehicle
Flags:
🔴 Blue cars → Red bounding box
🔵 Other cars → Blue bounding box
Visualization
Displays bounding boxes with labels
Shows traffic statistics on screen

Data Preprocessing

Frame conversion:
BGR → RGB (for model input)
BGR → HSV (for color detection)
Region of Interest (ROI):
Uses upper portion of vehicle to avoid noise (road/shadows)
Confidence filtering:
Only detections above threshold (≥ 0.45) considered

Feature Engineering

Bounding Box Extraction
Used for localization of vehicles
HSV Color Segmentation
Detects dominant vehicle color
Color Aggregation
Combines red ranges for better detection
Traffic Metrics
Total vehicles
Blue cars (flagged)
Other vehicles
People count
Frame Skipping
Improves real-time performance

Model Selection

Faster R-CNN (ResNet50 + FPN)
Pretrained on COCO dataset
Chosen for:
High detection accuracy
Ability to detect multiple objects
Robust performance in complex scenes
HSV-Based Color Detection
Used for:
Efficient color identification
Chosen for:
Simplicity
Fast computation
Good performance in varying lighting

Results

Successfully detects:
Vehicles
People
Accurately identifies:
Car colors
Blue cars (flagged separately)
Displays:
Real-time traffic statistics
Color-wise vehicle distribution
Performs efficiently in:
Images
Videos
Live webcam
Features
Image input support
Video input support
Live webcam detection
Vehicle detection
Car color identification
Blue car alert system
People detection
Real-time traffic statistics
Interactive GUI (Tkinter)

Installation
pip install torch torchvision opencv-python Pillow

Usage
python task6_car_color_detection.py

Steps:
Open image/video or use webcam
System detects vehicles and people
View color classification and statistics

Conclusion

This project demonstrates how deep learning and image processing can be combined for intelligent traffic monitoring. By integrating object detection with color analysis, the system provides valuable insights for smart surveillance and traffic management applications.