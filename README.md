
# Object Detection and Tracking

## Overview

This Python script is designed to detect and track an object in a video feed. It uses computer vision techniques, including color filtering, noise addition, de-noising, and contour detection to locate and track the ball.

## Prerequisites

Before running the code, make sure you have the following prerequisites installed:

- **Python 3.x**
- **OpenCV (cv2)**
- **NumPy**
- **imutils**
- **VideoStream** (if not using an external video file)


##Usage
To run the code, use the following command-line arguments:

-v or --video: (optional) Path to a video file. If not provided, the script will use the webcam as the video source.
-b or --buffer: (optional) Maximum buffer size for tracking the ball. Default is 64.
example:
python red_ball_detection.py -v path_to_video.mp4 -b 100

## Image Processing Techniques

The code includes functions and options to apply various image processing techniques to the video frames. You can enable or disable these techniques by commenting/uncommenting the relevant lines in the code.

### 1. Sharpening
Uncomment the relevant line in the code to apply sharpening to the video frames.

### 2. Gaussian Blur
You can apply Gaussian blur to the frames with a specified kernel size. Modify the kernel size in the code to control the degree of blurring.

### 3. Illumination Variation
The code supports two illumination variation effects:

- **Brightening:** Uncomment the relevant line to increase the brightness of the frames.
- **Darkening:** Uncomment the relevant line to decrease the brightness of the frames.

### 4. Noise Addition
You can add various types of noise to the video frames:

- **Gaussian Noise:** Uncomment the relevant line to add Gaussian noise to the frames.
- **Salt and Pepper Noise:** Uncomment the relevant line to add salt and pepper noise to the frames.
- **Poisson Noise:** Uncomment the relevant line to add Poisson noise to the frames.

These image processing techniques can be adjusted to fine-tune the visual output and enhance the detection and tracking of the red ball.

## Tracking and Visualization

The script tracks the red ball in the video and draws a contrail of its movement. The tracked ball is displayed as a red circle, and the centroid is marked as a small red dot.

To exit the script, press 'q' in the window displaying the video.

## Notes

- Ensure that the color range (`redLower` and `redUpper`) for the red ball matches your specific setup.
- The script may require adjustments to work with different lighting conditions or ball colors.
"""
