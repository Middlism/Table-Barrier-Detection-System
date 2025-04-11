# Cup Liquid Detector

A computer vision system that detects cups and identifies liquids inside them using depth estimation, object detection, and color segmentation.

## Features

- Detects cups, glasses, and bottles using YOLOv8
- Estimates liquid type (water, coffee, tea) via color analysis
- Calculates fill levels for detected containers
- Measures approximate distance to objects
- Visualizes depth information with color mapping
- Optional segmentation using Segment Anything Model (SAM)

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics (YOLOv8)
- NumPy
- imutils
- Requests (for IP camera support)
- PIL

Optional:

- Segment Anything Model (for improved segmentation)

## Installation

```bash
pip install torch opencv-python ultralytics numpy imutils requests pillow
# Optional for SAM functionality
pip install segment-anything
```

## Usage

Run the main script to start detection:

```bash
python liquid_detector.py
```

The program will:

1. Try to connect to an IP camera (if available)
2. Fall back to webcam if no IP camera is detected
3. Display a window showing detections with bounding boxes, liquid type, fill level, and distance

## How It Works

1. **Object Detection**: YOLOv8 identifies cups and containers in the frame
2. **Depth Estimation**: MiDaS model creates a depth map to estimate distance
3. **Cup Segmentation**: Either SAM or conventional methods create a mask of the cup
4. **Liquid Detection**: HSV-based color segmentation identifies liquid type and fill level
5. **Visualization**: Results are displayed with colored overlays and text annotations

## Performance Optimizations

- Caching detection results to reduce processing overhead
- Dynamic model selection based on available resources
- Spatial attention mechanism to focus processing on relevant regions
- Reduced resolution for faster detection cycles

## License

[MIT License](LICENSE)
