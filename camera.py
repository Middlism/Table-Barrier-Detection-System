import requests
import numpy as np
import imutils
import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import time

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Calculate attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(x_cat))
        
        # Apply attention
        return x * attention_map

# Load YOLO model for object detection
model = YOLO("yolov8n.pt")

def filter_cup_detections(results):
    """Filter detection results to find only cups and glasses"""
    cups = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if "cup" in label.lower() or "glass" in label.lower():  # Fixed the missing quotes
                cups.append(box)
    return cups

# Initialize MiDaS model for monocular depth estimation
def initialize_midas():
    # Load MiDaS model (small version for faster inference)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    
    # Initialize transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    # Create attention model
    attention_model = SpatialAttention()
    attention_model.to(device)
    
    return midas, transform, attention_model, device

# Function to estimate depth using MiDaS with attention mechanism
def estimate_depth_midas(img, midas_model, transform, attention_model, device, roi=None):
    # Transform input for model
    img_tensor = transform(img).to(device)
    
    # Apply attention if we have a region of interest
    if roi is not None:
        # Convert img_tensor to format expected by attention model (B,C,H,W)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
        # Apply spatial attention
        img_tensor = attention_model(img_tensor)
        
        # Remove batch dimension if it was added
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze(0)
    
    # Make prediction
    with torch.no_grad():
        prediction = midas_model(img_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy and normalize for visualization
    depth_map = prediction.cpu().numpy()
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    depth_map_colorized = (depth_map_normalized * 255).astype(np.uint8)
    depth_map_colorized = cv2.applyColorMap(depth_map_colorized, cv2.COLORMAP_INFERNO)
    
    return depth_map_colorized, depth_map

# Function to create an ROI mask from detections
def create_roi_mask(img_shape, cup_detections, padding=20):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    for box in cup_detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Add padding to ROI
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_shape[1], x2 + padding)
        y2 = min(img_shape[0], y2 + padding)
        
        # Set the cup region to 1 in mask
        mask[y1:y2, x1:x2] = 255
    
    return mask

# Function to calculate accurate distance using depth and object size
def calculate_real_distance(depth_value, focal_length=500, actual_size=0.1, pixel_size=None):
    """
    Calculate real-world distance using depth, focal length, and object size
    depth_value: relative depth from MiDaS
    focal_length: camera focal length in pixels
    actual_size: actual size of the object in meters
    pixel_size: size of the object in pixels
    """
    if depth_value <= 0:
        return float('inf')
        
    # Convert relative depth to actual distance
    # This is a simplified approach - would need calibration for exact values
    estimated_distance = 1.0 / (depth_value + 1e-6) * 10
    
    # If we have pixel size and actual size, we can refine the estimate
    if pixel_size is not None:
        # Distance = (Actual size * Focal length) / Pixel size
        size_based_distance = (actual_size * focal_length) / pixel_size
        
        # Combine both estimates (weighted average)
        final_distance = 0.7 * estimated_distance + 0.3 * size_based_distance
        return final_distance
    
    return estimated_distance

# Main processing function with caching and performance optimizations
def process_frame(img, models, cache=None, frame_count=0):
    midas_model, transform, attention_model, device = models
    
    # Initialize cache if not provided
    if cache is None:
        cache = {
            'last_full_detection': 0,
            'last_detections': None,
            'detection_interval': 5  # Run full detection every 5 frames
        }
    
    # Run object detection with YOLO (only on certain frames)
    if frame_count % cache['detection_interval'] == 0:
        results = model(img, verbose=False)
        cache['last_detections'] = results
        cache['last_full_detection'] = frame_count
    else:
        # Use cached detections
        results = cache['last_detections']
    
    # Filter for cups and glasses
    cup_detections = filter_cup_detections(results)
    
    # Create ROI mask based on cup detections
    if len(cup_detections) > 0:
        roi_mask = create_roi_mask(img.shape, cup_detections)
        roi = cv2.bitwise_and(img, img, mask=roi_mask)
    else:
        roi = None
    
    # Process depth estimation with attention focused on ROIs
    start_time = time.time()
    depth_map_colorized, depth_map_raw = estimate_depth_midas(img, midas_model, transform, attention_model, device, roi)
    depth_processing_time = time.time() - start_time
    
    # Resize depth map for display
    depth_display = cv2.resize(depth_map_colorized, (img.shape[1] // 3, img.shape[0] // 3))
    
    # Create result image
    result_img = img.copy()
    result_img[0:depth_display.shape[0], 0:depth_display.shape[1]] = depth_display
    
    # Draw object detection results with improved distance estimation
    for box in cup_detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf = box.conf[0]
        
        # Calculate center and size of bounding box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        pixel_size = max(x2 - x1, y2 - y1)
        
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get depth value at object center
        distance_text = "Unknown"
        
        if depth_map_raw is not None:
            # Check if the point is within bounds
            if 0 <= center_y < depth_map_raw.shape[0] and 0 <= center_x < depth_map_raw.shape[1]:
                # Get depth value at the center of the object
                relative_depth = depth_map_raw[center_y, center_x]
                
                # Get more accurate distance estimation
                estimated_distance = calculate_real_distance(
                    relative_depth, 
                    focal_length=500,  # Adjust based on your camera
                    actual_size=0.1,   # Assume cup is 10cm
                    pixel_size=pixel_size
                )
                
                distance_text = f"{estimated_distance:.2f}m"
        
        text = f"{label} {conf:.2f} Dist: {distance_text}"
        cv2.putText(result_img, text, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add performance metrics
    cv2.putText(result_img, f"Depth: {depth_processing_time:.2f}s", 
                (result_img.shape[1]-200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return result_img, cache

# Main execution
def main():
    # Configure camera URL
    url = "http://192.168.1.106:8080/shot.jpg"
    
    # Initialize models
    midas_model, transform, attention_model, device = initialize_midas()
    models = (midas_model, transform, attention_model, device)
    
    # Initialize cache
    cache = None
    frame_count = 0
    
    # Process stream
    while True:
        # Get image from IP camera
        try:
            img_resp = requests.get(url, timeout=3)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            img = imutils.resize(img, width=1000)
            
            # Process frame
            result_img, cache = process_frame(img, models, cache, frame_count)
            frame_count += 1
            
            # Display the result
            cv2.imshow("Camera with Attention-Based Depth Estimation", result_img)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            time.sleep(1)  # Wait before retrying
        
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()