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

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(x_cat))
        return x * attention_map

# Load YOLO for object detection
model = YOLO("yolov8n.pt")

def filter_cup_detections(results):
    cups = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if "cup" in label.lower() or "glass" in label.lower():
                cups.append(box)
    return cups

# Load MiDaS for depth estimation
def initialize_midas():
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    attention_model = SpatialAttention()
    attention_model.to(device)
    
    return midas, transform, attention_model, device

# Depth estimation with optional attention
def estimate_depth_midas(img, midas_model, transform, attention_model, device, roi=None):
    img_tensor = transform(img).to(device)
    
    if roi is not None:
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = attention_model(img_tensor)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze(0)
    
    with torch.no_grad():
        prediction = midas_model(img_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    depth_map_colorized = (depth_map_normalized * 255).astype(np.uint8)
    depth_map_colorized = cv2.applyColorMap(depth_map_colorized, cv2.COLORMAP_INFERNO)
    
    return depth_map_colorized, depth_map

def create_roi_mask(img_shape, cup_detections, padding=20):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for box in cup_detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_shape[1], x2 + padding)
        y2 = min(img_shape[0], y2 + padding)
        mask[y1:y2, x1:x2] = 255
    return mask

# Estimate real-world distance
def calculate_real_distance(depth_value, focal_length=500, actual_size=0.1, pixel_size=None):
    if depth_value <= 0:
        return float('inf')
        
    estimated_distance = 1.0 / (depth_value + 1e-6) * 10
    
    if pixel_size is not None:
        size_based_distance = (actual_size * focal_length) / pixel_size
        final_distance = 0.7 * estimated_distance + 0.3 * size_based_distance
        return final_distance
    
    return estimated_distance

# Frame processing pipeline
def process_frame(img, models, cache=None, frame_count=0):
    midas_model, transform, attention_model, device = models
    
    if cache is None:
        cache = {
            'last_full_detection': 0,
            'last_detections': None,
            'detection_interval': 5
        }
    
    if frame_count % cache['detection_interval'] == 0:
        results = model(img, verbose=False)
        cache['last_detections'] = results
        cache['last_full_detection'] = frame_count
    else:
        results = cache['last_detections']
    
    cup_detections = filter_cup_detections(results)
    
    if len(cup_detections) > 0:
        roi_mask = create_roi_mask(img.shape, cup_detections)
        roi = cv2.bitwise_and(img, img, mask=roi_mask)
    else:
        roi = None
    
    start_time = time.time()
    depth_map_colorized, depth_map_raw = estimate_depth_midas(img, midas_model, transform, attention_model, device, roi)
    depth_processing_time = time.time() - start_time
    
    depth_display = cv2.resize(depth_map_colorized, (img.shape[1] // 3, img.shape[0] // 3))
    
    result_img = img.copy()
    result_img[0:depth_display.shape[0], 0:depth_display.shape[1]] = depth_display
    
    for box in cup_detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf = box.conf[0]
        
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        pixel_size = max(x2 - x1, y2 - y1)
        
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        distance_text = "Unknown"
        
        if depth_map_raw is not None:
            if 0 <= center_y < depth_map_raw.shape[0] and 0 <= center_x < depth_map_raw.shape[1]:
                relative_depth = depth_map_raw[center_y, center_x]
                estimated_distance = calculate_real_distance(
                    relative_depth, 
                    focal_length=500,
                    actual_size=0.1,
                    pixel_size=pixel_size
                )
                distance_text = f"{estimated_distance:.2f}m"
        
        text = f"{label} {conf:.2f} Dist: {distance_text}"
        cv2.putText(result_img, text, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(result_img, f"Depth: {depth_processing_time:.2f}s", 
                (result_img.shape[1]-200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return result_img, cache

# Run the main loop
def main():
    url = "http://192.168.1.106:8080/shot.jpg"
    
    midas_model, transform, attention_model, device = initialize_midas()
    models = (midas_model, transform, attention_model, device)
    
    cache = None
    frame_count = 0
    
    while True:
        try:
            img_resp = requests.get(url, timeout=3)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            img = imutils.resize(img, width=1000)
            
            result_img, cache = process_frame(img, models, cache, frame_count)
            frame_count += 1
            
            cv2.imshow("Camera with Attention-Based Depth Estimation", result_img)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            time.sleep(1)
        
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
