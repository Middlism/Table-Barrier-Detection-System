import requests
import numpy as np
import imutils
import cv2
from ultralytics import YOLO
import torch

# Load YOLO model for object detection
model = YOLO("yolov8n.pt")

def filter_cup_detections(results, model):
    cups = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names(cls_id)
            if "cup" in label.lower() or glass in label.lower():
                cups.append(box)
    return cups

# Initialize MiDaS model for monocular depth estimation
def initialize_midas():
    # Load MiDaS model (small version for faster inference)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to("cuda" if torch.cuda.is_available() else "cpu")
    midas.eval()
    
    # Initialize transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    return midas, transform

# Function to estimate depth using MiDaS
def estimate_depth_midas(img, midas_model, transform):
    # Transform input for model
    input_batch = transform(img).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Make prediction
    with torch.no_grad():
        prediction = midas_model(input_batch)
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

# Function to estimate stereo depth (if you have calibrated stereo cameras)
def estimate_stereo_depth(left_img, right_img):
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Create stereo matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    # Compute disparity
    disparity = stereo.compute(left_gray, right_gray)
    
    # Normalize and convert to depth map
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_INFERNO)
    
    return depth_map, disparity_normalized

# Configure camera URL
url = "http://192.168.1.106:8080/shot.jpg"

# Choose depth estimation method (comment/uncomment as needed)
USE_MIDAS = True  # Set to False if you want to use stereo instead
USE_STEREO = False  # Set to True if you have calibrated stereo cameras

# Initialize MiDaS if needed
midas_model, midas_transform = initialize_midas() if USE_MIDAS else (None, None)

while True:
    # Get image from IP camera
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000)
    
    # Run object detection with YOLO
    results = model(img, verbose=False)
    
    # Process depth estimation
    depth_map_raw = None  # Initialize variable to store raw depth values
    
    if USE_MIDAS:
        depth_map_colorized, depth_map_raw = estimate_depth_midas(img, midas_model, midas_transform)
        
        # Display depth map alongside original image
        depth_display = cv2.resize(depth_map_colorized, (img.shape[1] // 3, img.shape[0] // 3))
        img[0:depth_display.shape[0], 0:depth_display.shape[1]] = depth_display
    
    elif USE_STEREO:
        # For stereo, you would need two camera inputs
        # This is a placeholder - you'd need to get right_img from a second camera
        right_img = img.copy()  # Replace with actual right camera image
        depth_map_colorized, disparity = estimate_stereo_depth(img, right_img)
        
        # Display depth map alongside original image
        depth_display = cv2.resize(depth_map_colorized, (img.shape[1] // 3, img.shape[0] // 3))
        img[0:depth_display.shape[0], 0:depth_display.shape[1]] = depth_display
    
    # Draw object detection results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = box.conf[0]
            
            # Calculate center of bounding box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get depth value at object center
            distance_text = "Unknown"
            
            if USE_MIDAS and depth_map_raw is not None:
                # Check if the point is within bounds
                if 0 <= center_y < depth_map_raw.shape[0] and 0 <= center_x < depth_map_raw.shape[1]:
                    # Get depth value at the center of the object
                    relative_depth = depth_map_raw[center_y, center_x]
                    
                    # Convert relative depth to an estimated distance in meters
                    # Note: This requires calibration for your specific camera
                    estimated_distance = 1.0 / (relative_depth + 1e-6) * 10
                    distance_text = f"{estimated_distance:.2f}m"
            
            elif USE_STEREO and 'disparity' in locals():
                # Convert disparity to distance using calibrated parameters
                # This is a simplification - real calculation depends on your stereo calibration
                if 0 <= center_y < disparity.shape[0] and 0 <= center_x < disparity.shape[1]:
                    disp_value = disparity[center_y, center_x]
                    if disp_value > 0:
                        # Assuming baseline of 0.1m and focal length of 500 pixels (adjust for your setup)
                        estimated_distance = (0.1 * 500) / (disp_value + 1e-6)
                        distance_text = f"{estimated_distance:.2f}m"
                    else:
                        distance_text = "Inf"
                else:
                    distance_text = "Out of bounds"
            
            text = f"{label} {conf:.2f} Dist: {distance_text}"
            cv2.putText(img, text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Camera with Depth Estimation", img)
    
    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()