import requests
import numpy as np
import imutils
import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from PIL import Image
import urllib.request

# Set torch threads for optimal CPU performance
torch.set_num_threads(4)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Ensure input is 4D: [batch_size, channels, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(x_cat))
        return x * attention_map

class LiquidDetector:
    def __init__(self):
        # Initialize YOLO model with smaller image size for faster processing
        try:
            self.yolo_model = YOLO("yolov8m.pt")
        except Exception as e:
            print(f"Could not load YOLOv8m: {e}, falling back to YOLOv8n")
            self.yolo_model = YOLO("yolov8n.pt")
        
        # Initialize MiDaS for depth estimation
        self.midas_model, self.midas_transform, self.attention_model, self.device = self.initialize_midas()
        
        # Check if SAM is available and initialize with smaller model
        self.use_sam = self.initialize_sam()
        
        # Setup color-based liquid detection parameters
        self.liquid_hsv_ranges = {
            'water': ([90, 0, 0], [140, 80, 255]),  # Blue-ish transparent
            'coffee': ([10, 50, 20], [30, 255, 200]),  # Brown
            'tea': ([15, 30, 20], [35, 255, 180])  # Brown-yellow
        }
        
        # Detection cache to improve performance - increased interval
        self.cache = {
            'last_full_detection': 0,
            'last_detections': None,
            'detection_interval': 10,  # Increased from 5 to 10
            'last_cup_boxes': None
        }
        
        self.frame_count = 0

    def initialize_midas(self):
        """Initialize MiDaS depth estimation model"""
        try:
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            midas.to(device)
            midas.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midas_transforms.small_transform
            
            attention_model = SpatialAttention()
            attention_model.to(device)
            
            return midas, transform, attention_model, device
        except Exception as e:
            print(f"Error initializing MiDaS: {e}")
            # Create dummy models as fallback
            return None, None, None, torch.device("cpu")

    def download_file(self, url, save_path):
        """Download a file using urllib instead of wget"""
        try:
            print(f"Downloading {url} to {save_path}...")
            urllib.request.urlretrieve(url, save_path)
            print("Download complete!")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def initialize_sam(self):
        """Initialize Segment Anything Model with a smaller, faster model"""
        try:
            # Check if SAM libraries are installed
            import segment_anything
            from segment_anything import sam_model_registry, SamPredictor
            
            # Download SAM checkpoint if it doesn't exist (using ViT-B instead of ViT-H)
            sam_checkpoint = "sam_vit_b_01ec64.pth"
            if not os.path.exists(sam_checkpoint):
                print("Downloading SAM ViT-B checkpoint...")
                sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                success = self.download_file(sam_url, sam_checkpoint)
                if not success:
                    print("Warning: Could not download SAM model. Using fallback method.")
                    return False
            
            # Initialize SAM model
            sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
            sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
            return True
        except ImportError:
            print("SAM libraries not available, using conventional segmentation")
            return False
        except Exception as e:
            print(f"Error initializing SAM: {e}")
            print("Using fallback segmentation method instead.")
            return False

    def filter_cup_detections(self, results):
        """Filter YOLO detections to keep only cups and glasses"""
        cups = []
        if results is None:
            return cups
            
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self.yolo_model.names[cls_id]
                if "cup" in label.lower() or "glass" in label.lower() or "bottle" in label.lower():
                    cups.append(box)
        return cups

    def estimate_depth(self, img, roi=None):
        """Estimate depth using MiDaS with attention model"""
        if self.midas_model is None:
            # Return dummy depth map if MiDaS failed to initialize
            dummy_depth = np.zeros(img.shape[:2], dtype=np.uint8)
            dummy_colored = cv2.applyColorMap(dummy_depth, cv2.COLORMAP_INFERNO)
            return dummy_colored, dummy_depth
            
        try:
            # Resize image before depth estimation for speed
            h, w = img.shape[:2]
            max_dim = 384  # Reduced max dimension for faster processing
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img_resized = cv2.resize(img, new_size)
            
            # Convert image to tensor and move to device
            img_tensor = self.midas_transform(img_resized).to(self.device)
            
            # Apply attention if ROI is provided
            if roi is not None:
                # Ensure tensor has batch dimension [batch, channels, height, width]
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                
                # Apply attention model
                img_tensor = self.attention_model(img_tensor)
            
            # Run depth estimation
            with torch.no_grad():
                # Ensure tensor has appropriate dimensions for MiDaS
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                
                prediction = self.midas_model(img_tensor)
                
                # Resize depth map to original image size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy and visualize
            depth_map = prediction.cpu().numpy()
            depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
            depth_map_colorized = (depth_map_normalized * 255).astype(np.uint8)
            depth_map_colorized = cv2.applyColorMap(depth_map_colorized, cv2.COLORMAP_INFERNO)
            
            return depth_map_colorized, depth_map
            
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            dummy_depth = np.zeros(img.shape[:2], dtype=np.uint8)
            dummy_colored = cv2.applyColorMap(dummy_depth, cv2.COLORMAP_INFERNO)
            return dummy_colored, dummy_depth

    def create_roi_mask(self, img_shape, cup_detections, padding=20):
        """Create region of interest mask from cup detections"""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        for box in cup_detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_shape[1], x2 + padding)
            y2 = min(img_shape[0], y2 + padding)
            mask[y1:y2, x1:x2] = 255
        return mask

    def calculate_real_distance(self, depth_value, focal_length=500, actual_size=0.1, pixel_size=None):
        """Calculate real-world distance based on depth and object size"""
        if depth_value <= 0:
            return float('inf')
            
        estimated_distance = 1.0 / (depth_value + 1e-6) * 10
        
        if pixel_size is not None:
            size_based_distance = (actual_size * focal_length) / max(pixel_size, 1)
            final_distance = 0.7 * estimated_distance + 0.3 * size_based_distance
            return final_distance
        
        return estimated_distance

    def segment_cup_conventional(self, img, cup_box):
        """Conventional method to segment the cup when SAM is not available"""
        try:
            x1, y1, x2, y2 = map(int, cup_box.xyxy[0])
            cup_region = img[y1:y2, x1:x2]
            
            if cup_region.size == 0:
                return np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cup_region, cv2.COLOR_BGR2GRAY)
            
            # Apply GaussianBlur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold
            _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create mask
            mask = np.zeros_like(gray)
            
            # If contours found, use the largest one
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [largest_contour], 0, 255, -1)
            
            return mask
        except Exception as e:
            print(f"Error in conventional segmentation: {e}")
            # Return empty mask on error
            return np.zeros((y2-y1, x2-x1), dtype=np.uint8)

    def detect_liquid_in_cup(self, img, cup_box):
        """Detect liquid in a cup using color segmentation"""
        try:
            x1, y1, x2, y2 = map(int, cup_box.xyxy[0])
            cup_region = img[y1:y2, x1:x2]
            
            if cup_region.size == 0:
                return None, 0, "empty"
            
            # Get cup mask
            cup_mask = None
            if self.use_sam:
                try:
                    # Use SAM for precise segmentation if available
                    # Convert to RGB for SAM
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.sam_predictor.set_image(img_rgb)
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    input_point = np.array([[center_x, center_y]])
                    input_label = np.array([1])
                    
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        box=np.array([x1, y1, x2, y2]),
                        multimask_output=True
                    )
                    
                    # Get the best mask
                    best_mask_idx = np.argmax(scores)
                    cup_mask = masks[best_mask_idx][y1:y2, x1:x2]
                    cup_mask = (cup_mask * 255).astype(np.uint8)
                except Exception as e:
                    print(f"SAM segmentation failed: {e}, using conventional method")
                    cup_mask = self.segment_cup_conventional(img, cup_box)
            else:
                # Use conventional segmentation
                cup_mask = self.segment_cup_conventional(img, cup_box)
            
            if cup_mask is None or cup_mask.sum() == 0:
                # If no mask or empty mask, create a simple oval mask as fallback
                h, w = cup_region.shape[:2]
                cup_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(cup_mask, (w//2, h//2), (w//2-5, h//2-5), 0, 0, 360, 255, -1)
            
            # Convert to HSV for better color detection
            hsv_region = cv2.cvtColor(cup_region, cv2.COLOR_BGR2HSV)
            
            # Try to detect various liquids
            liquid_type = "empty"
            best_liquid_ratio = 0
            liquid_mask = None
            
            for liquid_name, (lower, upper) in self.liquid_hsv_ranges.items():
                lower = np.array(lower)
                upper = np.array(upper)
                
                # Create liquid mask based on color
                current_liquid_mask = cv2.inRange(hsv_region, lower, upper)
                
                # Apply cup mask to liquid mask
                current_liquid_mask = cv2.bitwise_and(current_liquid_mask, cup_mask)
                
                # Calculate the ratio of liquid pixels to cup mask pixels
                liquid_pixels = np.sum(current_liquid_mask > 0)
                cup_pixels = np.sum(cup_mask > 0)
                
                if cup_pixels > 0:
                    liquid_ratio = liquid_pixels / cup_pixels
                    if liquid_ratio > best_liquid_ratio and liquid_ratio > 0.05:  # Lower threshold to detect more cases
                        best_liquid_ratio = liquid_ratio
                        liquid_type = liquid_name
                        liquid_mask = current_liquid_mask
            
            # Calculate fill level
            fill_level = 0
            if liquid_mask is not None and cup_mask is not None and np.sum(cup_mask) > 0:
                fill_level = np.sum(liquid_mask > 0) / np.sum(cup_mask > 0)
            
            return liquid_mask, fill_level, liquid_type
        
        except Exception as e:
            print(f"Error in liquid detection: {e}")
            return None, 0, "unknown"

    def process_frame(self, img):
        """Process a frame to detect cups and liquids"""
        try:
            start_time = time.time()
            
            # If img is None, return a black image
            if img is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Resize input image for faster YOLO processing
            img_yolo = cv2.resize(img, (320, 320))  # Smaller size for YOLO
            
            # Run YOLO detection periodically to save processing time
            if self.frame_count % self.cache['detection_interval'] == 0:
                results = self.yolo_model(img_yolo, verbose=False)
                
                # Scale detection boxes to original image size
                orig_h, orig_w = img.shape[:2]
                yolo_h, yolo_w = img_yolo.shape[:2]
                scale_x, scale_y = orig_w / yolo_w, orig_h / yolo_h
                
                for result in results:
                    for box in result.boxes:
                        # Get original xyxy values
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Scale to original image size
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        # Update box coordinates
                        box.xyxy[0] = torch.tensor([x1, y1, x2, y2]).to(box.xyxy[0].device)
                
                self.cache['last_detections'] = results
                self.cache['last_full_detection'] = self.frame_count
            else:
                results = self.cache['last_detections']
            
            # Filter cup detections
            cup_detections = self.filter_cup_detections(results)
            
            # Create ROI for depth estimation
            if len(cup_detections) > 0:
                roi_mask = self.create_roi_mask(img.shape, cup_detections)
                roi = cv2.bitwise_and(img, img, mask=roi_mask)
                self.cache['last_cup_boxes'] = cup_detections
            else:
                roi = None
                if self.cache['last_cup_boxes'] is not None:
                    cup_detections = self.cache['last_cup_boxes']
            
            # Estimate depth
            depth_start = time.time()
            depth_map_colorized, depth_map_raw = self.estimate_depth(img, roi)
            depth_time = time.time() - depth_start
            
            # Create visualization image
            result_img = img.copy()
            
            # Add small depth map visualization to corner
            depth_display = cv2.resize(depth_map_colorized, (img.shape[1] // 4, img.shape[0] // 4))
            h, w = depth_display.shape[:2]
            result_img[0:h, 0:w] = depth_display
            
            # Process each detected cup
            for box in cup_detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.yolo_model.names[int(box.cls[0])]
                conf = box.conf[0]
                
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                pixel_size = max(x2 - x1, y2 - y1)
                
                # Draw bounding box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Detect liquid in cup
                liquid_mask, fill_level, liquid_type = self.detect_liquid_in_cup(img, box)
                
                # Calculate distance
                distance_text = "Unknown"
                if depth_map_raw is not None:
                    if 0 <= center_y < depth_map_raw.shape[0] and 0 <= center_x < depth_map_raw.shape[1]:
                        relative_depth = depth_map_raw[center_y, center_x]
                        estimated_distance = self.calculate_real_distance(
                            relative_depth, 
                            focal_length=500,
                            actual_size=0.1,
                            pixel_size=pixel_size
                        )
                        distance_text = f"{estimated_distance:.2f}m"
                
                # Add liquid information to label
                fill_text = ""
                if fill_level > 0:
                    fill_text = f" {liquid_type.capitalize()} {fill_level:.0%} full"
                
                text = f"{label} {conf:.2f} Dist: {distance_text}{fill_text}"
                cv2.putText(result_img, text, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # If we have a liquid mask, overlay it on the image
                if liquid_mask is not None and liquid_mask.size > 0:
                    try:
                        # Get the region where we'll overlay the liquid
                        region = result_img[y1:y2, x1:x2]
                        
                        # Create a colored overlay based on liquid type
                        if liquid_type == "water":
                            color = (255, 0, 0)  # Blue for water
                        elif liquid_type == "coffee":
                            color = (0, 165, 255)  # Brown for coffee
                        elif liquid_type == "tea":
                            color = (0, 215, 255)  # Light brown for tea
                        else:
                            color = (0, 255, 255)  # Default yellow
                        
                        # Create a colored mask
                        overlay = np.zeros_like(region)
                        for c in range(3):
                            overlay[:, :, c] = np.where(liquid_mask > 0, color[c], 0)
                        
                        # Apply overlay with transparency
                        cv2.addWeighted(overlay, 0.4, region, 0.6, 0, region)
                        result_img[y1:y2, x1:x2] = region
                    except Exception as e:
                        print(f"Error applying liquid overlay: {e}")
            
            # Add processing time information
            total_time = time.time() - start_time
            cv2.putText(result_img, f"Total: {total_time:.2f}s Depth: {depth_time:.2f}s", 
                        (result_img.shape[1]-300, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            self.frame_count += 1
            return result_img
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            # Return original image on error
            return img if img is not None else np.zeros((480, 640, 3), dtype=np.uint8)

def main():
    # Replace with your camera URL or use webcam
    try:
        # Try to use IP camera first
        url = "http://192.168.1.106:8080/shot.jpg"
        print("Trying to connect to IP camera...")
        response = requests.get(url, timeout=2)
        use_ip_camera = True
        print("IP camera connected!")
    except Exception as e:
        print(f"Failed to connect to IP camera: {e}")
        print("Using webcam instead")
        use_ip_camera = False
    
    detector = LiquidDetector()
    
    # Open webcam if no IP camera
    if not use_ip_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
    
    while True:
        try:
            if use_ip_camera:
                # Get image from IP camera
                img_resp = requests.get(url, timeout=3)
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                img = cv2.imdecode(img_arr, -1)
            else:
                # Get image from webcam
                ret, img = cap.read()
                if not ret:
                    print("Failed to get frame from webcam")
                    break
            
            img = imutils.resize(img, width=1000)
            result_img = detector.process_frame(img)
            cv2.imshow("Cup and Liquid Detection", result_img)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(1)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        elif key == ord('q'):
            break

    # Clean up
    if not use_ip_camera and 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()