# License Plate Recognition with  YOLOv8, EasyOCR, and MongoDB (Python Only)
# ===============================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
import torch
import time
import requests
from datetime import datetime
import uuid
from db_connector import MongoDBConnector

class GPSProvider:
    """
    Class to provide GPS coordinates.
    This is a simple implementation - in a real system, you would
    integrate with actual GPS hardware or a GPS service.
    """
    def __init__(self, mock_coordinates=None):
        """
        Initialize GPS provider
        
        Args:
            mock_coordinates (tuple): Optional mock coordinates (longitude, latitude) for testing
        """
        self.mock_coordinates = mock_coordinates
        
    def get_current_location(self):
        """
        Get current GPS coordinates
        
        Returns:
            tuple: (longitude, latitude)
        """
        if self.mock_coordinates:
            # Return mock coordinates for testing
            return self.mock_coordinates
        
        try:
            # In a real implementation, you would get actual GPS data here
            # For example, using a GPS module with the gps library
            # Example with Raspberry Pi GPS hat:
            # 
            # import gps
            # session = gps.gps("localhost", "2947")
            # session.stream(gps.WATCH_ENABLE | gps.WATCH_NEWSTYLE)
            # report = session.next()
            # if hasattr(report, 'lon') and hasattr(report, 'lat'):
            #     return (report.lon, report.lat)
            
            # This is just a placeholder using a free IP geolocation API
            # Not accurate for actual vehicle tracking but useful for testing
            response = requests.get('https://ipapi.co/json/')
            data = response.json()
            return (data['longitude'], data['latitude'])
        except Exception as e:
            print(f"Error getting GPS coordinates: {e}")
            # Return default coordinates if unable to get actual location
            return (0.0, 0.0)


class LicensePlateRecognizer:
    def __init__(self, model_path='./license_plate_detector.pt', ocr_langs=['en'], 
                 gps_provider=None, db_connector=None, save_dir='./plate_images'):
        """
        Initialize the license plate recognizer with pretrained models
        
        Args:
            model_path (str): Path to YOLOv8 model weights
            ocr_langs (list): Languages for EasyOCR
            gps_provider: GPS provider instance
            db_connector: MongoDB connector instance
            save_dir (str): Directory to save plate images
        """
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(ocr_langs, gpu=torch.cuda.is_available())
        
        # Detection parameters
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        
        # GPS provider for location tracking
        self.gps_provider = gps_provider or GPSProvider()
        
        # MongoDB connector
        self.db_connector = db_connector or MongoDBConnector()
        
        # Directory to save plate images
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def preprocess_license_plate(self, plate_img):
        """
        Preprocess the license plate image for better OCR results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return morph
    
    def recognize_plate(self, plate_img):
        """
        Recognize text on the license plate using EasyOCR
        """
        # Preprocess the plate image
        processed_img = self.preprocess_license_plate(plate_img)
        
        # Perform OCR
        results = self.reader.readtext(processed_img)
        
        # Extract text
        plate_text = ""
        for _, text, conf in results:
            # Filter out results with low confidence
            if conf > 0.15:
                plate_text += text + " "
        
        return plate_text.strip()
    
    def save_plate_image(self, plate_img, plate_text):
        """
        Save the license plate image with a unique filename
        
        Returns:
            str: Path to the saved image
        """
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean the plate text for use in filename
        sanitized_text = ''.join(c if c.isalnum() else '_' for c in plate_text)
        # Add a UUID to ensure uniqueness
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{sanitized_text}_{timestamp}_{unique_id}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save the image
        cv2.imwrite(filepath, plate_img)
        
        return filepath
    
    def detect_and_recognize(self, image, class_filter=None, save_to_db=True):
        """
        Detect license plates in an image and recognize the text
        
        Args:
            image: Input image
            class_filter: Specific class ID to filter
            save_to_db: Whether to save detections to MongoDB
            
        Returns:
            processed_img: Image with license plates and text drawn
            plates_info: List of dictionaries containing plate information
        """
        # Make a copy of the image
        processed_img = image.copy()
        
        # Detect license plates
        results = self.model.predict(image, conf=self.conf_threshold, iou=self.iou_threshold)
        
        plates_info = []
        
        # Get current GPS coordinates
        coordinates = self.gps_provider.get_current_location()
        
        # Process detection results
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # If class filter is specified, check if this detection matches
                if class_filter is not None and box.cls[0].item() != class_filter:
                    continue
                
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Extract license plate region
                plate_img = image[y1:y2, x1:x2]
                
                # Skip if plate image is empty
                if plate_img.size == 0:
                    continue
                
                # Recognize text on the plate
                plate_text = self.recognize_plate(plate_img)
                
                # Skip if no text was recognized
                if not plate_text:
                    continue
                
                # Save the plate image
                plate_image_path = self.save_plate_image(plate_img, plate_text)
                
                # Draw bounding box
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add text above the bounding box
                cv2.putText(processed_img, plate_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Confidence of detection
                confidence = float(box.conf)
                
                # Save plate information
                plate_info = {
                    'bbox': (x1, y1, x2, y2),
                    'text': plate_text,
                    'confidence': confidence,
                    'image_path': plate_image_path,
                    'coordinates': coordinates
                }
                plates_info.append(plate_info)
                
                # Save to MongoDB if requested
                if save_to_db and plate_text:
                    self.db_connector.save_detection(
                        plate_text, 
                        coordinates, 
                        plate_image_path, 
                        confidence,
                        source='image'
                    )
        
        return processed_img, plates_info
    
    def process_image(self, image_path, class_filter=None, save_to_db=True):
        """
        Process a single image file
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Detect and recognize license plates
        processed_img, plates_info = self.detect_and_recognize(
            image, class_filter, save_to_db=save_to_db
        )
        
        return processed_img, plates_info
    
    def process_video(self, video_path, output_path=None, display=True, 
                      class_filter=None, save_to_db=True, frame_interval=5):
        """
        Process a video file
        
        Args:
            frame_interval: Process every Nth frame (default: 5)
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Set up video writer if output path is provided
        out = None
        if output_path:
            # Try safe MP4 codec; fallback to AVI if platform doesn't support
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # more widely supported than 'mp4v'
            if output_path.endswith('.mp4'):
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            elif output_path.endswith('.avi'):
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
            else:
                raise ValueError("Unsupported output video format. Use .mp4 or .avi")
        
        frame_count = 0
        all_plates_info = []
        
        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame for efficiency
            if frame_count % frame_interval == 0:
                # Get current GPS coordinates
                coordinates = self.gps_provider.get_current_location()
                
                # Detect and recognize license plates
                processed_frame, plates_info = self.detect_and_recognize(
                    frame, class_filter, save_to_db=False  # We'll save manually below
                )
                
                # Save plate information and to MongoDB
                if plates_info:
                    frame_plates = {
                        'frame': frame_count,
                        'plates': plates_info,
                        'coordinates': coordinates
                    }
                    all_plates_info.append(frame_plates)
                    
                    # Save each plate to MongoDB
                    if save_to_db:
                        for plate_info in plates_info:
                            self.db_connector.save_detection(
                                plate_info['text'],
                                coordinates,
                                plate_info['image_path'],
                                plate_info['confidence'],
                                source='video',
                                frame_number=frame_count
                            )
                
                # Write frame to output video
                if out is not None:
                    out.write(processed_frame)
            else:
                # For skipped frames, still write the original frame to output if needed
                if out is not None:
                    out.write(frame)
            
            frame_count += 1
        
        # Release resources
        cap.release()
        if out is not None:
            out.release()
        
        return all_plates_info


# Usage examples for both image and video processing

def test_on_image(model_path, image_path, output_path=None, class_filter=None, 
                  mock_gps=None, save_to_db=True):
    """
    Test the license plate recognition on a single image
    
    Args:
        model_path (str): Path to YOLOv8 model weights
        image_path (str): Path to the test image
        output_path (str, optional): Path to save the output image
        class_filter (int, optional): Class ID to filter detections
        mock_gps (tuple, optional): Mock GPS coordinates (longitude, latitude)
        save_to_db (bool): Whether to save detections to MongoDB
    """
    # Initialize GPS provider with mock coordinates if provided
    gps_provider = GPSProvider(mock_coordinates=mock_gps)
    
    # Initialize MongoDB connector
    db_connector = MongoDBConnector()
    
    # Initialize license plate recognizer
    lp_recognizer = LicensePlateRecognizer(
        model_path=model_path,
        gps_provider=gps_provider,
        db_connector=db_connector
    )
    
    # Process the image
    processed_img, plates_info = lp_recognizer.process_image(
        image_path, class_filter, save_to_db=save_to_db
    )
    
    # Display results
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    plt.title('License Plate Detection and Recognition')
    plt.axis('off')
    
    # Print recognized license plates
    print(f"Detected {len(plates_info)} license plates:")
    for i, plate in enumerate(plates_info):
        print(f"Plate {i+1}: {plate['text']} (Confidence: {plate['confidence']:.2f})")
        print(f"  Location: {plate['coordinates']} (Image: {plate['image_path']})")
    
    # Save the output image if requested
    if output_path:
        cv2.imwrite(output_path, processed_img)
        print(f"Output image saved to {output_path}")
    
    plt.show()
    
    return plates_info

def test_on_video(model_path, video_path, output_path=None, class_filter=None, 
                  mock_gps=None, save_to_db=True):
    """
    Test the license plate recognition on a video
    
    Args:
        model_path (str): Path to YOLOv8 model weights
        video_path (str): Path to the test video
        output_path (str, optional): Path to save the output video
        class_filter (int, optional): Class ID to filter detections
        mock_gps (tuple, optional): Mock GPS coordinates (longitude, latitude)
        save_to_db (bool): Whether to save detections to MongoDB
    """
    # Initialize GPS provider with mock coordinates if provided
    gps_provider = GPSProvider(mock_coordinates=mock_gps)
    
    # Initialize MongoDB connector
    db_connector = MongoDBConnector()
    
    # Initialize license plate recognizer
    lp_recognizer = LicensePlateRecognizer(
        model_path=model_path,
        gps_provider=gps_provider,
        db_connector=db_connector
    )
    
    # Process the video
    print("Processing video...")
    start_time = time.time()
    plates_info = lp_recognizer.process_video(
        video_path, output_path=output_path, class_filter=class_filter, save_to_db=save_to_db
    )
    processing_time = time.time() - start_time
    
    # Print summary
    total_plates = sum(len(frame['plates']) for frame in plates_info)
    print(f"Video processing completed in {processing_time:.2f} seconds")
    print(f"Detected license plates in {len(plates_info)} frames")
    print(f"Total license plates detected: {total_plates}")
    
    # Print a sample of recognized plates (first 5)
    print("\nSample of recognized plates:")
    count = 0
    for frame_info in plates_info:
        for plate in frame_info['plates']:
            print(f"Frame {frame_info['frame']}: {plate['text']} (Confidence: {plate['confidence']:.2f})")
            print(f"  Location: {frame_info['coordinates']} (Image: {plate['image_path']})")
            count += 1
            if count >= 5:
                break
        if count >= 5:
            break
    
    return plates_info


# Example main function
if __name__ == "__main__":
    # For license plate specific models
    MODEL_PATH = "./license_plate_detector.pt"  # Replace with path to your model
    
    # Mock GPS coordinates for testing (longitude, latitude)
    # In a Raspberry Pi setup, you would likely use a GPS module instead
    MOCK_GPS = (-73.935242, 40.730610)  # Example: New York City
    
    # Image test
    IMAGE_PATH = "imgTest.jpg"  # Replace with your image path
    test_on_image(MODEL_PATH, IMAGE_PATH, "output_image.jpg", mock_gps=MOCK_GPS)
    
    # Video test
    VIDEO_PATH = "mycarplate.mp4"  # Replace with your video path
    output_path = "output_video.avi"  # use .avi if .mp4 fails
    test_on_video(MODEL_PATH, VIDEO_PATH, output_path, mock_gps=MOCK_GPS)