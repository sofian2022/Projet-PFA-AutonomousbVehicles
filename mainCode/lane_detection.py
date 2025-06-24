import cv2
import numpy as np
import time
from picamera2 import Picamera2
from libcamera import controls

# Global variables for left_line and right_line
left_line = None
right_line = None

# Step 2: Canny Transformation (Edge Detection)
def canny_transformation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


# Step 3: Region of Interest Definition
def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)
    
    # Adjust these values based on your camera view
    triangle = np.array([[ (0, height),
                           (0, height//3),
                           (width, height//3),
                           (width, height)]], np.int32)
    
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# Step 4: Display Lines and Color the Lane
def display_lines(image, lines):
    image_with_lines = np.zeros_like(image)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Ensure no overflow errors with coordinates
                if (x1 > image.shape[1] or x2 > image.shape[1] or 
                    x1 < 0 or x2 < 0 or 
                    y1 > image.shape[0] or y2 > image.shape[0] or
                    y1 < 0 or y2 < 0):
                    continue
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 10)

    return image_with_lines


# Step 5: Line Optimization
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1 * (3 / 5))
    
    # Avoid division by zero
    if slope == 0:
        slope = 0.1
        
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return [[x1, y1, x2, y2]]


# Step 5: Average Lines
def average_slope_intercept(image, lines):
    global left_line, right_line
    
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None
        
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Avoid division by zero
            if x2 - x1 == 0:
                continue
                
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            
            # Lines on the left
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    
    # Ensure we have detected lines
    result_lines = []
    
    # Process left lines if detected
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        result_lines.append(left_line)
    
    # Process right lines if detected
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
        result_lines.append(right_line)
    
    return result_lines if result_lines else None


# Function to color the area between the lanes and draw the lines
def create_lane_overlay(image, lines):
    lane_image = np.zeros_like(image)
    
    if lines is None or len(lines) < 2:
        return lane_image
    
    # Get the coordinates of both lines
    left_line = lines[0][0]
    right_line = lines[1][0]
    
    # Check for invalid coordinates
    height, width = image.shape[0], image.shape[1]
    
    # Ensure coordinates are within bounds
    for coord in [left_line, right_line]:
        for i in range(4):
            if i % 2 == 0:  # x-coordinate
                coord[i] = max(0, min(coord[i], width-1))
            else:  # y-coordinate
                coord[i] = max(0, min(coord[i], height-1))
    
    # Create a polygon representing the lane area
    try:
        # Create points for the polygon
        points = np.array([
            [left_line[0], left_line[1]],
            [left_line[2], left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ], np.int32)
        
        # Reshape the points
        points = points.reshape((-1, 1, 2))
        
        # Fill the polygon with green
        cv2.fillPoly(lane_image, [points], (0, 255, 0))
        
        # Draw the left line in red
        cv2.line(lane_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 10)
        
        # Draw the right line in blue
        cv2.line(lane_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 0, 0), 10)
    except Exception as e:
        print(f"Error creating lane overlay: {e}")
    
    return lane_image


def main():
    global left_line, right_line
    
    # Initialize the Picamera2
    picam2 = Picamera2()
    
    # Configure the camera
    config = picam2.create_preview_configuration(main={"size": (840, 680), "format": "RGB888"})
    picam2.configure(config)
    
    # Set camera controls (optional)
    picam2.set_controls({"AwbEnable": True, "AeEnable": True})
    
    # Start the camera
    picam2.start()
    
    # Allow camera to warm up
    time.sleep(2)
    
    # Create a video file for the output video
    frame_width = 640
    frame_height = 480
    size = (frame_width, frame_height)
    output_video = cv2.VideoWriter('./image/output.avi', 
                                  cv2.VideoWriter_fourcc(*'MJPG'), 
                                  10, size)
    
    print("Lane detection started. Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame from Picamera2
            frame = picam2.capture_array()
            
            # Step 2: Apply Canny Transformation
            canny_image = canny_transformation(frame)
    
            # Step 3: Apply Region of Interest Mask
            masked_image = region_of_interest(canny_image)
    
            # Step 4: Detect Lines with Hough Transform
            lines = cv2.HoughLinesP(masked_image, 
                                   rho=2, 
                                   theta=np.pi/180, 
                                   threshold=50,  
                                   minLineLength=40, 
                                   maxLineGap=5)
    
            # Step 5: Average and Optimize Detected Lines
            averaged_lines = average_slope_intercept(frame, lines)
    
            # Step 6: Create lane overlay with colored area
            if averaged_lines is not None and len(averaged_lines) >= 2:
                lane_overlay = create_lane_overlay(frame, averaged_lines)
            else:
                lane_overlay = np.zeros_like(frame)
            
            # Step 7: Combine Images and Display Final Output
            final_image = cv2.addWeighted(frame, 0.8, lane_overlay, 1, 1)
            
            # Write to output video
            output_video.write(final_image)
            
            # Display the processed image
            cv2.imshow("Lane Detection", final_image)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stopping the program")
    finally:
        # Release resources
        picam2.stop()
        output_video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()