import cv2
import numpy as np
image_path ='test/1.png'
def find_chakra_center(image):
    # Load the image
    # image = cv2.imread(image_path)
    
    # Convert the image to grayscale for better edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to find the largest one (the chakra)
    chakra_contour = max(contours, key=cv2.contourArea)
    
    # Find the center of the chakra by calculating its moments
    M = cv2.moments(chakra_contour)
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    
    return center_x, center_y

# Example usage to detect the center
# center_x, center_y = find_chakra_center(image_path)
# print(f'Detected Chakra Center: ({center_x}, {center_y})')
