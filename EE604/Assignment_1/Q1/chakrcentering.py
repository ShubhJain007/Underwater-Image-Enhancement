import cv2
import numpy as np
from chakr import *
import matplotlib.pyplot as plt
image_path ='test/1.png'

def solution(image_path):
    image= cv2.imread(image_path)
    padding = (10,10,10,10)  # Adjust the values as needed

    image = cv2.copyMakeBorder(image, padding[2], padding[3], padding[0], padding[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    contours, _ = cv2.findContours(gray_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_image = np.zeros_like(image)
    filled_image.fill(0)  # Fill with white color

    cv2.fillPoly(filled_image, contours, (255, 255, 255))  # Fill with black color

    blurred_image = cv2.GaussianBlur(filled_image, (11, 11), 9)  # Adjust the kernel size as needed

# Display the padded image
    # cv2.imshow('Padded Image', blurred_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  
    img = blurred_image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 10, 0.02, 70)
    corners = np.int0(corners)

    corners = sorted(corners, key=lambda x: x.ravel()[1])

    top_left = corners[0].ravel()
    top_right = corners[1].ravel()

    if top_left[0] > top_right[0]:
        top_left, top_right = top_right, top_left

    bottom_right = corners[2].ravel()
    bottom_left = corners[3].ravel()

    if bottom_left[0] > bottom_right[0]:
        bottom_left, bottom_right = bottom_right, bottom_left

# Print the sorted corner coordinates
    # print(f"Top Left: ({top_left[0]}, {top_left[1]})")
    # print(f"Top Right: ({top_right[0]}, {top_right[1]})")
    # print(f"Bottom Left: ({bottom_left[0]}, {bottom_left[1]})")
    # print(f"Bottom Right: ({bottom_right[0]}, {bottom_right[1]})")

# Draw circles on the corners
# cv2.circle(img, (top_left[0], top_left[1]), 3, 255, -1)
# cv2.circle(img, (top_right[0], top_right[1]), 3, 255, -1)
# cv2.circle(img, (bottom_left[0], bottom_left[1]), 3, 255, -1)
# cv2.circle(img, (bottom_right[0], bottom_right[1]), 3, 255, -1)

    # plt.imshow(img), plt.show()
    img = cv2.imread(image_path)

    pts1 = np.float32([top_left-10, top_right-10.6, bottom_left-9, bottom_right-10])
    pts2 = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

# Perform Perspective Transformation
    dst = cv2.warpPerspective(img, M, (600,600))
    # plt.subplot(121), plt.imshow(img), plt.title('Input')
    # plt.subplot(122), plt.imshow(dst), plt.title('Output')
    # plt.show()
    # Create a window with a fixed size and display the output image
    # cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Output', 600, 600)
    # cv2.imshow('rgrgr', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    image = dst
    return dst

def solution3(image):
    
    padding = (10,10,10,10)  # Adjust the values as needed

    image = cv2.copyMakeBorder(image, padding[2], padding[3], padding[0], padding[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    contours, _ = cv2.findContours(gray_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_image = np.zeros_like(image)
    filled_image.fill(0)  # Fill with white color

    cv2.fillPoly(filled_image, contours, (255, 255, 255))  # Fill with black color

    blurred_image = cv2.GaussianBlur(filled_image, (11, 11), 9)  # Adjust the kernel size as needed

# Display the padded image
    # cv2.imshow('Padded Image', blurred_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  
    img = blurred_image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 10, 0.02, 70)
    corners = np.int0(corners)

    corners = sorted(corners, key=lambda x: x.ravel()[1])

    top_left = corners[0].ravel()
    top_right = corners[1].ravel()

    if top_left[0] > top_right[0]:
        top_left, top_right = top_right, top_left

    bottom_right = corners[2].ravel()
    bottom_left = corners[3].ravel()

    if bottom_left[0] > bottom_right[0]:
        bottom_left, bottom_right = bottom_right, bottom_left

# Print the sorted corner coordinates
    # print(f"Top Left: ({top_left[0]}, {top_left[1]})")
    # print(f"Top Right: ({top_right[0]}, {top_right[1]})")
    # print(f"Bottom Left: ({bottom_left[0]}, {bottom_left[1]})")
    # print(f"Bottom Right: ({bottom_right[0]}, {bottom_right[1]})")

# Draw circles on the corners
# cv2.circle(img, (top_left[0], top_left[1]), 3, 255, -1)
# cv2.circle(img, (top_right[0], top_right[1]), 3, 255, -1)
# cv2.circle(img, (bottom_left[0], bottom_left[1]), 3, 255, -1)
# cv2.circle(img, (bottom_right[0], bottom_right[1]), 3, 255, -1)

    # plt.imshow(img), plt.show()
    img = image

    pts1 = np.float32([top_left-1, top_right-1, bottom_left-1, bottom_right-1])
    pts2 = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

# Perform Perspective Transformation
    dst = cv2.warpPerspective(img, M, (600,600))
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()
    # Create a window with a fixed size and display the output image
    # cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Output', 600, 600)
    # cv2.imshow('rgrgr', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    image = dst
    return dst

def move_chakra_center(image, original_center):
    # Load the image

    
    # Define the original center of the chakra
    # original_center = (image.shape[1] // 2, image.shape[0] // 2)
    
    # Define the desired center position
    desired_center = (300, 300)
    
    # Calculate the translation matrix to move the chakra center to the desired position
    translation_matrix = np.float32([[1, 0, desired_center[0] - original_center[0]],
                                     [0, 1, desired_center[1] - original_center[1]]])
    
    # Apply the translation to the image
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    
    return translated_image

# Example usage to move the chakra center


def solution2(image, detected_center):
    # image = cv2.imread(image_path)
    detected_center = np.array(detected_center)

    # Calculate the translation matrix to move the detected center to (300, 300)
    translation_matrix = np.float32([[1, 0, 300 - detected_center[0]],
                                     [0, 1, 300 - detected_center[1]]])
    
    # Apply the translation to the image
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    
    # Define the corners of the chakra's bounding box
    top_left = (300 - 150, 300 - 150)
    top_right = (300 + 150, 300 - 150)
    bottom_left = (300 - 150, 300 + 150)
    bottom_right = (300 + 150, 300 + 150)
    
    # Define the destination points for the perspective transformation
    dst_points = np.float32([top_left, top_right, bottom_left, bottom_right])
    
    # Calculate the perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(detected_center.reshape(-1, 1, 2), dst_points)
    
    # Apply the perspective transformation to unwrap the chakra
    unwrapped_image = cv2.warpPerspective(translated_image, perspective_matrix, (600, 600))
    
    return unwrapped_image

# Example usage to detect the center and process the image
image = solution('test/1.png')
cv2.imshow('2 iteration',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
image2= solution3(image)
cv2.imshow('2 iteration',image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
center_x, center_y = find_chakra_center(image)
print(f'Detected Chakra Center: ({center_x}, {center_y})')
# output_image = 
output_image = solution2(image, (center_x, center_y))
cv2.imshow('Unwrapped Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
original = [center_x, center_y]
output_image = move_chakra_center(image,original)
cv2.imshow('Moved Chakra Center', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(output_image)
plt.axis('off')  # Turn off axis labels
plt.show()