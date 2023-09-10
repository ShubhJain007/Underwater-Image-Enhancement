import cv2
import numpy as np

def solution(image_path):
    image= cv2.imread(image_path)
    padding = (10,10,10,10)  

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
