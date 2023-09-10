import cv2
import numpy as np
input_image_path= 'test/3_a.png'
output_image_path = 'out.png'
image = cv2.imread(input_image_path)
cv2.imshow('image',image)
cv2.waitKey(0)
def realign_image(input_image_path, output_image_path):
    # Load the input image
    image = cv2.imread(input_image_path)
    # cv2.imshow('image',image)
    # Convert the image to grayscale for better edge detection
    gray = image

    # Perform edge detection using Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    print(edges)
    # Find lines in the image using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Calculate the average angle of detected lines
    angle_sum = 0.0
    num_lines = 0

    for line in lines:
        rho, theta = line[0]
        angle_sum += theta
        num_lines += 1

    average_angle = angle_sum / num_lines

    # Rotate the image to align text horizontally
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Save the aligned image
    cv2.imwrite(output_image_path, rotated_image)

if __name__ == "__main__":
    input_image_path = "input.jpg"  # Replace with the path to your input image
    output_image_path = "output.jpg"  # Replace with the desired output image path

    realign_image(input_image_path, output_image_path)

    print("Image realigned and saved as", output_image_path)

realign_image(input_image_path,output_image_path)