import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the spectrogram image (replace 'spectrogram.png' with your spectrogram image file)
spectrogram = cv2.imread('test/metal_banging1_spectrogram.png', cv2.IMREAD_GRAYSCALE)

# Apply Sobel edge detection to the spectrogram
sobel_x = cv2.Sobel(spectrogram, cv2.CV_64F, 1, 0, ksize=1)  # Sobel operator for horizontal edges
sobel_y = cv2.Sobel(spectrogram, cv2.CV_64F, 0, 1, ksize=1)  # Sobel operator for vertical edges

# Combine the horizontal and vertical edges to get the magnitude
edges_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Display the original spectrogram and the Sobel edges
plt.subplot(131), plt.imshow(spectrogram, cmap='gray'), plt.title('Original Spectrogram')
plt.subplot(132), plt.imshow(np.abs(sobel_x), cmap='gray'), plt.title('Sobel X')
plt.subplot(133), plt.imshow(np.abs(sobel_y), cmap='gray'), plt.title('Sobel Y')
plt.figure()
plt.imshow(edges_magnitude, cmap='gray'), plt.title('Sobel Edges (Magnitude)')
plt.show()
