import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# Load an audio file
audio_path = '/Users/batputer/Library/CloudStorage/OneDrive-IITKanpur/Semester_5/EE604/Assignment_1/Q2/test/cardboard1.mp3'
y, sr = librosa.load(audio_path)

# Compute the spectrogram
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert the spectrogram to a grayscale image
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()
import cv2

# Read the spectrogram image
spectrogram_image = cv2.imread('your_spectrogram_image.png', cv2.IMREAD_GRAYSCALE)

# Apply image processing techniques (e.g., thresholding)
_, binary_image = cv2.threshold(spectrogram_image, threshold_value, max_value, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around detected objects
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(spectrogram_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Object Detection', spectrogram_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
