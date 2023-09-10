import librosa
import librosa.display
import numpy as np
import cv2

# Load and preprocess audio files
def extract_spectrogram_features(audio_file):
    y, sr = librosa.load(audio_file)
    # Compute the spectrogram
    spectrogram = np.abs(librosa.stft(y))
    return spectrogram

# Convert the spectrogram to an image
def spectrogram_to_image(spectrogram):
    # Normalize the spectrogram to the range [0, 255]
    spectrogram = 255 * (spectrogram / np.max(spectrogram))
    # Convert to 8-bit unsigned integer
    spectrogram = spectrogram.astype(np.uint8)
    # Create a grayscale image
    image = cv2.cvtColor(spectrogram, cv2.COLOR_GRAY2BGR)
    return image

# Calculate mean intensity from the image
def calculate_mean_intensity(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the mean intensity
    mean_intensity = np.mean(gray_image)  # Normalize to [0, 1]
    return mean_intensity

# Feature extraction
metal_spectrogram = extract_spectrogram_features('test/metal_banging2.mp3')
cardboard_spectrogram = extract_spectrogram_features('test/cardboard4.mp3')

# Convert spectrograms to images
metal_image = spectrogram_to_image(metal_spectrogram)
cardboard_image = spectrogram_to_image(cardboard_spectrogram)

# Calculate mean intensity
metal_intensity = calculate_mean_intensity(metal_image)
cardboard_intensity = calculate_mean_intensity(cardboard_image)

print(f'Metal Brick Mean Intensity: {metal_intensity:.2f}')
print(f'Cardboard Brick Mean Intensity: {cardboard_intensity:.2f}')
