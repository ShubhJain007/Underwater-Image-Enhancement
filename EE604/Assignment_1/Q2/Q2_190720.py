import cv2
import numpy as np
import librosa

def solution(audio_file):
# Load and preprocess audio files
# Load and preprocess audio files
    def extract_spectrogram_features(audio_file):
        y, sr = librosa.load(audio_file)
        # Compute the spectrogram
        spectrogram = np.abs(librosa.stft(y))
        return spectrogram

    # Define a function to classify spectrogram images
    def classify_spectrogram(spectrogram):
        # Apply your image processing and thresholding logic here
        # For demonstration purposes, let's assume a simple threshold on mean intensity
        mean_intensity = np.mean(spectrogram)
        if mean_intensity > 0.5:  # Adjust the threshold as needed
            return "metal"
        else:
            return "cardboard"

    # Feature extraction
    metal_spectrogram = extract_spectrogram_features(audio_file)
    # cardboard_spectrogram = extract_spectrogram_features('cardboard_brick.mp3')

    # Classify spectrogram images
    metal_label = classify_spectrogram(metal_spectrogram)
    # cardboard_label = classify_spectrogram(cardboard_spectrogram)
    # if metal_spectrogram=
    # print(f'Metal Brick: {metal_label}')
    # print(f'Cardboard Brick: {cardboard_label}')
    class_name = metal_label
    return class_name
