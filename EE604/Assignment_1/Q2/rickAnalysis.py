import librosa
import librosa.display
import numpy as np
import cv2

# Function to calculate regularity of a shape
def calculate_regularity(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    regularity = 1 / (1 + abs(aspect_ratio - 1))
    return regularity

# Load and preprocess audio files
def extract_spectrogram_features(audio_file):
    y, sr = librosa.load(audio_file)
    spectrogram = np.abs(librosa.stft(y))
    return spectrogram

# Convert the spectrogram to an image
def spectrogram_to_image(spectrogram):
    spectrogram = 255 * (spectrogram / np.max(spectrogram))
    spectrogram = spectrogram.astype(np.uint8)
    image = cv2.cvtColor(spectrogram, cv2.COLOR_GRAY2BGR)
    return image

# Extract brick-like shapes from the spectrogram
def extract_brick_shapes(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    brick_like_contours = []
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            brick_like_contours.append(contour)
    
    result_image = image.copy()
    cv2.drawContours(result_image, brick_like_contours, -1, (0, 0, 255), 2)
    
    return result_image

# Function to inspect brick quality and assign scores
def inspect_brick_quality(audio_file, regularity_threshold=0.8):
    # Feature extraction
    spectrogram = extract_spectrogram_features(audio_file)
    
    # Convert spectrogram to an image
    spectrogram_image = spectrogram_to_image(spectrogram)
    
    # Extract and analyze brick-like shapes
    brick_shapes_image = extract_brick_shapes(spectrogram_image)
    
    # Convert the brick shapes image to grayscale for quality inspection
    gray_brick_shapes = cv2.cvtColor(brick_shapes_image, cv2.COLOR_BGR2GRAY)
    
    # Find contours in the grayscale brick shapes image
    contours, _ = cv2.findContours(gray_brick_shapes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate regularity for each detected shape and assign scores
    shape_scores = []
    for contour in contours:
        regularity = calculate_regularity(contour)
        # Assign scores based on regularity
        if regularity >= regularity_threshold:
            score = 10  # High-quality shape
        else:
            score = 5   # Medium-quality shape
        shape_scores.append(score)
    
    # Calculate the aggregate score for the entire spectrogram
    aggregate_score = sum(shape_scores)
    
    return aggregate_score

# Brick quality inspection and scoring for audio files
metal_score = inspect_brick_quality('test/metal_banging1.mp3')
cardboard_score = inspect_brick_quality('test/cardboard1.mp3')

print(f'Metal Brick Spectrogram Score: {metal_score}')
print(f'Cardboard Brick Spectrogram Score: {cardboard_score}')
