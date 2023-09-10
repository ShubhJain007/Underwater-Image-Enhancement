from PIL import Image
import numpy as np

# Open the image
image = Image.open('/Users/batputer/Library/CloudStorage/OneDrive-IITKanpur/Semester_5/EE604/Assignment_1/Q2/test/cardboard1_spectrogram.png')

# Convert to grayscale
image = image.convert('L')

# Convert to a NumPy array
image_array = np.array(image)

# Apply the log transformation
log_transformed_array = np.log1p(image_array)  # Adding 1 to avoid log(0)

# Scale the log-transformed array to 0-255 range
log_transformed_array = (255 * log_transformed_array / np.max(log_transformed_array)).astype(np.uint8)

# Create a new PIL image from the NumPy array
log_transformed_image = Image.fromarray(log_transformed_array)

# Save or display the log-transformed image
log_transformed_image.save('log_transformed_image.jpg')
