import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
# Define the path to your MP3 audio file
mp3_audio_file = '/Users/batputer/Library/CloudStorage/OneDrive-IITKanpur/Semester_5/EE604/Assignment_1/Q2/test/cardboard1.mp3'

# Convert MP3 to WAV
wav_audio_file = mp3_audio_file.replace('.mp3', '.wav')
audio = AudioSegment.from_mp3(mp3_audio_file)
audio.export(wav_audio_file, format="wav")

# Load the WAV audio file
y, sr = librosa.load(wav_audio_file)

# Create the waveform plot
plt.figure(figsize=(10, 6))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
waveform_image_path = mp3_audio_file.replace('.mp3', '_waveform.png')

# Save the waveform plot
plt.savefig(waveform_image_path)
plt.close()

# Create the spectrogram plot
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
plt.figure(figsize=(10, 6))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
spectrogram_image_path = mp3_audio_file.replace('.mp3', '_spectrogram.png')

# Save the spectrogram plot
plt.savefig(spectrogram_image_path)
plt.close()

print(f'Waveform and spectrogram saved as:\n{waveform_image_path}\n{spectrogram_image_path}')
