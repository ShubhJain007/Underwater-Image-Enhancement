import librosa
import sounddevice as sd
import time
def extract_beat(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file)

    # Extract beat frames using onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    
    # Convert beat frames to samples
    onset_samples = librosa.frames_to_samples(onset_frames)

    # Extract the first beat segment
    if len(onset_samples) >= 2:
        first_beat_start = onset_samples[0]
        first_beat_end = onset_samples[1]
    else:
        # Handle cases with only one onset frame
        first_beat_start = onset_samples[0]
        first_beat_end = len(y)
    
    extracted_beat = y[first_beat_start:first_beat_end]

    return extracted_beat, sr

def play_beat(extracted_beat, sr, beat_interval):
    
    while True:
        sd.play(extracted_beat, sr)
        # sd.wait(beat_interval)
        # sd.play(extracted_beat, sr)  # Play the beat again for continuous looping
        # sd.wait()
        time.sleep(beat_interval)

def main():
    audio_file = "/Users/batputer/Downloads/test.mp3"
    bpm = float(input("Enter the BPM value: "))
    beat_interval = 60 / bpm

    extracted_beat, sr = extract_beat(audio_file)
    
    play_beat(extracted_beat, sr, beat_interval)

if __name__ == "__main__":
    main()
