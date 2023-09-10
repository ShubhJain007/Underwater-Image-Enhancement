import numpy as np
import pyaudio
import matplotlib.pyplot as plt

def estimate_pitch(signal, sample_rate):
    # Autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # Find the first peak after the first zero-crossing
    zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
    first_zero_crossing = zero_crossings[0]
    first_peak = np.argmax(autocorr[first_zero_crossing:]) + first_zero_crossing
    # print("Signal:", signal)

    # Calculate the pitch (in Hz)
    pitch = sample_rate / first_peak
    return pitch

def main():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for pitch...")

    while True:
        data = stream.read(CHUNK)
        signal = np.frombuffer(data, dtype=np.int16)

        pitch = estimate_pitch(signal, RATE)
        if (pitch>1000):
            print("Estimated Pitch (Hz):", pitch)

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()
