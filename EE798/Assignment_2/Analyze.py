import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd

# Function to record and analyze spoken sounds and save the plots
def record_and_analyze_sounds(save_plots=False):
    sample_rate = 22050
    duration = 1.0

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    plt.figure(figsize=(12, 6))

    vowel_sounds = ['aa', 'ee', 'o', 'u']
    for i, vowel in enumerate(vowel_sounds):
        print(f"Speak '{vowel}' sound and press Enter...")
        input()
        sound = sd.rec(int(sample_rate * duration), sample_rate, channels=1, dtype='float64')
        sd.wait()

        plt.subplot(4, 4, i + 1)
        D_dB = 20 * np.log10(np.abs(librosa.stft(sound[:, 0], hop_length=512, n_fft=1024)) + 1e-6)
        librosa.display.specshow(D_dB, x_axis='time', y_axis='log', sr=sample_rate)
        plt.title(f"Spectrogram of {vowel} (dB Scale)")
        plt.colorbar(format="%+2.0f dB")

        plt.subplot(4, 4, i + 5)
        D_mel = librosa.feature.melspectrogram(y=sound[:, 0], sr=sample_rate, n_fft=1024, hop_length=512, n_mels=128)
        D_mel_dB = 20 * np.log10(D_mel + 1e-6)
        librosa.display.specshow(D_mel_dB, x_axis='time', y_axis='mel', sr=sample_rate)
        plt.title(f"Spectrogram of {vowel} (Mel Scale)")
        plt.colorbar(format="%+2.0f dB")

        if save_plots:
            plt.savefig(f"spectrogram_{vowel}.png")

    consonant_sounds = ['t', 'th', 't', 'th']
    for i, consonant in enumerate(consonant_sounds):
        print(f"Speak '{consonant}' sound and press Enter...")
        input()
        sound = sd.rec(int(sample_rate * duration), sample_rate, channels=1, dtype='float64')
        sd.wait()

        plt.subplot(4, 4, i + 9)
        D_dB = 20 * np.log10(np.abs(librosa.stft(sound[:, 0], hop_length=512, n_fft=1024)) + 1e-6)
        librosa.display.specshow(D_dB, x_axis='time', y_axis='log', sr=sample_rate)
        plt.title(f"Spectrogram of {consonant} (dB Scale)")
        plt.colorbar(format="%+2.0f dB")

        plt.subplot(4, 4, i + 13)
        D_mel = librosa.feature.melspectrogram(y=sound[:, 0], sr=sample_rate, n_fft=1024, hop_length=512, n_mels=128)
        D_mel_dB = 20 * np.log10(D_mel + 1e-6)
        librosa.display.specshow(D_mel_dB, x_axis='time', y_axis='mel', sr=sample_rate)
        plt.title(f"Spectrogram of {consonant} (Mel Scale)")
        plt.colorbar(format="%+2.0f dB")

        if save_plots:
            plt.savefig(f"spectrogram_{consonant}.png")

    # plt.tight_layout()
    if save_plots:
        plt.savefig("all_spectrograms.png")
    else:
        plt.show()

# Call the function to record and analyze spoken sounds and save plots (if save_plots=True)
record_and_analyze_sounds(save_plots=True)
