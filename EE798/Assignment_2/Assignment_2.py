import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def audio_features(audio_file, window_sizes_ms=[20], hop_sizes_ms=[10], start_times_s=[0.0], save_fig=False):
    # Load the audio file using librosa
    audio_data, sample_rate = librosa.load(audio_file, sr=None)

    for window_size_ms in window_sizes_ms:
        for hop_size_ms in hop_sizes_ms:
            for start_time_s in start_times_s:
                # Convert window size and hop size to samples
                window_size = int(window_size_ms * sample_rate / 1000)
                hop_size = int(hop_size_ms * sample_rate / 1000)

                # Plot the waveform
                plt.figure(figsize=(12, 6))
                time = np.arange(len(audio_data)) / sample_rate
                plt.plot(time, audio_data)
                plt.title("Waveform")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.tight_layout()

                if save_fig:
                    plt.savefig(f"waveform_{window_size}_{hop_size}_{start_time_s}.png")
                else:
                    plt.show()

                # Plot the spectrogram
                plt.figure(figsize=(12, 6))
                Db = np.abs(librosa.stft(audio_data, n_fft=window_size, hop_length=hop_size))
                D_dB = 20 * np.log10(Db + 1e-6)  # Adding a small constant to avoid log(0)
                librosa.display.specshow(D_dB, x_axis='time', y_axis='log', sr=sample_rate)
                plt.colorbar(format="%+2.0f dB")
                plt.title(f"Spectrogram (Window: {window_size} samples, Hop: {hop_size} samples)")
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz)")
                plt.tight_layout()

                if save_fig:
                    plt.savefig(f"spectrogram_{window_size}_{hop_size}_{start_time_s}.png")
                else:
                    plt.show()

                # Plot the spectrum within the specified start time
                start_sample = int(start_time_s * sample_rate)
                end_sample = start_sample + window_size  # Consider one window size
                audio_slice = audio_data[start_sample:end_sample]

                spectrum = np.abs(librosa.stft(audio_slice, n_fft=window_size, hop_length=hop_size))
                spectrum = spectrum[:, 0]  

                frequency_axis = librosa.fft_frequencies(sr=sample_rate, n_fft=window_size)

                spectrum_dB = 20 * np.log10(spectrum + 1e-6)  

                plt.figure(figsize=(12, 6))
                plt.plot(frequency_axis, spectrum_dB)
                plt.title(f"Spectrum (Start Time: {start_time_s} s)")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Amplitude (dB)")
                plt.grid()
                plt.tight_layout()

                if save_fig:
                    plt.savefig(f"spectrum_{window_size}_{hop_size}_{start_time_s}.png")
                else:
                    plt.show()

audio_features("wine-glass-tuned-to-400hz.wav", save_fig=True)
