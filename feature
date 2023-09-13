import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import scipy.signal as signal
from scipy.signal import stft
import pandas as pd
#from python_speech_features import logfbank
from sklearn.svm import SVC
start=time.perf_counter()
# video files
directory = ''

all_mfcc_features = []
labels = []
all_mfcc32_features = []
all_mfcc64_features = []
all_fft_features = []
all_fbank_features = []
all_labels = []
for filename in os.listdir(directory):
    if filename.endswith('.wav'): 
        # Define a dictionary to map letters to numerical values
        label = int(filename.replace('type', '').replace('.wav', ''))
        #labels.append(label)
        filepath = os.path.join(directory, filename)
        y, sr = librosa.load(filepath)
        #stft
        D = librosa.stft(y)
        S_dB_y = librosa.amplitude_to_db(abs(D), ref=np.max)
        # Calculating the amplitude envelope
        frame_size = 100
        hop_length = 10
        def amplitude_envelope(signal, frame_size, hop_length):
            return np.array([max(signal[i:i + frame_size]) for i in range(0, signal.size, hop_length)])
        # Amplitude Envelope for individual genre
        ae_y = amplitude_envelope(y, frame_size, hop_length)
        frames = range(0, ae_y.size)
        t = librosa.frames_to_time(frames, hop_length=hop_length)

        plot the Amplitude Envelope
        plt.subplot(1, 1, 1)
        librosa.display.waveshow(ae_y, alpha=0.5)
        plt.plot(tee , rms_db , color="r")
        plt.plot(t, ae_y, color="r")
        plt.title('Amplitude Envelope')
        plt.tight_layout()
        plt.show()

        cutoff_freq = 17640
        sampling_rate = 44100
        num_taps = 800
        lpf = signal.firwin(num_taps, cutoff_freq, fs=sampling_rate, pass_zero=True)
        smoothed_ae_y = signal.filtfilt(lpf, [1], ae_y)
        threshold = 0.1
        smoothed_ae_y[smoothed_ae_y < threshold] = 0

        # plot original and smoothed amplitude envelope
        frames = range(0, ae_y.size)
        t = librosa.frames_to_time(frames, hop_length=hop_length)
        # Find zero crossings in the first derivative
        diff_smoothed_ae_y = np.diff(smoothed_ae_y)
        zero_crossings = np.argwhere(np.diff(np.sign(diff_smoothed_ae_y)) != 0).flatten()
        # Initialize lists to store zero crossings
        zeros_x = []
        zeros_y = []
        # Locate zero crossings satisfying criteria
        for i, zero_idx in enumerate(zero_crossings):
            if diff_smoothed_ae_y[zero_idx] > 0 and diff_smoothed_ae_y[zero_idx + 1] < 0:
                # Add zero crossing to lists
                zeros_x.append(t[zero_idx])
                zeros_y.append(smoothed_ae_y[zero_idx])
        # Plot amplitude envelope with zero crossings
        plt.figure(figsize=(10, 4))
        plt.plot(t, ae_y, alpha=0.5, color='r', label='Original')
        plt.plot(t, smoothed_ae_y, label='Smoothed')
        plt.plot(zeros_x, zeros_y, 'ro', markersize=10, label='Zero crossings')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Amplitude Envelope with Zero Crossings')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #window
        window = signal.windows.hamming(frame_size)
        nperseg = frame_size
        noverlap = frame_size / 2
        frame_size = int(0.04 * sr) + int(0.15 * sr)
        frame_stride = int(0.01 * sr)
        nfft = None

        # cut frame
        frames = []
        for i, zero_idx in enumerate(zero_crossings):
            if diff_smoothed_ae_y[zero_idx] > 0 and diff_smoothed_ae_y[zero_idx + 1] < 0:
                # Calculate frame start and end indices
                frame_start = max(0, zero_idx - int(0.04 * sr))
                frame_end = min(len(y), zero_idx + int(0.15 * sr))
                # Extract frame
                frame = y[frame_start:frame_end]
                # Store frame in list
                frames.append(frame)
        peaks = librosa.util.peak_pick(y, pre_max=int(sr * 0.04), post_max=int(sr * 0.15),
                                       pre_avg=int(sr * 0.1), post_avg=int(sr * 0.1),
                                       delta=0.08, wait=0.001)
        # plot
        frame_size = int(0.04 * sr) + int(0.15 * sr)
        frame_stride = int(0.001 * sr)
        for i in range(len(peaks) - 1):
            start = peaks[i] - int(0.04 * sr)
            end = peaks[i] + int(0.15 * sr)
            if end > len(y):
                break
            frame = y[start:end]
            plt.figure()
            plt.plot(frame)
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.title(f"Frame {i + 1}")
            plt.show()
        frames.append(frame)
        max_frame_length = max(len(frame) for frame in frames)
        frames = [np.pad(frame, (0, max_frame_length - len(frame))) for frame in frames]
        windowed_frames = [frame * np.hamming(len(frame)) for frame in frames]
        # print the number of frames
        frames_array = np.array(frames)
        num_frames = frames_array.shape[0]
        print("the number of frames:", num_frames)
        labels_array = np.array(labels)
        np.save('frames.npy', frames_array)
        import numpy as np
        import librosa

        # Loop over each frame feature
        for frame in frames:
            mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
            mfcc_32 = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=32)
            mfcc_64 = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=64)
            fft_feature = np.abs(np.fft.fft(frame))
            # FBANK
            melspec = librosa.feature.melspectrogram(y=frame, sr=sr)
            all_mfcc_features.append(mfcc)
            all_mfcc32_features.append(mfcc_32)
            all_mfcc64_features.append(mfcc_64)
            all_fft_features.append(fft_feature)
            all_fbank_features.append(melspec)
        
        mfcc_features_array = np.array(all_mfcc_features)
        mfcc32_features_array = np.array(all_mfcc32_features)
        mfcc64_features_array = np.array(all_mfcc64_features)
        fft_features_array = np.array(all_fft_features)
        fbank_features_array = np.array(all_fbank_features)
        # save
        np.save("mfcc_features.npy", mfcc_features_array)
        np.save("mfcc32_features.npy", mfcc32_features_array)
        np.save("mfcc64_features.npy", mfcc64_features_array)
        np.save("fft_features.npy", fft_features_array)
        np.save("fbank_features.npy", fbank_features_array)
        
        label = int(filename.replace('type', '').replace('.wav', ''))  # Assign the label based on your criteria
        all_labels.extend([label] * num_frames)
            # label file
        labels_file = f"labels_{label}.npy"
        np.save(labels_file, all_labels)

labels = np.load('labels_9.npy')
features=np.load('mfcc_features.npy')
features32=np.load('mfcc32_features.npy')
features64=np.load('mfcc64_features.npy')
fft=np.load('fft_features.npy')
fbank=np.load('fbank_features.npy')

data_dict = {
    "labels": labels,
    "mfcc_features": features,
    "mfcc32_features": features32,
    "mfcc64_features": features64,
    "fft_features": fft,
    "fbank_features": fbank
}

# print information
for name, array in data_dict.items():
    print(f"{name}:")
    print(f"  Number of items (length): {array.shape[0]}")

  
    if len(array.shape) > 1:
        print(f"  Number of features: {array.shape[1]}")
    print()
print(label)
