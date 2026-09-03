# ~/dev/vits/data_tools/audio_diff.py
import sys
import os
import librosa
import numpy as np

def analyze_file(filepath):
    # Load raw audio
    y, sr = librosa.load(filepath, sr=None)
    duration = len(y) / sr
    
    # 1. Calculate Peak and RMS (Loudness)
    peak = np.max(np.abs(y))
    peak_db = 20 * np.log10(peak + 1e-6)
    rms = np.sqrt(np.mean(y**2))
    rms_db = 20 * np.log10(rms + 1e-6)
    
    # 2. Spectral Centroid (Brightness/Presence)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # 3. Sub-bass Energy Ratio (0Hz - 80Hz plosive rumble)
    stft = np.abs(librosa.stft(y))
    frequencies = librosa.fft_frequencies(sr=sr)
    sub_80hz_indices = np.where(frequencies <= 80)
    sub_bass_energy = np.sum(stft[sub_80hz_indices, :])
    total_energy = np.sum(stft)
    sub_bass_ratio = (sub_bass_energy / (total_energy + 1e-6)) * 100

    return {
        "duration": duration,
        "peak_db": peak_db,
        "rms_db": rms_db,
        "centroid": centroid,
        "sub_bass_ratio": sub_bass_ratio
    }

def print_diff(file_org, file_enh):
    print("==========================================================")
    print("           SUTTAPLAYER ACOUSTIC QUALITY DIFF              ")
    print("==========================================================\n")
    
    org = analyze_file(file_org)
    enh = analyze_file(file_enh)
    
    print(f"Original: {os.path.basename(file_org)}")
    print(f"Enhanced: {os.path.basename(file_enh)}\n")
    print(f"Metric                  | Original       | Enhanced       | Delta")
    print(f"------------------------+----------------+----------------+---------")
    print(f"Play Duration (sec)     | {org['duration']:<14.3f} | {enh['duration']:<14.3f} | {enh['duration'] - org['duration']:+.3f}")
    print(f"Peak Amplitude (dBFS)   | {org['peak_db']:<14.1f} | {enh['peak_db']:<14.1f} | {enh['peak_db'] - org['peak_db']:+.1f}")
    print(f"Average Volume (RMS dB) | {org['rms_db']:<14.1f} | {enh['rms_db']:<14.1f} | {enh['rms_db'] - org['rms_db']:+.1f}")
    print(f"Spectral Center (Hz)    | {org['centroid']:<14.1f} | {enh['centroid']:<14.1f} | {enh['centroid'] - org['centroid']:+.1f}")
    print(f"Sub-80Hz Rumble Ratio   | {org['sub_bass_ratio']:<14.2f}% | {enh['sub_bass_ratio']:<14.2f}% | {enh['sub_bass_ratio'] - org['sub_bass_ratio']:+.2f}%")
    print("\n==========================================================")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python audio_diff.py <original.wav> <enhanced.wav>")
        sys.argv = ["audio_diff.py", "../colab/wavs/0.wav", "../colab/wavs.enh/0.wav"] # fallback path
    print_diff(sys.argv[1], sys.argv[2])