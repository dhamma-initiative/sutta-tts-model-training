# corpus-quality-diff.py
import os
import sys
import argparse
import librosa
import numpy as np

# Define our highly-targeted diagnostic test cases
TEST_CASES = {
    "2.wav": {
        "category": "Pali Retroflex & Plosive Transient Test",
        "text": "pūraṇa kassapa, makkhali gosāla, ajita kesakambalin, pakudha kaccāyana, sañjaya velaṭṭhaputta...",
        "focus": "Assesses low-frequency bilabial stop plosives and complex Pali consonant clusters."
    },
    "48.wav": {
        "category": "High-Energy Dynamic Contrast Test",
        "text": "...would repeatedly exclaim, 'what bliss! what bliss!'",
        "focus": "Evaluates soft-knee compression performance and brickwall limiter transient pumping."
    },
    "16.wav": {
        "category": "Rhythmic Pause & Silence Spacing Test",
        "text": "where is mahā moggallāna? *.* where is mahā kassapa? *.* where is mahā kaccāna?...",
        "focus": "Checks punctuation spacing consistency and trailing silent boundaries."
    },
    "600.wav": {
        "category": "Ultra-Short Sequence Temporal Compression Test",
        "text": "yes.",
        "focus": "Verifies that short inputs retain formants and do not evaporate during aggressive gating."
    }
}

def analyze_audio(filepath):
    try:
        # Load audio (sr=None preserves native sampling rate)
        y, sr = librosa.load(filepath, sr=None)
        duration = len(y) / sr
        
        # Calculate Peak and RMS Loudness
        peak = np.max(np.abs(y))
        peak_db = 20 * np.log10(peak + 1e-6)
        rms = np.sqrt(np.mean(y**2))
        rms_db = 20 * np.log10(rms + 1e-6)
        
        # Spectral Centroid (measures high-frequency presence/brightness)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Calculate Sub-80Hz Energy Ratio (low-end rumble check)
        stft = np.abs(librosa.stft(y))
        frequencies = librosa.fft_frequencies(sr=sr)
        sub_80hz_indices = np.where(frequencies <= 80)[0]
        sub_bass_energy = np.sum(stft[sub_80hz_indices, :])
        total_energy = np.sum(stft)
        sub_bass_ratio = (sub_bass_energy / (total_energy + 1e-6)) * 100
        
        return {
            "status": "OK",
            "duration": duration,
            "peak_db": peak_db,
            "rms_db": rms_db,
            "centroid": centroid,
            "sub_bass_ratio": sub_bass_ratio
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e)
        }

def run_diff(orig_dir, enh_dir):
    print("=========================================================================")
    print("                SUTTAPLAYER ACOUSTIC QUALITY DIFF REPORT                 ")
    print("=========================================================================")
    print(f"Original Folder: {orig_dir}")
    print(f"Enhanced Folder: {enh_dir}")
    print("=========================================================================\n")

    for filename, meta in TEST_CASES.items():
        orig_path = os.path.join(orig_dir, filename)
        enh_path = os.path.join(enh_dir, filename)
        
        print(f"STRESS TEST: {filename} ({meta['category']})")
        print(f"Utterance:   \"{meta['text']}\"")
        print(f"Acoustic Focus: {meta['focus']}\n")
        
        if not os.path.exists(orig_path):
            print(f"  ❌ Missing Original File: {orig_path}")
            print("-" * 73)
            continue
        if not os.path.exists(enh_path):
            print(f"  ❌ Missing Enhanced File: {enh_path}")
            print("-" * 73)
            continue
            
        orig = analyze_audio(orig_path)
        enh = analyze_audio(enh_path)
        
        if orig["status"] == "ERROR" or enh["status"] == "ERROR":
            print(f"  ❌ Processing Error: Orig={orig.get('error')}, Enh={enh.get('error')}")
            print("-" * 73)
            continue
            
        print(f"  Metric                  | Original       | Enhanced       | Delta")
        print(f"  ------------------------+----------------+----------------+---------")
        print(f"  Play Duration (sec)     | {orig['duration']:<14.3f} | {enh['duration']:<14.3f} | {enh['duration'] - orig['duration']:+.3f}")
        print(f"  Peak Amplitude (dBFS)   | {orig['peak_db']:<14.1f} | {enh['peak_db']:<14.1f} | {enh['peak_db'] - orig['peak_db']:+.1f}")
        print(f"  Average Volume (RMS dB) | {orig['rms_db']:<14.1f} | {enh['rms_db']:<14.1f} | {enh['rms_db'] - orig['rms_db']:+.1f}")
        print(f"  Spectral Center (Hz)    | {orig['centroid']:<14.1f} | {enh['centroid']:<14.1f} | {enh['centroid'] - orig['centroid']:+.1f}")
        print(f"  Sub-80Hz Rumble Ratio   | {orig['sub_bass_ratio']:<14.2f}% | {enh['sub_bass_ratio']:<14.2f}% | {enh['sub_bass_ratio'] - orig['sub_bass_ratio']:+.2f}%")
        print("\n" + "-" * 73 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Programmatic Acoustic Quality Diff Tool")
    parser.add_argument("-i", "--input", default="./tts.train/corpus/colab/wavs", help="Original WAV files directory")
    parser.add_argument("-o", "--output", default="./tts.train/corpus/colab/wavs.enh", help="Enhanced WAV files directory")
    args = parser.parse_args()
    
    run_diff(args.input, args.output)
