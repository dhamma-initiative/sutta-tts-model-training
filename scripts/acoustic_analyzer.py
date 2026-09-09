import json
import os
import librosa
import numpy as np

def analyze_probes():
    config_params = {}
    
    # Baseline directory checks
    probe_dir = "."
    files = {
        "sibilance": os.path.join(probe_dir, "probe_01_sibilance.wav"),
        "plosives": os.path.join(probe_dir, "probe_02_plosives.wav"),
        "formants": os.path.join(probe_dir, "probe_03_formants.wav"),
        "loudness": os.path.join(probe_dir, "probe_04_loudness.wav")
    }
    
    # -------------------------------------------------------------
    # 1. ANALYZE PROBE 01: Sibilance & Fricative Energy
    # -------------------------------------------------------------
    y, sr = librosa.load(files["sibilance"], sr=22050)
    s_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    avg_centroid = np.mean(s_centroids)
    
    # Calibrate de-esser index
    if avg_centroid > 3200:
        config_params["deesser_intensity"] = 0.12
    elif avg_centroid > 2800:
        config_params["deesser_intensity"] = 0.08
    else:
        config_params["deesser_intensity"] = 0.04

    # -------------------------------------------------------------
    # 2. ANALYZE PROBE 02: Low-Frequency Plosive Surge
    # -------------------------------------------------------------
    y_p, _ = librosa.load(files["plosives"], sr=22050)
    stft = np.abs(librosa.stft(y_p))
    # Low-end sub-bass bins (0 Hz to 60 Hz) vs Mid-range bins (150 Hz to 1000 Hz)
    sub_bass_energy = np.mean(stft[0:6, :])
    mid_range_energy = np.mean(stft[15:100, :])
    
    ratio = sub_bass_energy / (mid_range_energy + 1e-6)
    if ratio > 0.18:
        config_params["highpass_cutoff"] = 80  # Aggressive cut for heavy popping
    elif ratio > 0.10:
        config_params["highpass_cutoff"] = 65  # Moderate cleanup
    else:
        config_params["highpass_cutoff"] = 50  # Gentle roll-off

    # -------------------------------------------------------------
    # 3. ANALYZE PROBE 03: Vowel Resonance & Formant Tilt
    # -------------------------------------------------------------
    # VITS models often exhibit "muffled low-mids" (300Hz-600Hz) or 
    # "presence decay" in the clarity band (2.5kHz-4kHz) during dense Pali runs.
    y_f, _ = librosa.load(files["formants"], sr=22050)
    stft_f = np.abs(librosa.stft(y_f))
    
    # Low-mid mud band (approx 300Hz - 600Hz)
    mud_energy = np.mean(stft_f[30:60, :])
    # Presence clarity band (approx 2500Hz - 4000Hz)
    presence_energy = np.mean(stft_f[250:400, :])
    
    # Mud notch EQ target
    if mud_energy > (presence_energy * 1.5):
        config_params["eq_mid_low_notch_db"] = -2.5  # Pull down the mud
    else:
        config_params["eq_mid_low_notch_db"] = -1.0  # Smooth correction
        
    # Presence boost EQ target
    tilt_ratio = presence_energy / (mud_energy + 1e-6)
    if tilt_ratio < 0.4:
        config_params["eq_presence_boost_db"] = 2.0  # Muffled; inject high-mid presence
    elif tilt_ratio < 0.6:
        config_params["eq_presence_boost_db"] = 1.0  # Slight correction
    else:
        config_params["eq_presence_boost_db"] = 0.0  # Balanced spectrum

    # -------------------------------------------------------------
    # 4. ANALYZE PROBE 04: Dynamic Range & Peak-to-RMS (Crest Factor)
    # -------------------------------------------------------------
    # Evaluate high-energy exclamations ("What bliss!") vs standard decay
    y_l, _ = librosa.load(files["loudness"], sr=22050)
    rms = librosa.feature.rms(y=y_l)
    peak = np.max(np.abs(y_l))
    mean_rms = np.mean(rms)
    
    # Crest Factor: Peak amplitude divided by RMS energy
    crest_factor = peak / (mean_rms + 1e-6)
    
    # If Crest Factor is high (> 6.0), the VITS exclamations are too spikey.
    # We calibrate a soft-knee compressor to prevent loudnorm from crushing transients.
    if crest_factor > 6.0:
        config_params["compand_clipping_preventer"] = "compand=0.005:0.1:-50/-50|-20/-20|0/-4"
    else:
        config_params["compand_clipping_preventer"] = "compand=0.01:0.1:-50/-50|-15/-15|0/-1"
                
    # Measure integrated loudness (approximation to check raw level)
    config_params["loudnorm_measured_i"] = float(round(-20.0 - (10.0 * np.log10(mean_rms + 1e-6)), 1))

    # Re-save configuration with all 6 parameters populated
    with open("optimal_dsp_config.json", "w") as f:
        json.dump(config_params, f, indent=2)
    print("Acoustic analysis complete. Balanced 6-Parameter Config saved!")

if __name__ == "__main__":
    analyze_probes()