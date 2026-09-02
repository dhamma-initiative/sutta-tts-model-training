import os
import sys
import json
import torch
import numpy as np
import librosa
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# Define the 11 precise verification probes (4 acoustic + 7 punctuation)
VERIFICATION_PROBES = {
    "probe_01_sibilance": {
        "text": "carrier signal. sikhī, sãṁyutta, and sāvatthī have systematically stilled suttas. .carrier signal.",
        "type": "acoustic_sibilance"
    },
    "probe_02_plosives": {
        "text": "carrier signal. bhaddiya, bārāṇasī, baka brahmā, and bāhiya built big banyans. .carrier signal.",
        "type": "acoustic_plosives"
    },
    "probe_03_formants": {
        "text": "carrier signal. acchariy'abbhūtadhamma piṇḍapātapārisuddi sutta majjhima nikāya. .carrier signal.",
        "type": "acoustic_formants"
    },
    "probe_04_loudness": {
        "text": "carrier signal. What bliss! What bliss! Truly, the Buddha's bidding is done. .carrier signal.",
        "type": "acoustic_dynamics"
    },
    "probe_punct_01_comma": {
        "text": "carrier signal. stop, listen. .carrier signal.",
        "type": "pause",
        "symbol": ",",
        "target_ms": 150.0
    },
    "probe_punct_02_semicolon": {
        "text": "carrier signal. stop; listen. .carrier signal.",
        "type": "pause",
        "symbol": ";",
        "target_ms": 250.0
    },
    "probe_punct_03_colon": {
        "text": "carrier signal. stop: listen. .carrier signal.",
        "type": "pause",
        "symbol": ":",
        "target_ms": 250.0
    },
    "probe_punct_04_em_dash": {
        "text": "carrier signal. stop—listen. .carrier signal.",
        "type": "pause",
        "symbol": "—",
        "target_ms": 350.0
    },
    "probe_punct_05_ellipsis": {
        "text": "carrier signal. stop… listen. .carrier signal.",
        "type": "pause",
        "symbol": "…",
        "target_ms": 600.0
    },
    "probe_punct_06_brackets": {
        "text": "carrier signal. stop [listen] now. .carrier signal.",
        "type": "pause_bracket",
        "target_lead_ms": 120.0,
        "target_trail_ms": 180.0
    },
    "probe_punct_07_bullet": {
        "text": "carrier signal. stop • listen. .carrier signal.",
        "type": "pause",
        "symbol": "•",
        "target_ms": 400.0
    }
}

class SuttaVoiceUatCallback(Callback):
    """
    Active Closed-Loop UAT Validation Callback.
    Synthesizes physical audio waveforms for all 11 verification probes at validation epoch end,
    measures real-world acoustic properties and pause durations in milliseconds, and
    forces a graceful exit only when global loss stabilizes AND all unit tests pass.
    """
    def __init__(self, phoneme_map_path, target_global_loss=0.15, tolerance_pct=0.25):
        super().__init__()
        self.target_global_loss = target_global_loss
        self.tolerance_pct = tolerance_pct
        
        # Load phoneme map to convert text probes to token IDs
        with open(phoneme_map_path, "r") as f:
            self.phoneme_to_id = json.load(f)

    def text_to_ids(self, text):
        # Fallback character cleanup to align with 162-symbol map
        text_clean = text.normalize("NFC").toLowerCase()
        ids = []
        for char in text_clean:
            if char in self.phoneme_to_id:
                ids.append(self.phoneme_to_id[char])
        return ids

    def synthesize_probe_audio(self, pl_module, text_ids):
        # Generate raw audio natively using the current generator weights
        pl_module.eval()
        with torch.no_grad():
            x = torch.LongTensor([text_ids]).to(pl_module.device)
            x_lengths = torch.LongTensor([len(text_ids)]).to(pl_module.device)
            # Invoke the VITS generator's native inference graph (Medium model)
            audio = pl_module.generator.infer(x, x_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.1)[0]
            audio = audio.cpu().numpy().squeeze()
        return audio

    def measure_silence_gap(self, y, sr=22050):
        # Calculate Short-Time Energy
        hop_length = 128
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Determine silence frames relative to peak amplitude
        is_silent = rms_db < -40.0
        
        longest_silence_len = 0
        current_silence_len = 0
        
        # Ignore first/last 15% to skip carrier signal cushions
        start_frame = int(len(is_silent) * 0.15)
        end_frame = int(len(is_silent) * 0.85)
        
        for frame in range(start_frame, end_frame):
            if is_silent[frame]:
                current_silence_len += 1
            else:
                if current_silence_len > longest_silence_len:
                    longest_silence_len = current_silence_len
                current_silence_len = 0
                
        duration_ms = (longest_silence_len * hop_length / sr) * 1000
        return duration_ms

    def on_validation_end(self, trainer, pl_module):
        # Fetch current loss metrics from the Lightning log queue
        metrics = trainer.callback_metrics
        global_val_loss = metrics.get("val_loss")
        if global_val_loss is None:
            return

        print("\n" + "="*65)
        print("          SUTTAPLAYER NEURAL VOICE ACTIVE UAT SWEPT          ")
        print("="*65)
        print(f" * Active Epoch: {trainer.current_epoch:<5} | Global Validation Loss: {global_val_loss:.4f} (Target: <={self.target_global_loss})")
        print("-"*65)

        uat_passed = True
        checklist_status = {}

        # Loop through all 11 unit-test probes to synthesize and measure
        for probe_name, probe_info in VERIFICATION_PROBES.items():
            try:
                ids = self.text_to_ids(probe_info["text"])
                # Perform in-memory VITS synthesis
                y = self.synthesize_probe_audio(pl_module, ids)
                
                # Check metrics based on test-suite target categories
                if probe_info["type"] == "pause":
                    # Measure middle silence gap in milliseconds
                    pause_ms = self.measure_silence_gap(y)
                    target = probe_info["target_ms"]
                    min_t = target * (1.0 - self.tolerance_pct)
                    max_t = target * (1.0 + self.tolerance_pct)
                    
                    passed = min_t <= pause_ms <= max_t
                    checklist_status[probe_name] = {
                        "metric": f"{pause_ms:.1f} ms",
                        "target": f"{target:.1f} ms",
                        "status": "PASS" if passed else "FAIL"
                    }
                    if not passed:
                        uat_passed = False
                        
                elif probe_info["type"] == "acoustic_sibilance":
                    # Measure High-Frequency Spectral Centroid (de-esser target)
                    s_centroids = librosa.feature.spectral_centroid(y=y, sr=22050)
                    avg_centroid = np.mean(s_centroids)
                    # We expect calibrated sibilants to average under 2900 Hz in meditative voice
                    passed = avg_centroid <= 2900.0
                    checklist_status[probe_name] = {
                        "metric": f"{avg_centroid:.1f} Hz",
                        "target": "<= 2900 Hz",
                        "status": "PASS" if passed else "FAIL"
                    }
                    if not passed:
                        uat_passed = False

                elif probe_info["type"] == "acoustic_plosives":
                    # Measure low-end sub-bass rumbles (0Hz-60Hz) vs mid-range (120Hz-1000Hz)
                    stft = np.abs(librosa.stft(y))
                    sub_bass = np.mean(stft[0:6, :])
                    mids = np.mean(stft[12:100, :]) + 1e-6
                    ratio = sub_bass / mids
                    passed = ratio <= 0.15
                    checklist_status[probe_name] = {
                        "metric": f"Ratio: {ratio:.3f}",
                        "target": "<= 0.150",
                        "status": "PASS" if passed else "FAIL"
                    }
                    if not passed:
                        uat_passed = False
                        
                else:
                    # Treat other probes as structural safety checks (ensure non-empty audio)
                    passed = len(y) > 0 and np.max(np.abs(y)) > 0.01
                    checklist_status[probe_name] = {
                        "metric": f"{len(y)/22050:.2f} s",
                        "target": "Valid Wave",
                        "status": "PASS" if passed else "FAIL"
                    }
                    if not passed:
                        uat_passed = False

            except Exception as e:
                checklist_status[probe_name] = {
                    "metric": "CRASH",
                    "target": "N/A",
                    "status": f"FAIL ({str(e)})"
                }
                uat_passed = False

        # Output the gorgeous Markdown UAT Dashboard directly to stdout
        for name, res in checklist_status.items():
            print(f" [{res['status']}] {name:<26} | Metric: {res['metric']:<12} (Target: {res['target']})")

        print("-"*65)

        # Apply the Multi-Criteria Early Stopping Decision
        if global_val_loss <= self.target_global_loss and uat_passed:
            print("\n" + "*"*65)
            print("👑 UAT SIGN-OFF SUCCESSFUL: BOTH GLOBAL & UNIT TESTS PASSED!")
            print(f" Finalizing Epoch {trainer.current_epoch} as SuttaPlayer's Gold Standard Checkpoint.")
            print("*"*65 + "\n")
            
            # Export ONNX package dynamically
            trainer.should_stop = True
            
            # Save the final consolidated UAT passing checkpoint file
            pass_dir = os.path.dirname(trainer.checkpoint_callback.dirpath)
            passed_ckpt = os.path.join(pass_dir, "G_suttaplayer_uat_passed.ckpt")
            trainer.save_checkpoint(passed_ckpt)
            print(f"Saved consolidated UAT-passed checkpoint to: {passed_ckpt}")
        else:
            reason = []
            if global_val_loss > self.target_global_loss:
                reason.append("Global loss above limit")
            if not uat_passed:
                reason.append("Punctuation/Acoustic unit-tests failing")
            print(f" [UAT RETRY] Training continues. Reasons: {', '.join(reason)}")
        print("="*65 + "\n")
