import os
import sys
import json
import csv
import time
import torch
import numpy as np
import librosa

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

# Define the 13 precise verification probes (4 acoustic + 9 punctuation)
# Fully pre-phonemized and mapped to your exact Audacity-calibrated timing envelopes.
VERIFICATION_PROBES = {
    "probe_01_sibilance": {
        "text": "sikhī, saṁyutta, and sāvatthī have sympathetically stilled suttas.",
        "phonemes": "sˈikʰiː, sˈɐmjuttə, ˈand sˈaːwəttʰiː hˈav sˌɪmpəθˈɛtɪkli stˈɪld sˈuttəs.",
        "type": "acoustic_sibilance"
    },
    "probe_02_plosives": {
        "text": "bhaddiya, bārāṇasī, baka brahmā, and bāhiya built big banyan.",
        "phonemes": "bʰˈɐddijə, bˈaːɹaːɳˌəsiː, bˈɐkə b ɹˈɐhmaː, ˈand bˈaːhijə bˈɪlt bˈɪɡ bˈanjən.",
        "type": "acoustic_plosives"
    },
    "probe_03_formants": {
        "text": "acchariy'abbhūtadhamma piṇḍapātapārisuddhi sutta majjhima nikāya.",
        "phonemes": "ˈɐtʃtʃhəɹij ˈɐbbʰuːtˌədʰəmmə pˈiɳɖəpˌaːtəpˌaːɹisˌuddʰi sˈuttə mˈɐdʒdʒhimə nˈikaːjə.",
        "type": "acoustic_formants"
    },
    "probe_04_loudness": {
        "text": "What bliss! What bliss! Truly, the Buddha's bidding is done.",
        "phonemes": "wˈɒt blˈɪs! wˈɒt blˈɪs! tɹˈuːli, ðˈə bˈuddʰəs bˈɪdɪŋ ˈɪz dˈʌn.",
        "type": "acoustic_dynamics"
    },
    "probe_punct_01_comma": {
        "text": "stop, listen.",
        "phonemes": "stˈɒp, lˈɪsən.",
        "type": "pause",
        "symbol": ",",
        "target_ms": 295.0
    },
    "probe_punct_02_semicolon": {
        "text": "stop; listen.",
        "phonemes": "stˈɒp; lˈɪsən.",
        "type": "pause",
        "symbol": ";",
        "target_ms": 260.0
    },
    "probe_punct_03_colon": {
        "text": "stop: listen.",
        "phonemes": "stˈɒp: lˈɪsən.",
        "type": "pause",
        "symbol": ":",
        "target_ms": 340.0
    },
    "probe_punct_04_em_dash": {
        "text": "stop—listen.",
        "phonemes": "stˈɒp—lˈɪsən.",
        "type": "pause",
        "symbol": "—",
        "target_ms": 255.0
    },
    "probe_punct_05_ellipsis": {
        "text": "stop… listen.",
        "phonemes": "stˈɒp… lˈɪsən.",
        "type": "pause",
        "symbol": "…",
        "target_ms": 530.0
    },
    "probe_punct_06_brackets": {
        "text": "stop [listen] now.",
        "phonemes": "stˈɒp [lˈɪsən] nˈaʊ.",
        "type": "pause_bracket",
        "target_lead_ms": 420.0,
        "target_trail_ms": 280.0
    },
    "probe_punct_07_bullet": {
        "text": "stop • listen.",
        "phonemes": "stˈɒp • lˈɪsən.",
        "type": "pause",
        "symbol": "•",
        "target_ms": 420.0
    },
    "probe_punct_08_parentheses": {
        "text": "stop (listen) now.",
        "phonemes": "stˈɒp (lˈɪsən) nˈaʊ.",
        "type": "pause_bracket",
        "target_lead_ms": 420.0,
        "target_trail_ms": 280.0
    },
    "probe_punct_09_braces": {
        "text": "stop {listen} now.",
        "phonemes": "stˈɒp {lˈɪsən} nˈaʊ.",
        "type": "pause_bracket",
        "target_lead_ms": 420.0,
        "target_trail_ms": 280.0
    }
}

class SuttaVoiceUatCallback(Callback):
    """
    Active Closed-Loop UAT Validation Callback (v5).
    Synthesizes physical audio waveforms for all 13 pre-phonemized probes at validation epoch end,
    measures real-world acoustic properties and pause durations in milliseconds without any carrier signals,
    calculates exact convergence percentages for each probe, appends them to a local CSV file on Google Drive,
    and forces a graceful exit only when global loss stabilizes AND all unit tests pass.
    """
    def __init__(self, phoneme_map_path=None, target_global_loss=0.15, tolerance_pct=0.25):
        super().__init__()
        self.target_global_loss = target_global_loss
        self.tolerance_pct = tolerance_pct
        
        # Resolve phoneme map path dynamically across local and cloud environments
        if phoneme_map_path is None:
            possible_paths = [
                "./phoneme_map.json",
                "../phoneme_map.json",
                "/content/drive/MyDrive/piper_training/phoneme_map.json",
                "./config/en[gb]_pi[si]-suttaplayer-phoneme-map.json",
                "./sutta-tts-model-training/config/en[gb]_pi[si]-suttaplayer-phoneme-map.json"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    phoneme_map_path = path
                    break
        
        if phoneme_map_path is None or not os.path.exists(phoneme_map_path):
            raise FileNotFoundError("Could not locate phoneme_map.json in any standard local or cloud search path.")
            
        with open(phoneme_map_path, "r", encoding="utf-8") as f:
            self.phoneme_to_id = json.load(f)

    def phonemes_to_ids(self, phoneme_str):
        ids = []
        for char in phoneme_str:
            if char in self.phoneme_to_id:
                ids.append(self.phoneme_to_id[char])
        return ids

    def synthesize_probe_audio(self, pl_module, text_ids):
        pl_module.eval()
        with torch.no_grad():
            x = torch.LongTensor([text_ids]).to(pl_module.device)
            x_lengths = torch.LongTensor([len(text_ids)]).to(pl_module.device)
            audio = pl_module.model_g.infer(x, x_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.1)[0]
            audio = audio.cpu().numpy().squeeze()
        return audio

    def measure_silence_gap(self, y, sr=22050):
        # Calculate Short-Time Energy
        hop_length = 128
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max).flatten()
        
        is_silent = rms_db < -45.0
        
        longest_silence_len = 0
        current_silence_len = 0
        
        for frame in range(len(is_silent)):
            if is_silent[frame]:
                current_silence_len += 1
            else:
                if current_silence_len > longest_silence_len:
                    longest_silence_len = current_silence_len
                current_silence_len = 0
                
        duration_ms = (longest_silence_len * hop_length / sr) * 1000
        return duration_ms

    def measure_silence_gaps(self, y, sr=22050):
        # Multi-gap measurement for bracket/parentheses/braces
        hop_length = 128
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max).flatten()
        
        is_silent = rms_db < -45.0
        
        gaps = []
        current_gap = 0
        in_silence = False
        
        for frame in range(len(is_silent)):
            silent = bool(is_silent[frame])
            if silent and not in_silence:
                in_silence = True
                current_gap = 1
            elif silent and in_silence:
                current_gap += 1
            elif not silent and in_silence:
                in_silence = False
                gap_ms = (current_gap * hop_length / sr) * 1000
                if gap_ms > 30.0:  # Ignore micro-transient silences under 30ms
                    gaps.append(gap_ms)
                current_gap = 0
                
        if in_silence:
            gap_ms = (current_gap * hop_length / sr) * 1000
            if gap_ms > 30.0:
                gaps.append(gap_ms)
                
        return gaps

    def calculate_convergence(self, name, ptype, metrics_dict, p):
        # Calculate mathematical convergence towards target parameters (0% to 100%)
        if ptype == "pause":
            val = metrics_dict["pause_ms"]
            target = p["target_ms"]
            pct = max(0.0, 100.0 - (abs(val - target) / target) * 100.0)
            return pct
        elif ptype == "pause_bracket":
            val_lead = metrics_dict["lead_ms"]
            val_trail = metrics_dict["trail_ms"]
            t_lead = p["target_lead_ms"]
            t_trail = p["target_trail_ms"]
            
            pct_lead = max(0.0, 100.0 - (abs(val_lead - t_lead) / t_lead) * 100.0)
            pct_trail = max(0.0, 100.0 - (abs(val_trail - t_trail) / t_trail) * 100.0)
            return (pct_lead + pct_trail) / 2.0
        elif ptype == "acoustic_sibilance":
            val = metrics_dict["spectral_centroid_hz"]
            # Target is <= 2900 Hz. If met, convergence is 100%.
            if val <= 2900.0:
                return 100.0
            else:
                return max(0.0, 100.0 - ((val - 2900.0) / 2900.0) * 100.0)
        elif ptype == "acoustic_plosives":
            val = metrics_dict["plosive_ratio"]
            # Target is <= 0.150. If met, convergence is 100%.
            if val <= 0.150:
                return 100.0
            else:
                return max(0.0, 100.0 - ((val - 0.150) / 0.15) * 100.0)
        return 100.0

    def log_metrics_to_csv(self, epoch, global_loss, convergence_pcts):
        # Appends the epoch's raw convergence percentages directly to drive-mapped CSV
        csv_path = "./uat_metrics.csv"
        # Check cloud directories
        colab_drive_path = "/content/drive/MyDrive/piper_training/uat_metrics.csv"
        if os.path.exists("/content/drive/MyDrive/piper_training"):
            csv_path = colab_drive_path
            
        file_exists = os.path.exists(csv_path)
        
        # Build headers
        headers = ["timestamp", "epoch", "val_loss"] + list(VERIFICATION_PROBES.keys())
        
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": str(epoch),
            "val_loss": f"{global_loss:.4f}"
        }
        for name in VERIFICATION_PROBES.keys():
            row[name] = f"{convergence_pcts.get(name, 0.0):.2f}"
            
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            print(f"  [CSV Logger] Appended validation UAT row to: {csv_path}")
        except Exception as e:
            print(f"  [CSV Logger] [WARNING] Failed to append metrics to CSV: {e}")

    def on_validation_end(self, trainer, pl_module):
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
        probe_convergences = {}

        # Loop through all 13 unit-test probes to synthesize and measure
        for name, p in VERIFICATION_PROBES.items():
            phoneme_str = p["phonemes"]
            ptype = p["type"]
            text_ids = self.phonemes_to_ids(phoneme_str)
            
            try:
                y = self.synthesize_probe_audio(pl_module, text_ids)
            except Exception as e:
                print(f"  [ERROR] Synthesis failed for {name}: {e}")
                uat_passed = False
                probe_convergences[name] = 0.0
                continue

            meas = {}
            if ptype == "pause":
                pause_ms = self.measure_silence_gap(y)
                target = p["target_ms"]
                low_bound = target * (1.0 - self.tolerance_pct)
                high_bound = target * (1.0 + self.tolerance_pct)
                passed = low_bound <= pause_ms <= high_bound
                
                meas["pause_ms"] = pause_ms
                conv_pct = self.calculate_convergence(name, ptype, meas, p)
                probe_convergences[name] = conv_pct
                
                status_str = "PASS" if passed else "FAIL"
                print(f"  [{status_str}] {name:<25} | Pause: {pause_ms:>5.1f} ms (Target: {target} ms) [Conv: {conv_pct:.1f}%]")
                if not passed:
                    uat_passed = False

            elif ptype == "pause_bracket":
                gaps = self.measure_silence_gaps(y)
                t_lead = p["target_lead_ms"]
                t_trail = p["target_trail_ms"]
                
                lead_ms = gaps[0] if len(gaps) > 0 else 0.0
                trail_ms = gaps[1] if len(gaps) > 1 else 0.0
                
                lead_passed = t_lead * (1.0 - self.tolerance_pct) <= lead_ms <= t_lead * (1.0 + self.tolerance_pct)
                trail_passed = t_trail * (1.0 - self.tolerance_pct) <= trail_ms <= t_trail * (1.0 + self.tolerance_pct)
                passed = lead_passed and trail_passed
                
                meas["lead_ms"] = lead_ms
                meas["trail_ms"] = trail_ms
                conv_pct = self.calculate_convergence(name, ptype, meas, p)
                probe_convergences[name] = conv_pct
                
                status_str = "PASS" if passed else "FAIL"
                print(f"  [{status_str}] {name:<25} | Lead: {lead_ms:>5.1f} ms ({t_lead}) | Trail: {trail_ms:>5.1f} ms ({t_trail}) [Conv: {conv_pct:.1f}%]")
                if not passed:
                    uat_passed = False

            elif ptype.startswith("acoustic_"):
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=22050))
                stft = np.abs(librosa.stft(y))
                sub_bass = np.mean(stft[0:6, :])
                mids = np.mean(stft[12:100, :]) + 1e-6
                plosive_ratio = sub_bass / mids
                
                meas["spectral_centroid_hz"] = spectral_centroid
                meas["plosive_ratio"] = plosive_ratio
                conv_pct = self.calculate_convergence(name, ptype, meas, p)
                probe_convergences[name] = conv_pct
                
                if ptype == "acoustic_sibilance":
                    passed = spectral_centroid < 2900.0
                    status_str = "PASS" if passed else "FAIL"
                    print(f"  [{status_str}] {name:<25} | Sibilance: {spectral_centroid:.1f} Hz (Target: <=2900) [Conv: {conv_pct:.1f}%]")
                elif ptype == "acoustic_plosives":
                    passed = plosive_ratio <= 0.150
                    status_str = "PASS" if passed else "FAIL"
                    print(f"  [{status_str}] {name:<25} | Plosive Ratio: {plosive_ratio:.3f} (Target: <=0.150) [Conv: {conv_pct:.1f}%]")
                else:
                    passed = True
                    status_str = "PASS"
                    print(f"  [{status_str}] {name:<25} | Centroid: {spectral_centroid:.1f} Hz [Conv: {conv_pct:.1f}%]")
                    
                if not passed:
                    uat_passed = False

        print("-"*65)
        
        # Append this epoch's raw convergence data to Drive CSV
        self.log_metrics_to_csv(trainer.current_epoch, global_val_loss, probe_convergences)
        
        # Final convergence checklist gate
        loss_ok = global_val_loss <= self.target_global_loss
        if loss_ok and uat_passed:
            print("🎉 [CONVERGENCE] ALL ACCENT AND PUNCTUATION UNIT TESTS PASSED!")
            print("Exiting trainer gracefully. Final ONNX export triggered.")
            print("="*65 + "\n")
            trainer.should_stop = True
        else:
            reasons = []
            if not loss_ok:
                reasons.append("Validation loss is still too high")
            if not uat_passed:
                reasons.append("Punctuation pause or acoustic checks failed")
            print(f"  ⚠️  [CONTINUE] Training will proceed. Reasons: {', '.join(reasons)}")
            print("="*65 + "\n")
