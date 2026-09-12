import os
import sys
import json
import csv
import time
import torch
import numpy as np
import librosa
import scipy.io.wavfile as wavfile

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

from lightning.pytorch.callbacks import Callback, ModelCheckpoint

# =========================================================================
# SUTTAPLAYER UAT VERIFICATION PROBES
# =========================================================================
# Re-calibrated to align with the Piper/VITS configuration.
# Brackets [], parentheses (), and curly braces {} have been fully restored
# with exact target durations defined in generate-padded-dataset.ts
# =========================================================================
VERIFICATION_PROBES = {
    "probe_01_sibilance": {
        "text": "sikhī, saṁyutta, and sāvatthī have sympathetically stilled suttas.",
        "phonemes": "s̪ɪkʰiː, s̪əŋjut̪t̪ə, ˈand s̪ɑːʋət̪t̪ʰiː hˈav sˌɪmpəθˈɛtɪkli stˈɪld s̪ut̪t̪əs̪.",
        "type": "acoustic_sibilance"
    },
    "probe_punct_01_comma": {
        "text": "stop, listen.",
        "phonemes": "stˈɒp, lˈɪsən.",
        "type": "pause",
        "symbol": ",",
        "target_ms": 250.0
    },
    "probe_punct_02_semicolon": {
        "text": "stop; listen.",
        "phonemes": "stˈɒp; lˈɪsən.",
        "type": "pause",
        "symbol": ";",
        "target_ms": 350.0
    },
    "probe_punct_03_colon": {
        "text": "stop: listen.",
        "phonemes": "stˈɒp: lˈɪsən.",
        "type": "pause",
        "symbol": ":",
        "target_ms": 400.0
    },
    "probe_punct_04_em_dash": {
        "text": "stop—listen.",
        "phonemes": "stˈɒp—lˈɪsən.",
        "type": "pause",
        "symbol": "—",
        "target_ms": 450.0
    },
    "probe_punct_05_ellipsis": {
        "text": "stop… listen.",
        "phonemes": "stˈɒp… lˈɪsən.",
        "type": "pause",
        "symbol": "…",
        "target_ms": 600.0
    },
    "probe_punct_06_brackets": {
        "text": "stop [ listen ] now",
        "phonemes": "stˈɒp [ lˈɪsən ] nˈaʊ",
        "type": "pause",
        "symbol": "]",
        "target_ms": 420.0
    },
    "probe_punct_07_bullet": {
        "text": "stop • listen",
        "phonemes": "stˈɒp • lˈɪsən",
        "type": "pause",
        "symbol": "•",
        "target_ms": 420.0
    },
    "probe_punct_08_parentheses": {
        "text": "stop ( listen ) now",
        "phonemes": "stˈɒp ( lˈɪsən ) nˈaʊ",
        "type": "pause",
        "symbol": ")",
        "target_ms": 280.0
    },
    "probe_punct_09_braces": {
        "text": "stop { listen } now",
        "phonemes": "stˈɒp { lˈɪsən } nˈaʊ",
        "type": "pause",
        "symbol": "}",
        "target_ms": 420.0
    }
}

class SuttaVoiceUatCallback(Callback):
    def __init__(self, phoneme_map_path=None):
        super().__init__()
        if phoneme_map_path is None:
            possible_paths = [
                "./phoneme_map.json",
                "/content/drive/MyDrive/piper_training/phoneme_map.json",
                "/content/drive/MyDrive/sutta-tts-model-training/config/en[gb]_pi[si]-suttaplayer-phoneme-map.json"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    phoneme_map_path = path
                    break
            
        if phoneme_map_path is None or not os.path.exists(phoneme_map_path):
            raise FileNotFoundError("Could not locate phoneme_map.json.")
            
        with open(phoneme_map_path, "r", encoding="utf-8") as f:
            self.phoneme_to_id = json.load(f)

    def on_fit_start(self, trainer, pl_module):
        # 1. Automatically retarget ModelCheckpoint away from val_mos
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.monitor == "val_mos":
                cb.monitor = "val_mel"
                cb.mode = "min"
                print("\n💡 [SuttaVoiceUatCallback] Retargeted ModelCheckpoint monitor: 'val_mos' -> 'val_mel' (mode='min')\n")

        # 2. Existing hyperparameter dump...
        print("\n" + "="*70)
        print("          ACTIVE VITS TRAINING HYPERPARAMETERS          ")
        print("="*70)
        for attr in ["c_mel", "c_kl", "c_dur"]:
            val = getattr(pl_module, attr, None) or getattr(getattr(pl_module, "model_g", None), attr, "N/A")
            print(f"  {attr}: {val}")
        print("="*70 + "\n")

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
            # Evaluate at standard baseline length_scale = 1.0 (native learned timing)
            audio = pl_module.model_g.infer(x, x_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0]
            audio = audio.cpu().numpy().squeeze()
        return audio

    def measure_silence_gap(self, y, sr=22050):
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
                
        return (longest_silence_len * hop_length / sr) * 1000

    def check_graceful_stop(self, trainer):
        """Checks for stop signal file to terminate training cleanly."""
        possible_stop_paths = [
            "/content/drive/MyDrive/piper_training/stop.txt",
            "/content/drive/MyDrive/sutta-tts-model-training/stop.txt",
            "./stop.txt"
        ]
        for stop_path in possible_stop_paths:
            if os.path.exists(stop_path):
                print(f"\n🛑 Graceful Stop Signal File Detected at: {stop_path}")
                print("Setting trainer.should_stop = True to safely exit on current step/epoch boundary...")
                trainer.should_stop = True
                try:
                    os.remove(stop_path)
                    print("Removed stop signal file successfully.")
                except Exception as e:
                    print(f"Warning: Could not remove stop signal file: {e}")
                break

    def on_train_epoch_end(self, trainer, pl_module):
        # Check stop signal at the end of each training epoch
        self.check_graceful_stop(trainer)

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        global_val_loss = metrics.get("val_loss", 0.0)

        print("\n" + "="*65)
        print("          SUTTAPLAYER NEURAL VOICE ACTIVE UAT SWEPT          ")
        print("="*65)
        print(f" * Active Epoch: {trainer.current_epoch:<5} | Global Val Loss: {global_val_loss:.4f}")
        print("-"*65)

        # Enforce target directory for colab preview playback
        preview_dir = "/content/preview"
        os.makedirs(preview_dir, exist_ok=True)

        for name, p in VERIFICATION_PROBES.items():
            text_ids = self.phonemes_to_ids(p["phonemes"])
            try:
                y = self.synthesize_probe_audio(pl_module, text_ids)
                
                # Write preview WAV file programmatically
                wav_path = os.path.join(preview_dir, f"{name}.wav")
                audio_int16 = (y * 32767.0).clip(-32768, 32767).astype(np.int16)
                wavfile.write(wav_path, 22050, audio_int16)
                
                if p["type"] == "pause":
                    pause_ms = self.measure_silence_gap(y)
                    target = p["target_ms"]
                    delta = pause_ms - target
                    print(f"  [{p['symbol']}] {name:<25} | Measured Pause: {pause_ms:>5.1f} ms (Target: {target} ms | Delta: {delta:+.1f} ms) [🔊 Preview Ready]")
                else:
                    print(f"  [🔊] {name:<25} | Acoustic sibilance test synthesized and exported cleanly.")
            except Exception as e:
                print(f"  [ERROR] Probe synthesis failed for {name}: {e}")

        print("="*65 + "\n")
        
        # Check stop signal at the end of validation loop
        self.check_graceful_stop(trainer)
