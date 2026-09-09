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

VERIFICATION_PROBES = {
    "probe_01_sibilance": {
        "text": "sikhī, saṁyutta, and sāvatthī have sympathetically stilled suttas.",
        "phonemes": "sˈikʰiː, sˈɐmjuttə, ˈand sˈaːwəttʰiː hˈav sˌɪmpəθˈɛtɪkli stˈɪld sˈuttəs.",
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
        """Dumps all bound hyperparameters and model configuration parameters to stdout."""
        print("\n" + "="*70)
        print("          PYTORCH LIGHTNING MODEL HYPERPARAMETERS DUMP          ")
        print("="*70)
        
        # Dump hparams registered inside LightningModule
        if hasattr(pl_module, "hparams"):
            print("[hparams]:")
            for k, v in dict(pl_module.hparams).items():
                print(f"  {k}: {v}")
                
        # Check specific loss parameters on VITS model
        print("\n[VITS Loss Coefficients]:")
        for attr in ["c_mel", "c_kl", "c_dur", "loss_dur_scale"]:
            if hasattr(pl_module, attr):
                print(f"  pl_module.{attr} = {getattr(pl_module, attr)}")
            elif hasattr(pl_module, "model_g") and hasattr(pl_module.model_g, attr):
                print(f"  pl_module.model_g.{attr} = {getattr(pl_module.model_g, attr)}")
                
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
            # FIX: Evaluate at standard baseline length_scale = 1.0 (native learned timing)
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

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        global_val_loss = metrics.get("val_loss", 0.0)

        print("\n" + "="*65)
        print("          SUTTAPLAYER NEURAL VOICE ACTIVE UAT SWEPT          ")
        print("="*65)
        print(f" * Active Epoch: {trainer.current_epoch:<5} | Global Val Loss: {global_val_loss:.4f}")
        print("-"*65)

        for name, p in VERIFICATION_PROBES.items():
            text_ids = self.phonemes_to_ids(p["phonemes"])
            try:
                y = self.synthesize_probe_audio(pl_module, text_ids)
                if p["type"] == "pause":
                    pause_ms = self.measure_silence_gap(y)
                    target = p["target_ms"]
                    print(f"  [{p['symbol']}] {name:<25} | Measured Pause: {pause_ms:>5.1f} ms (Target: {target} ms)")
            except Exception as e:
                print(f"  [ERROR] Probe synthesis failed for {name}: {e}")

        print("="*65 + "\n")
        # NOTE: should_stop is intentionally NOT set here, allowing duration predictor to train through target epochs.