import os
import sys
import json
import csv
import time
import unicodedata
import torch
import numpy as np

try:
    import librosa
    import scipy.io.wavfile as wavfile
except ImportError:
    pass

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
except ImportError:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    except ImportError:
        Callback = object
        ModelCheckpoint = object

# =========================================================================
# SUTTAPLAYER UAT VERIFICATION PROBES (EXACT REFERENCE PHONEMES & TEXT)
# =========================================================================
# Updated to match the exact Aksharamukha/Deno compiled phoneme outputs in
# corpus-preperation/uat-metadata-phonemes.csv and uat-metadata-text.csv.
# =========================================================================
DEFAULT_VERIFICATION_PROBES = {
    "probe_01_sibilance": {
        "text": "sikhī, saṁyutta, and sāvatthī have sympathetically stilled suttas.",
        "phonemes": " s̪ɪkʰiː, s̪əŋjut̪t̪ə, ˈand s̪ɑːʋət̪t̪ʰiː hˈav sˌɪmpəθˈɛtɪkli stˈɪld s̪ut̪t̪əs̪.",
        "type": "acoustic_sibilance"
    },
    "probe_punct_01_comma": {
        "text": "stop, listen.",
        "phonemes": " stˈɒp, lˈɪsən.",
        "type": "pause",
        "symbol": ",",
        "target_ms": 250.0
    },
    "probe_punct_02_semicolon": {
        "text": "stop; listen.",
        "phonemes": " stˈɒp; lˈɪsən.",
        "type": "pause",
        "symbol": ";",
        "target_ms": 350.0
    },
    "probe_punct_03_colon": {
        "text": "stop: listen.",
        "phonemes": " stˈɒp: lˈɪsən.",
        "type": "pause",
        "symbol": ":",
        "target_ms": 400.0
    },
    "probe_punct_04_em_dash": {
        "text": "stop—listen.",
        "phonemes": " stˈɒp—lˈɪsən.",
        "type": "pause",
        "symbol": "—",
        "target_ms": 450.0
    },
    "probe_punct_05_ellipsis": {
        "text": "stop… listen.",
        "phonemes": " stˈɒp… lˈɪsən.",
        "type": "pause",
        "symbol": "…",
        "target_ms": 600.0
    },
    "probe_punct_06_brackets": {
        "text": "start [ listen ] stop",
        "phonemes": " stˈɑːt [ lˈɪsən ] stˈɒp",
        "type": "pause",
        "symbol": "]",
        "target_ms": 420.0
    },
    "probe_punct_07_bullet": {
        "text": "stop • listen",
        "phonemes": " stˈɒp • lˈɪsən",
        "type": "pause",
        "symbol": "•",
        "target_ms": 420.0
    },
    "probe_punct_08_parentheses": {
        "text": "start ( listen ) stop",
        "phonemes": " stˈɑːt ( lˈɪsən ) stˈɒp",
        "type": "pause",
        "symbol": ")",
        "target_ms": 280.0
    },
    "probe_punct_09_braces": {
        "text": "start { listen } stop",
        "phonemes": " stˈɑːt { lˈɪsən } stˈɒp",
        "type": "pause",
        "symbol": "}",
        "target_ms": 420.0
    }
}

def load_uat_probes_from_csv():
    """Loads UAT probe text & phonemes dynamically from uat-metadata CSVs if available."""
    probes = json.loads(json.dumps(DEFAULT_VERIFICATION_PROBES))
    
    candidate_pho_paths = [
        "/content/drive/MyDrive/sutta-tts-model-training/corpus-preperation/uat-metadata-phonemes.csv",
        "./corpus-preperation/uat-metadata-phonemes.csv",
        "../corpus-preperation/uat-metadata-phonemes.csv",
        "./uat-metadata-phonemes.csv"
    ]
    candidate_txt_paths = [
        "/content/drive/MyDrive/sutta-tts-model-training/corpus-preperation/uat-metadata-text.csv",
        "./corpus-preperation/uat-metadata-text.csv",
        "../corpus-preperation/uat-metadata-text.csv",
        "./uat-metadata-text.csv"
    ]
    
    pho_file = next((p for p in candidate_pho_paths if os.path.exists(p)), None)
    txt_file = next((p for p in candidate_txt_paths if os.path.exists(p)), None)
    
    if pho_file:
        try:
            with open(pho_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str or "|" not in line_str:
                        continue
                    parts = line_str.split("|", 1)
                    wav_name = parts[0].strip().replace(".wav", "")
                    pho_str = parts[1]
                    if wav_name in probes:
                        probes[wav_name]["phonemes"] = pho_str
            print(f"  ✅ Dynamic UAT Probe Phonemes loaded from: {pho_file}")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not read {pho_file}: {e}")

    if txt_file:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str or "|" not in line_str:
                        continue
                    parts = line_str.split("|", 1)
                    wav_name = parts[0].strip().replace(".wav", "")
                    txt_str = parts[1].strip()
                    if wav_name in probes:
                        probes[wav_name]["text"] = txt_str
            print(f"  ✅ Dynamic UAT Probe Text loaded from: {txt_file}")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not read {txt_file}: {e}")

    return probes


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
            
        if phoneme_map_path and os.path.exists(phoneme_map_path):
            with open(phoneme_map_path, "r", encoding="utf-8") as f:
                self.phoneme_to_id = json.load(f)
        else:
            self.phoneme_to_id = {}

        # Longest-prefix Trie / key sorting for multi-character token matching
        self.sorted_keys = sorted([k for k in self.phoneme_to_id.keys() if k], key=len, reverse=True)
        
        def _get_id(key, default):
            val = self.phoneme_to_id.get(key, default)
            return val[0] if isinstance(val, list) else val

        self.pad_id = _get_id("_", 0)
        self.bos_id = _get_id("^", 1)
        self.eos_id = _get_id("$", 2)
        
        # Ingest dynamic probes from repo CSV if present, or fall back to defaults
        self.probes = load_uat_probes_from_csv()

    def on_fit_start(self, trainer, pl_module):
        """Retargets ModelCheckpoint away from val_mos and dumps model parameters."""
        if hasattr(trainer, "callbacks"):
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint) and getattr(cb, "monitor", None) == "val_mos":
                    cb.monitor = "val_mel"
                    cb.mode = "min"
                    print("\n💡 [SuttaVoiceUatCallback] Retargeted ModelCheckpoint monitor: 'val_mos' -> 'val_mel' (mode='min')\n")

        print("\n" + "="*70)
        print("          PYTORCH LIGHTNING MODEL HYPERPARAMETERS DUMP          ")
        print("="*70)
        
        if hasattr(pl_module, "hparams"):
            print("[hparams]:")
            for k, v in dict(pl_module.hparams).items():
                print(f"  {k}: {v}")
                
        print("\n[VITS Loss Coefficients]:")
        for attr in ["c_mel", "c_kl", "c_dur", "loss_dur_scale"]:
            if hasattr(pl_module, attr):
                print(f"  pl_module.{attr} = {getattr(pl_module, attr)}")
            elif hasattr(pl_module, "model_g") and hasattr(pl_module.model_g, attr):
                print(f"  pl_module.model_g.{attr} = {getattr(pl_module.model_g, attr)}")
                
        print("="*70 + "\n")
        self.check_graceful_stop(trainer)

    def phonemes_to_ids(self, phoneme_str, intersperse=True):
        """Converts UTF-8 phoneme strings to token IDs with NFD normalization,
        BOS/EOS boundary tokens, and interspersed PAD tokens.
        """
        phoneme_str = unicodedata.normalize("NFD", phoneme_str)
        raw_ids = []
        i = 0
        while i < len(phoneme_str):
            matched = False
            for k in self.sorted_keys:
                if phoneme_str.startswith(k, i):
                    val = self.phoneme_to_id[k]
                    if isinstance(val, list):
                        raw_ids.extend(val)
                    else:
                        raw_ids.append(val)
                    i += len(k)
                    matched = True
                    break
            if not matched:
                i += 1  # Skip unknown character

        if not raw_ids:
            return [self.bos_id, self.eos_id]

        if intersperse:
            ids = [self.bos_id]
            for item_id in raw_ids:
                ids.extend([self.pad_id, item_id])
            ids.extend([self.pad_id, self.eos_id])
            return ids
        else:
            return [self.bos_id] + raw_ids + [self.eos_id]

    def synthesize_probe_audio(self, pl_module, text_ids):
        pl_module.eval()
        with torch.no_grad():
            x = torch.LongTensor([text_ids]).to(pl_module.device)
            x_lengths = torch.LongTensor([len(text_ids)]).to(pl_module.device)
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
        self.check_graceful_stop(trainer)

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        global_val_loss = metrics.get("val_loss", 0.0)
        val_mel = metrics.get("val_mel", 0.0)
        if isinstance(global_val_loss, torch.Tensor):
            global_val_loss = global_val_loss.item()
        if isinstance(val_mel, torch.Tensor):
            val_mel = val_mel.item()

        print("\n" + "="*65)
        print("          SUTTAPLAYER NEURAL VOICE ACTIVE UAT SWEPT          ")
        print("="*65)
        print(f" * Active Epoch: {trainer.current_epoch:<5} | Global Val Loss: {global_val_loss:.4f}")
        print("-"*65)

        preview_dir = "/content/preview"
        os.makedirs(preview_dir, exist_ok=True)
        uat_rows = []

        for name, p in self.probes.items():
            text_ids = self.phonemes_to_ids(p["phonemes"])
            try:
                y = self.synthesize_probe_audio(pl_module, text_ids)
                
                # Write preview WAV file programmatically with peak normalization
                wav_path = os.path.join(preview_dir, f"{name}.wav")
                max_val = np.max(np.abs(y)) if len(y) > 0 else 0
                y_norm = (y / max_val * 0.95) if max_val > 1e-5 else y
                audio_int16 = (y_norm * 32767.0).clip(-32768, 32767).astype(np.int16)
                wavfile.write(wav_path, 22050, audio_int16)
                
                if p["type"] == "pause":
                    pause_ms = self.measure_silence_gap(y)
                    target = p["target_ms"]
                    delta = pause_ms - target
                    print(f"  [{p['symbol']}] {name:<25} | Measured Pause: {pause_ms:>5.1f} ms (Target: {target} ms | Delta: {delta:+.1f} ms) [🔊 Preview Ready]")
                    uat_rows.append({
                        "name": name,
                        "symbol": p["symbol"],
                        "type": p["type"],
                        "measured_ms": pause_ms,
                        "target_ms": target,
                        "delta_ms": delta
                    })
                else:
                    print(f"  [🔊] {name:<25} | Acoustic sibilance test synthesized and exported cleanly.")
                    uat_rows.append({
                        "name": name,
                        "symbol": "",
                        "type": p["type"],
                        "measured_ms": 0.0,
                        "target_ms": 0.0,
                        "delta_ms": 0.0
                    })
            except Exception as e:
                print(f"  [ERROR] Probe synthesis failed for {name}: {e}")

        print("="*65 + "\n")

        # Emit/Append UAT probe metrics to uat_metrics.csv for Sheets Keep-Alive tracking
        uat_csv_paths = [
            "/content/drive/MyDrive/piper_training/uat_metrics.csv",
            "/content/piper_training/uat_metrics.csv",
            "./uat_metrics.csv"
        ]
        target_csv_path = None
        for path in uat_csv_paths:
            parent = os.path.dirname(path)
            if parent == "" or os.path.exists(parent):
                target_csv_path = path
                break

        if target_csv_path and uat_rows:
            try:
                file_exists = os.path.exists(target_csv_path)
                with open(target_csv_path, "a", newline="", encoding="utf-8") as csv_file:
                    writer = csv.writer(csv_file)
                    if not file_exists:
                        writer.writerow(["epoch", "global_step", "probe_id", "symbol", "type", "measured_ms", "target_ms", "delta_ms", "val_loss", "val_mel"])
                    
                    for r in uat_rows:
                        writer.writerow([
                            trainer.current_epoch,
                            getattr(trainer, "global_step", 0),
                            r["name"],
                            r["symbol"],
                            r["type"],
                            round(r["measured_ms"], 1),
                            round(r["target_ms"], 1),
                            round(r["delta_ms"], 1),
                            f"{global_val_loss:.4f}",
                            f"{val_mel:.4f}"
                        ])
                    csv_file.flush()
                print(f"  📊 Appended {len(uat_rows)} UAT probe rows to: {target_csv_path}")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not write to uat_metrics.csv: {e}")

        self.check_graceful_stop(trainer)
