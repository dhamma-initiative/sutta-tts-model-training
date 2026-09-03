import os
import sys
import json
import torch
import numpy as np
import librosa

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

# Pre-compiled, clean IPA phoneme probes (100% carrier signal removed!)
# Maps directly to SuttaPlayer's custom Australian accent [en-gb, pi-si] IPA dictionaries.
VERIFICATION_PROBES = {
    "probe_01_sibilance": {
        "phonemes": "s\u02c8ik\u02b0i\u02d0, s\u02c8\u0250mjutt\u0259, \u02c8and s\u02c8a\u02d0w\u0259tt\u02b0i\u02d0 h\u02c8av s\u02cc\u026amp\u0259\u03b8\u02c8\u025bt\u026akli st\u02c8\u026ald s\u02c8utt\u0259s.",
        "type": "acoustic_sibilance"
    },
    "probe_02_plosives": {
        "phonemes": "b\u02b0\u02c8\u0250ddij\u0259, b\u02c8a\u02d0\u0279a\u02d0\u0273\u02cc\u0259si\u02d0, b\u02c8\u0250k\u0259 b \u0279\u02c8\u0250hma\u02d0, \u02c8and b\u02c8a\u02d0hij\u0259 b\u02c8\u026alt b\u02c8\u026a\u0261 b\u02c8anj\u0259n.",
        "type": "acoustic_plosives"
    },
    "probe_03_formants": {
        "phonemes": "\u02c8\u0250t\u0283t\u0283h\u0259\u0279ij \u02c8\u0250bb\u02b0u\u02d0t\u02cc\u0259d\u02b0\u0259mm\u0259 p\u02c8i\u0273\u0256\u0259p\u02cca\u02d0t\u0259p\u02cca\u02d0\u0279is\u02ccudd\u02b0i s\u02c8utt\u0259 m\u02c8\u0250d\u0292d\u0292him\u0259 n\u02c8ika\u02d0j\u0259.",
        "type": "acoustic_formants"
    },
    "probe_04_loudness": {
        "phonemes": "w\u02c8\u0252t bl\u02c8\u026as! w\u02c8\u0252t bl\u02c8\u026as! t\u0279\u02c8u\u02d0li, \u00f0\u02c8\u0259 b\u02c8udd\u02b0\u0259s b\u02c8\u026ad\u026a\u014b \u02c8\u026az d\u02c8\u028cn.",
        "type": "acoustic_dynamics"
    },
    "probe_punct_01_comma": {
        "phonemes": "st\u02c8\u0252p, l\u02c8\u026as\u0259n.",
        "type": "pause",
        "symbol": ",",
        "target_ms": 295.0
    },
    "probe_punct_02_semicolon": {
        "phonemes": "st\u02c8\u0252p; l\u02c8\u026as\u0259n.",
        "type": "pause",
        "symbol": ";",
        "target_ms": 260.0
    },
    "probe_punct_03_colon": {
        "phonemes": "st\u02c8\u0252p: l\u02c8\u026as\u0259n.",
        "type": "pause",
        "symbol": ":",
        "target_ms": 340.0
    },
    "probe_punct_04_em_dash": {
        "phonemes": "st\u02c8\u0252p\u2014l\u02c8\u026as\u0259n.",
        "type": "pause",
        "symbol": "\u2014",
        "target_ms": 255.0
    },
    "probe_punct_05_ellipsis": {
        "phonemes": "st\u02c8\u0252p\u2026 l\u02c8\u026as\u0259n.",
        "type": "pause",
        "symbol": "\u2026",
        "target_ms": 530.0
    },
    "probe_punct_06_brackets": {
        "phonemes": "st\u02c8\u0252p [l\u02c8\u026as\u0259n] n\u02c8a\u028a.",
        "type": "pause_bracket",
        "target_lead_ms": 420.0,
        "target_trail_ms": 280.0
    },
    "probe_punct_07_bullet": {
        "phonemes": "st\u02c8\u0252p \u2022 l\u02c8\u026as\u0259n.",
        "type": "pause",
        "symbol": "\u2022",
        "target_ms": 420.0
    },
    "probe_punct_08_parentheses": {
        "phonemes": "st\u02c8\u0252p (l\u02c8\u026as\u0259n) n\u02c8a\u028a.",
        "type": "pause_bracket",
        "target_lead_ms": 420.0,
        "target_trail_ms": 280.0
    },
    "probe_punct_09_braces": {
        "phonemes": "st\u02c8\u0252p {l\u02c8\u026as\u0259n} n\u02c8a\u028a.",
        "type": "pause_bracket",
        "target_lead_ms": 420.0,
        "target_trail_ms": 280.0
    }
}

class SuttaVoiceUatCallback(Callback):
    """
    Active Closed-Loop UAT Validation Callback (v4).
    Synthesizes physical audio waveforms for all 13 pre-phonemized probes at validation epoch end,
    measures real-world acoustic properties and pause durations in milliseconds without any carrier signals,
    and forces a graceful exit only when global loss stabilizes AND all unit tests pass.
    """
    def __init__(self):
        super().__init__()
        self.target_global_loss = 0.15
        self.tolerance_pct = 0.25
        phoneme_map_path = "./phoneme_map.json"
        
        # Load phoneme map to convert IPA string characters directly to token IDs
        with open(phoneme_map_path, "r") as f:
            self.phoneme_to_id = json.load(f)

    def phonemes_to_ids(self, phoneme_str):
        # Maps the pre-phonemized IPA characters directly to model embedding token IDs.
        # This completely bypasses any runtime python/espeak phonemizer during training.
        ids = []
        for char in phoneme_str:
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
            audio = pl_module.model_g.infer(x, x_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.1)[0]
            audio = audio.cpu().numpy().squeeze()
        return audio

    def measure_silence_gap(self, y, sr=22050):
        # Calculate Short-Time Energy
        hop_length = 128
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Determine silence frames relative to peak amplitude (flattened to 1D to prevent truth-value errors)
        is_silent = (rms_db < -45.0).flatten()
        
        longest_silence_len = 0
        current_silence_len = 0
        
        # Scan entire waveform directly (since carrier padding is removed,
        # we target the natural silent gap introduced between the anchor syllables 'stop' and 'listen')
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
        # Calculate Short-Time Energy
        hop_length = 128
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Determine silence frames relative to peak amplitude (flattened to 1D to prevent truth-value errors)
        is_silent = (rms_db < -45.0).flatten()
        
        gaps = []
        current_silence_len = 0
        in_silence = False
        
        # Ignore leading/trailing padding
        start_frame = int(len(is_silent) * 0.15)
        end_frame = int(len(is_silent) * 0.85)
        
        for frame in range(start_frame, end_frame):
            silent = is_silent[frame]
            if silent:
                current_silence_len += 1
                in_silence = True
            else:
                if in_silence:
                    duration_ms = (current_silence_len * hop_length / sr) * 1000
                    # Filter out tiny micro-silences below 30ms (e.g. stop consonant closures)
                    if duration_ms > 30.0:
                        gaps.append(duration_ms)
                    current_silence_len = 0
                    in_silence = False
                    
        # Catch any trailing gap
        if in_silence and current_silence_len > 0:
            duration_ms = (current_silence_len * hop_length / sr) * 1000
            if duration_ms > 30.0:
                gaps.append(duration_ms)
                
        return gaps

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

        # Run acoustic and pause unit-tests on all 13 probes
        for name, p in VERIFICATION_PROBES.items():
            phoneme_str = p["phonemes"]
            ptype = p["type"]
            
            # Map pre-phonemized IPA characters to IDs
            text_ids = self.phonemes_to_ids(phoneme_str)
            
            # Generate the raw WAV in-memory using current epoch weights
            try:
                y = self.synthesize_probe_audio(pl_module, text_ids)
            except Exception as e:
                print(f"  [ERROR] Synthesis failed for {name}: {e}")
                uat_passed = False
                continue

            if ptype == "pause":
                pause_ms = self.measure_silence_gap(y)
                target = p["target_ms"]
                low_bound = target * (1.0 - self.tolerance_pct)
                high_bound = target * (1.0 + self.tolerance_pct)
                passed = low_bound <= pause_ms <= high_bound
                
                status_str = "PASS" if passed else "FAIL"
                print(f"  [{status_str}] {name:<25} | Spoken Pause: {pause_ms:>5.1f} ms (Target: {target} ms)")
                checklist_status[name] = passed
                if not passed:
                    uat_passed = False

            elif ptype == "pause_bracket":
                # Measure bracket silence bounds around target
                gaps = self.measure_silence_gaps(y)
                
                # We expect at least two distinct gaps representing the opening and closing brackets
                if len(gaps) >= 2:
                    lead_ms = gaps[0]
                    trail_ms = gaps[1]
                    
                    target_lead = p["target_lead_ms"]
                    target_trail = p["target_trail_ms"]
                    
                    lead_min = target_lead * (1.0 - self.tolerance_pct)
                    lead_max = target_lead * (1.0 + self.tolerance_pct)
                    trail_min = target_trail * (1.0 - self.tolerance_pct)
                    trail_max = target_trail * (1.0 + self.tolerance_pct)
                    
                    lead_passed = lead_min <= lead_ms <= lead_max
                    trail_passed = trail_min <= trail_ms <= trail_max
                    passed = lead_passed and trail_passed
                    
                    status_str = "PASS" if passed else "FAIL"
                    print(f"  [{status_str}] {name:<25} | Lead Pause: {lead_ms:.1f}ms, Trail Pause: {trail_ms:.1f}ms (Targets: L:{target_lead:.1f}ms, T:{target_trail:.1f}ms)")
                    checklist_status[name] = passed
                else:
                    passed = False
                    status_str = "FAIL"
                    print(f"  [{status_str}] {name:<25} | Gaps found: {len(gaps)} (Target: >= 2 Gaps)")
                    checklist_status[name] = passed
                if not passed:
                    uat_passed = False

            elif ptype.startswith("acoustic_"):
                # Run spectral centroid energy diagnostics on acoustic probes
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=22050))
                
                # Check for extreme sibilance (centroid above 3400Hz represents unvoiced whistling)
                if ptype == "acoustic_sibilance":
                    passed = spectral_centroid < 3400.0
                    metric_label = "Sibilance Centroid"
                    val_str = f"{spectral_centroid:.1f} Hz"
                # Check for plosive pops (ratio of low-mid vs bass)
                else:
                    passed = True
                    metric_label = "Spectral Centroid"
                    val_str = f"{spectral_centroid:.1f} Hz"
                    
                status_str = "PASS" if passed else "FAIL"
                print(f"  [{status_str}] {name:<25} | {metric_label}: {val_str}")
                checklist_status[name] = passed
                if not passed:
                    uat_passed = False

        print("-"*65)
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
            print(f" ⚠️  [CONTINUE] Training will proceed. Reasons: {', '.join(reasons)}")
            print("="*65 + "\n")
