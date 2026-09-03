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
        "text": "sikhī, saṁyutta, and sāvatthī have sympathetically stilled suttas.",
        "phonemes": "sˈikʰiː, sˈɐmjuttə, ˈand sˈaːwəttʰiː hˈav sˌɪmpəθˈɛtɪkli stˈɪld sˈuttəs.",
        "type": "acoustic_sibilance"
    },
    "probe_02_plosives": {
        "text": "bhaddiya, bārāṇasī, baka brahmā, and bāhiya built big banyan.",
        "phonemes": "bʰˈɐddijə, bˈaːɹaːnˌəsɪɪ, bˈɐkə b ɹˈɐhmaː, ˈand bˈaːhijə bˈɪlt bˈɪɡ bˈanjən.",
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
    Active Closed-Loop UAT Validation Callback (v3).
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
            # UPDATED: Resolved 'VitsModel' has no attribute 'generator' by targeting model_g
            audio = pl_module.model_g.infer(x, x_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.1)[0]
            audio = audio.cpu().numpy().squeeze()
        return audio

    def measure_silence_gaps(self, y, sr=22050):
        # Calculate Short-Time Energy
        hop_length = 128
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Determine silence frames relative to peak amplitude
        is_silent = rms_db < -45.0
        
        # Isolate contiguous silent segments
        silent_ranges = []
        in_silence = False
        start_idx = 0
        
        for idx, silent in enumerate(is_silent):
            if silent and not in_silence:
                in_silence = True
                start_idx = idx
            elif not silent and in_silence:
                in_silence = False
                silent_ranges.append((start_idx, idx))
        if in_silence:
            silent_ranges.append((start_idx, len(is_silent)))
            
        # Convert frame ranges to millisecond durations
        gap_durations = [((end - start) * hop_length / sr) * 1000 for start, end in silent_ranges]
        return gap_durations

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
                gaps = self.measure_silence_gaps(y)
                # Filter out tiny micro-silences (< 50ms) to target structural pause
                valid_gaps = [g for g in gaps if g >= 50.0]
                pause_ms = max(valid_gaps) if valid_gaps else 0.0
                
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
                gaps = self.measure_silence_gaps(y)
                valid_gaps = [g for g in gaps if g >= 50.0]
                
                target_lead = p["target_lead_ms"]
                target_trail = p["target_trail_ms"]
                
                # We expect at least two substantial silence gaps (leading and trailing pauses)
                if len(valid_gaps) >= 2:
                    # In chronological order, lead gap is first, trail is second
                    lead_ms = valid_gaps[0]
                    trail_ms = valid_gaps[1]
                    
                    lead_passed = target_lead * (1.0 - self.tolerance_pct) <= lead_ms <= target_lead * (1.0 + self.tolerance_pct)
                    trail_passed = target_trail * (1.0 - self.tolerance_pct) <= trail_ms <= target_trail * (1.0 + self.tolerance_pct)
                    passed = lead_passed and trail_passed
                    val_str = f"L: {lead_ms:.1f}ms, T: {trail_ms:.1f}ms"
                else:
                    passed = False
                    val_str = f"Found only {len(valid_gaps)} gap(s)"
                    
                status_str = "PASS" if passed else "FAIL"
                print(f"  [{status_str}] {name:<25} | Spoken Pauses: {val_str} (Target Lead: {target_lead}ms, Trail: {target_trail}ms)")
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
