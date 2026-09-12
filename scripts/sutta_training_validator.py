# =================================================================
# SUTTAPLAYER TRAINER VALIDATOR (v1.0.0)
# =================================================================

import os
import sys
import shlex
import subprocess
import json
import torch

class PreFlightValidator:
    def __init__(self, command_str):
        self.command_str = command_str.strip()
        self.env_vars = os.environ.copy()
        self.args = []
        self.params = {}
        self._parse_command()

    def _parse_command(self):
        # 1. Parse command and extract environment variables
        tokens = shlex.split(self.command_str)
        cmd_start_idx = 0
        for i, token in enumerate(tokens):
            if "=" in token and not token.startswith("-"):
                key, val = token.split("=", 1)
                self.env_vars[key] = os.path.expandvars(val)
                cmd_start_idx = i + 1
            else:
                break
        self.args = tokens[cmd_start_idx:]
        
        # 2. Extract parameter pairs for audit assertions
        for i in range(len(self.args) - 1):
            flag = self.args[i]
            if flag.startswith("-"):
                self.params[flag] = self.args[i + 1]

    def run_audit(self):
        print("=" * 65)
        print("          SUTTAPLAYER LIVE 'GREEN FOR GO' PIPELINE AUD  ")
        print("=" * 65)
        
        csv_path = self.params.get("--data.csv_path")
        audio_dir = self.params.get("--data.audio_dir")
        map_path = self.params.get("--data.phonemes_path")
        ckpt_path = self.params.get("--ckpt_path")
        failures = []
        row_count = 0

        # Check A: Metadata CSV
        print("📋 1. Phonetic Metadata CSV:")
        if csv_path and os.path.exists(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    sample_lines = [f.readline().strip() for _ in range(3)]
                row_count = sum(1 for _ in open(csv_path, "r", encoding="utf-8"))
                print(f"  [PASS] CSV parsed successfully at: {csv_path}")
                print(f"  [INFO] Total Dataset Utterances: {row_count}")
                print("  [INFO] Header & Sample Rows:")
                for line in sample_lines:
                     print(f"    👉 {line[:95]}..." if len(line) > 95 else f"    👉 {line}")
            except Exception as e:
                failures.append(f"Metadata CSV read error: {e}")
        else:
            failures.append(f"Metadata CSV not found or missing from args. Path: {csv_path}")
        print("-" * 65)

        # Check B: WAV Audio Files
        print("🔊 2. Raw Audio WAV Directory:")
        if audio_dir and os.path.exists(audio_dir):
            try:
                wavs = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
                wav_count = len(wavs)
                anchors = ["0.wav", "500.wav", "999.wav"]
                missing_anchors = [a for a in anchors if not os.path.exists(os.path.join(audio_dir, a))]
                
                if wav_count == row_count and not missing_anchors:
                    print(f"  [PASS] Found exactly {wav_count} audio files in: {audio_dir}")
                    print("  [PASS] Anchor verification successful (0.wav, 500.wav, 999.wav are online).")
                else:
                    failures.append(f"Audio payload error: Found {wav_count}/{row_count} WAVs. Missing anchors: {missing_anchors}")
            except Exception as e:
                failures.append(f"Audio folder list error: {e}")
        else:
            failures.append(f"Audio directory not found or missing from args. Path: {audio_dir}")
        print("-" * 65)

        # Check C: Phoneme Map JSON
        print("🗺️ 3. Phoneme Map JSON Keys:")
        if map_path and os.path.exists(map_path):
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    pmap = json.load(f)
                map_keys = list(pmap.keys())
                print(f"  [PASS] Phoneme map loaded with {len(map_keys)} valid tokens.")
                print(f"  [INFO] Mapped token sample: {map_keys[:10]}")
            except Exception as e:
                failures.append(f"Phoneme map parse error: {e}")
        else:
            failures.append(f"Phoneme map JSON not found or missing from args. Path: {map_path}")
        print("-" * 65)

        # Check D: Checkpoint Resuming Metadata
        print("💾 4. Target Resuming Checkpoint State:")
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                if isinstance(checkpoint, dict):
                    epoch = checkpoint.get("epoch", "Unknown")
                    global_step = checkpoint.get("global_step", "Unknown")
                    print(f"  [PASS] Verified Checkpoint: {os.path.basename(ckpt_path)}")
                    print(f"  [INFO] Next training run will resume from: Epoch {epoch} | Step {global_step}")
                else:
                    failures.append("Checkpoint loaded but contains unexpected dict format.")
            except Exception as e:
                failures.append(f"Checkpoint state corruption: {e}")
        else:
            failures.append(f"Resume checkpoint not found or missing from args. Path: {ckpt_path}")
        print("=" * 65)

        if failures:
            print("\n❌ [CRITICAL FAILURES ENCOUNTERED] - Training blocked!")
            for fail in failures:
                print(f"  🚨 {fail}")
            print("=" * 65)
            raise AssertionError("Pre-Flight validation checks failed! Resolve errors above to proceed.")
        
        print("\n🟢 [GREEN FOR GO] All pre-flight parameters verified. Ready for launch!")
        print("=" * 65)

    def train(self):
        print("\n🚀 Spawning PyTorch Lightning training process...")
        sys.stdout.flush()
        
        try:
            # We run with stdout=None and stderr=None inside tmux so output stays fully native and interactive!
            result = subprocess.run(
                self.args,
                env=self.env_vars,
                stdout=None,
                stderr=None
            )
            if result.returncode != 0:
                print(f"\n❌ Trainer process exited with error code: {result.returncode}")
                sys.exit(result.returncode)
        except Exception as e:
            print(f"❌ Failed to run training process: {e}")
            sys.exit(1)
