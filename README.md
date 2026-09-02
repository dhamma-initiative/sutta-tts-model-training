# SuttaPlayer Neural Voice Training Pipeline

This repository contains the automated orchestration, configuration, and active validation scripts required to fine-tune a non-rhotic, Australian-accented male text-to-speech model (`au_male_57`) using the **OHF-Voice/piper1-gpl** PyTorch Lightning TTS framework.

This repository supports two execution targets:
1.  **Local Emulation (`rtx3050-colab-training-emul` branch)**: Designed to safely execute and debug the entire training loop locally on a resource-constrained NVIDIA RTX 3050 (4 GB VRAM) without risk of CUDA Out-Of-Memory (OOM) failures or wasting cloud compute credits.
2.  **Production Training (`main` or `colab` branch)**: Designed to run in Google Colab (Free Tier) leveraging a T4 GPU (15 GB VRAM) with real-time Google Drive synchronization, automated keep-alive heartbeats, and session disconnection survivability.

---

## 1. Directory Structure

To maintain a clean, professional, and modular repository, organize the files exactly as follows:

```text
sutta-tts-model-training/
├── config/
│   ├── rtx3050-train-config.yaml   # Scaled-down local testing configuration (low batch size, frequent checkpoints)
│   └── piper-train-config.yaml     # Production Google Colab T4 configuration (batch size 32, EBU R128 targets)
├── scripts/
│   ├── sutta-training-manager.ts   # Automated Deno script (handles environments, compilations, tmux, and rsync)
│   ├── train_sutta_voice.py        # Custom PyTorch Lightning callback (Pre-phonemized, zero-carrier UAT loop)
│   ├── corpus-quality-diff.py      # Quality assurance audio analyzer and diagnostic diff script
│   └── sutta-carrier-strip-v3.py   # Carrier signal stripping utility for raw 2023 VITS audio splits
├── README.md                       # This documentation and playbook
└── .gitignore                      # Python and system cache ignore rules
```

---

## 2. In-Training active UAT Validation Callback

We have decoupled training completion from loose statistical averages. Instead of waiting for an arbitrary global loss value, the training pipeline utilizes an **Active Closed-Loop UAT Validation Callback** (`train_sutta_voice.py`).

At the end of every validation epoch, the trainer automatically:
1.  Intercepts the training loop and runs in-memory inference on **11 pre-phonemized acoustic and punctuation validation probes** (bypassing any slow runtime phonemizer engines).
2.  Passes the generated waveforms through a physical acoustic profiler (using `librosa` and `scipy`).
3.  **Acoustic Tests**: Measures high-frequency sibilance centroid balances and low-frequency plosive ratios.
4.  **Punctuation Tests**: Measures the exact silence interval (in milliseconds) allocated to commas, semicolons, colons, em-dashes, ellipses, bullets, and brackets (`[ ]`).
5.  **Graceful Exit**: Automatically terminates training and outputs an ONNX-ready production model *only* when global validation loss stabilizes AND all 11 programmatic unit tests pass their specific millisecond thresholds.

---

## 3. Emulation Branch (`rtx3050-colab-training-emul`) Playbook

Run these steps locally on `kassapa-l` to verify that the Cython Monotonic Alignment Search (MAS) C-extensions compile correctly, datasets load cleanly, and the active UAT callback intercepts and monitors metrics natively.

### Step 3.1: Prepare your Sandbox Directory
Create an offline sandbox folder on your machine and lay out your external, heavy files relatives to the repo:

```text
sutta_train_sandbox/
├── sutta-tts-model-training/       # Clone of this repository
├── G^latest.pth                    # Your pre-trained non-rhotic base checkpoint
├── metadata.csv                    # Clean 2-column LJSpeech file: filename.wav|phonemes
├── phoneme_map.json                # Custom 162-symbol token-to-ID table
├── wavs/                           # Folder containing your 1,000 trimmed and polished audio files
│   ├── probe_01_sibilance.wav
│   └── ...
└── piper1-gpl/                     # Cloned upstream repository (auto-cloned by manager)
```

### Step 3.2: Local Requirements Installation & Cython MAS Compilation
Open your terminal, navigate to your sandbox root, activate your python virtual environment, and compile the PyTorch C-extensions:

```bash
# 1. Activate your local virtual environment
source ~/dev/dhammatalks.org-suttas/.venv-3.8.18/bin/activate

# 2. Enter the training repo folder (cloned by manager or cloned manually)
git clone https://github.com/OHF-Voice/piper1-gpl.git
cd piper1-gpl/

# 3. Install in editable training mode
pip install -e '.[train]'

# 4. Compile Cython C-Extensions for Monotonic Alignment Search (Speedup Factor: 10x-12x)
bash build_monotonic_align.sh
```
*Note: If the shell script fails to map compiler paths, compile via Python's setup script directly:*
```bash
python3 src/python/setup.py build_ext --inplace
```

### Step 3.3: Execute the Local Emulation Run
From your sandbox root directory, execute the Python training command pointing to your scaled-down RTX 3050 config:

```bash
python3 -m piper.train fit \
  --config sutta-tts-model-training/config/rtx3050-train-config.yaml \
  --data.voice_name=au_male_57 \
  --data.csv_path=metadata.csv \
  --data.audio_dir=wavs/ \
  --data.phoneme_type=text \
  --data.phonemes_path=phoneme_map.json \
  --data.cache_dir=./piper_cache_local \
  --data.batch_size=2 \
  --model.sample_rate=22050 \
  --ckpt_path=G^latest.pth
```

#### Verification Checks:
*   **Step Check**: Because `val_check_interval` is set to `10` in the RTX 3050 config, after exactly 10 steps, the active validation sweep will trigger.
*   **UAT Check**: The custom validation callback will boot up, print your validation metrics, run waveform analysis on your probes, and display the side-by-side programmatic unit-test results.
*   **Termination**: Once you see the first validation pass print its checkmarks, hit `Ctrl` + `C`. Your dry-run emulation is 100% verified!

---

## 4. Production Colab Playbook (Transitioning to cloud training)

Once the local dry-run is complete, prepare for production cloud training.

### Step 4.1: Populate Google Drive
Upload your sandbox files directly to your permanently mapped Google Drive path at `/My Drive/piper_training/`.

### Step 4.2: Open the Notebook and Run
1.  Upload `sutta-piper-trainer-v3.ipynb` to your Google Colab account.
2.  Set the runtime to **T4 GPU** (15 GB VRAM).
3.  Execute **Step 1** (Mount Drive) and **Step 2** (Provision Deno).
4.  Execute **Step 3** (Environment Initialization). The Deno orchestrator will automatically provision micromamba, create a sandboxed **Python 3.8.18** virtual environment, register `train_sutta_voice.py`, and compile Cython MAS C-extensions natively.
5.  Execute **Step 5** and **Step 6** to initiate training inside a background TMUX session and launch the keep-alive monitor daemon.

### Step 4.3: Real-Time Audio UAT
Open the TensorBoard instance embedded in your notebook (`Step 4`). Under the **Audio tab**, you can listen to the synthesized files of your 11 probes written after every epoch. You can literally *hear* the model learn your non-rhotic Australian accent and naturally space out its punctuation pauses as training progresses!
