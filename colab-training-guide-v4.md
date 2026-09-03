# Google Colab Training Guide & Architecture Blueprint (v4)

This guide details the structural layout, directory mappings, and script execution pathways required to fine-tune your non-rhotic `au_male_57` medium TTS model on Google Colab (Free Tier) using the pre-compiled `piper1-gpl` training loop.

By utilizing our automated **Deno training manager (v4)** and custom **Jupyter training console (v4)**, we establish a test-driven pipeline that compiles MAS C-extensions natively, manages background daemons under TMUX, and integrates your active validation callback with 100% pre-phonemized target IPA strings.

---

## 1. Directory Layout & Dependency Blueprint

Before launching your Google Colab instance, your permanent **Google Drive** storage folder must be structured as follows. The training console relies on these exact paths to load metadata, audio samples, and model configurations.

### Target Google Drive Directory Mapped Tree:
```text
/My Drive (Google Drive Root)
└── piper_training/                               ← Permanently Mapped Base Directory
    ├── metadata.csv                              ← LJSpeech 2-column manifest file (one per line)
    ├── phoneme_map.json                          ← Custom 162-symbol map extracted from base model config
    ├── piper-train-config.yaml                   ← PyTorch LightningCLI CLI configuration YAML (v4)
    ├── train_sutta_voice.py                      ← Your corrected, strict-mode calibrated validation callback
    ├── sutta-training-manager-v4.ts              ← Upgraded Deno manager script (maps Python 3.11.9)
    ├── en_GB-northern_english_male-medium.ckpt   ← Pre-trained non-rhotic base model checkpoint
    └── wavs/                                     ← Raw wav files directory
        └── ... (all 1,000 trimmed and enhanced WAV files)
```

---

## 2. Why the Modernization to Python 3.11.9?

Your investigation was an exceptional, highly-observant catch! Historically, the 2023 legacy `rhasspy/piper` repository was pinned to Python 3.8. However:
1.  **Maintenance Migration**: Active development has fully shifted to the Open Home Foundation (**`OHF-Voice/piper1-gpl`**).
2.  **Modern PyTorch Alignment**: The maintenance fork officially specifies `python_requires=">=3.9"`, `torch>=2,<3`, and `lightning>=2,<3`. 
3.  **Local Parity**: Running PyTorch 2.3+ and Lightning 2.3+ under an obsolete Python 3.8 environment results in immediate package incompatibilities. 
4.  **Local Venv on `kassapa-l`**: For your local sandbox dry-runs, you can now abandon Python 3.8. Create your virtual environment natively using **Python 3.11.9** (or Python 3.11/3.12) with zero installation conflicts!

---

## 3. The Cython Monotonic Alignment Search (MAS) Hurdle Solved!

If you follow standard Piper guides, compiling the C++ Monotonic Alignment Search extension is highly error-prone due to host python and GCC header mismatches. If this extension is missing, PyTorch falls back to a pure Python implementation of MAS, which is **10 to 12 times slower**, rendering training on Colab impossible.

Our upgraded **`sutta-training-manager-v4.ts`** automates this entirely under Python 3.11.9:
*   During the `--init` step, it enters your isolated Micromamba `py311` sandbox environment.
*   It executes the compiled shell script: `/content/bin/micromamba run -r /root/micromamba -n py311 bsh build_monotonic_align.sh`.
*   If compilation succeeds, PyTorch Lightning will train with a highly optimized C-extension, pushing epoch iteration times down to the absolute theoretical limit of the Colab T4 GPU!

---

## 4. Summary Training Parameters Reference

The core fit parameters dynamically compiled by our automation layer inside `/content/run_training.sh` are:

| Parameter | Execution Mapping | Purpose |
| :--- | :--- | :--- |
| **`--config`** | `/content/drive/MyDrive/piper_training/piper-train-config.yaml` | Pulls PyTorch Lightning configurations. |
| **`--data.voice_name`** | `au_male_57` | Voice ID embedded in output metadata. |
| **`--data.csv_path`** | `/content/drive/MyDrive/piper_training/metadata.csv` | File manifest path. |
| **`--data.audio_dir`** | `/content/drive/MyDrive/piper_training/wavs` | Direct wav files path. |
| **`--data.phoneme_type`** | `text` | Skips runtime espeak and loads pre-phonemized CSV. |
| **`--data.phonemes_path`** | `/content/drive/MyDrive/piper_training/phoneme_map.json` | Symbol ID map. |
| **`--data.cache_dir`** | `/content/piper_cache` | Ephemeral fast SSD folder. |
| **`--ckpt_path`** | `/content/drive/MyDrive/piper_training/en_GB-northern_english_male-medium.ckpt` | Pre-trained transfer model. |
