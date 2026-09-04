# Sutta TTS Model Training — Google Colab Guide (v11)
This guide documents the pristine, single-file "Source of Truth" architecture for running Piper1 VITS fine-tuning on Google Colab (Free Tier) using your Audacity-calibrated Australian Male training corpus.

---

## 🚀 Architectural Setup (Source of Truth)
To prevent chansing broken references and git conflicts, all configuration assets are stored directly inside your Google Drive clone under:
`/content/drive/MyDrive/sutta-tts-model-training`

### Step 1: Mount Google Drive
Create a new cell in Google Colab and run:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Install and Load `colab-xterm`
```python
!pip install colab-xterm
%reload_ext colabxterm
```

### Step 3: Copy the Clean Orchestrator
Copy the single versionless orchestrator script from Drive directly into your Colab container workspace:
```bash
!cp /content/drive/MyDrive/sutta-tts-model-training/sutta-training-manager.ts /content/sutta-training-manager.ts
```

### Step 4: Run Isolated Environment Initialization (Blocks for ~3 mins)
This cell download micromamba, establishes a Python 3.11.9 sandbox, compiles Cython MAS align extensions, and automatically symlinks your configuration files into local paths:
```bash
!deno run --allow-all /content/sutta-training-manager.ts --init
```

---

## 🏃‍♀️ Launching Training (Choose Your Mode)

### Path A: Foreground Training (Highly Recommended)
By running in the foreground, you get **instant live logs** stream directly below your cell, **completely bypassing TMUX/xterm keyboard shortcuts** and Brave browser shortcuts (`Ctrl+B`). This cell continuously streams stdout back to the browser websocket, **preventing Colab's idle reaper from timing out**:
```bash
!deno run --allow-all /content/sutta-training-manager.ts --train-fg
```
*(If you need to stop training, simply click the square "Stop" button next to this Colab cell!)*

### Path B: Background Training (TMUX Daemon)
If you want to free up your notebook play buttons so you can execute other cells while training runs silently in the background:
```bash
!deno run --allow-all /content/sutta-training-manager.ts --train
```
#### How to view active background training logs:
```bash
!tmux capture-pane -pt piper_train
```

---

## 💓 Background Services (Monitor & Sheets Sync)

To keep your training active and chart metrics on Google Sheets without blocking your notebook kernel, launch the background services in the background using `nohup`:

### Step 1: Start Checkpoint Sync and Keep-Alive Daemons (Non-blocking)
```bash
!nohup deno run --allow-all /content/sutta-training-manager.ts --monitor > /content/monitor.log 2>&1 &
```

### Step 2: Write your Step 7 Python Sheets Sync code to a script
```python
%%writefile /content/sync_sheets.py
# [Paste your entire Step 7 Python Sync code here]
```

### Step 3: Launch Google Sheets Sync in the background (Non-blocking)
```bash
!nohup python3 -u /content/sync_sheets.py > /content/sheets_sync.log 2>&1 &
```

---

## ⚙️ Backup/Restore Your `.venv` Tarball (Eliminating Configuration Troubles)
Since remotely troubleshooting Python system environments in Colab is slow and error-prone, **do not rebuild the virtual environment from scratch every time.**

Instead, once `--init` compiles a healthy environment, **package it as a single `.tar.gz` archive on Google Drive.** On your next Colab run, extracting this archive takes **under 30 seconds** and completely restores your compiled VITS alignments, PyTorch Lightning, and compilation hooks!

### How to Back up Your Env to Drive (Run Once)
After running `--init` successfully, run this in a cell to bundle your virtual environment:
```bash
# Compress the virtual env folder directly to your Drive
!tar -czf /content/drive/MyDrive/sutta-tts-model-training/py311_env_backup.tar.gz -C /root/micromamba/envs py311
```

### How to Instantly Restore Your Env (Run on Clean Runtime)
In your next fresh Colab session, you can skip `--init` compiling by running this:
```bash
# 1. Establish micromamba bin (takes 2 seconds)
!mkdir -p /root/micromamba/envs
!curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba

# 2. Extract your compiled py311 environment directly into root (takes ~25 seconds!)
!tar -xzf /content/drive/MyDrive/sutta-tts-model-training/py311_env_backup.tar.gz -C /root/micromamba/envs

# 3. Verify healthy restore instantly!
!deno run --allow-all /content/sutta-training-manager.ts --diag-setup
```
This entirely eliminates the 3-minute compiling and C-extension build times, ensuring your virtual env is healthy 100% of the time!
