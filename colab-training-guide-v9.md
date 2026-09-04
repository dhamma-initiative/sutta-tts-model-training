# Google Colab Training Guide & Google Sheets Integration (v6)

This guide outlines the execution strategy for fine-tuning your VITS `au_male_57` model [vits-colab-guidelines.md], setting up your Google Sheets real-time convergence dashboard, and configuring your local machine for seamless cloud sync [sutta-training-manager-v7.ts].

---

## 1. Directory Structure on Google Drive

Ensure your base Google Drive folder `/MyDrive/piper_training/` contains the following structured files [sutta-training-manager-v7.ts]:

```text
/My Drive (Google Drive Root)
└── piper_training/
    ├── metadata.csv                              ← Your clean, verified 1,000-line corpus
    ├── phoneme_map.json                          ← Custom 162-symbol token-to-ID table
    ├── train_sutta_voice.py                      ← train_sutta_voice-v5.py callback
    ├── sutta-training-manager-v7.ts              ← Deno orchestrator script
    ├── en_GB-northern_english_male-medium.ckpt   ← Pre-trained non-rhotic baseline model
    └── wavs/                                     ← Raw 22.05 kHz wav files directory
```

---

## 2. The Offline-First Logging Architecture

To keep the training thread completely fast and zero-latency [user query]:
1. **Zero-Blocking local CSV logging**: At the end of each validation pass, `train_sutta_voice.py` appends a single row to `uat_metrics.csv` [train_sutta_voice-v5.py]. This is a local file I/O append taking microseconds, avoiding blocking network requests, socket timeouts, or VRAM-eval stalls [user query].
2. **Colab Google Sheets Sync Cell**: A separate cell in your Jupyter notebook (`sutta-piper-trainer-v6.ipynb.txt`) uses the VM's native Google OAuth token to securely read this CSV from Google Drive and push row additions asynchronously to a designated Google Sheet [sutta-piper-trainer-v6.ipynb.txt].

---

## 3. Mathematical Normalization: Progress-to-Target (%)

The metrics appended to your CSV and written to Google Sheets are compiled as **strict mathematical convergence percentages** ($0\%$ to $100\%$), allowing all 13 heterogeneous metrics (milliseconds, hertz, decibel spectral ratios) to share a single, unified chart axis [user query]:

### A. Temporal Pause Probes (Commas, Semicolons, Colons, Em-Dashes, Ellipses, Bullets)
Calculates the absolute percentage delta relative to your Audacity-calibrated timing envelopes:
$$\text{Convergence}_i(E) = \max\left(0, 100 - \left| \frac{M_i - T_i}{T_i} \right| \times 100\right)$$

### B. Paired-Boundary Probes (Brackets, Parentheses, Braces)
Calculates the average of the leading (period-level) and trailing (comma-level) pause boundaries:
$$\text{Convergence}_i(E) = \frac{\max\left(0, 100 - \left| \frac{L_i - TL_i}{TL_i} \right| \times 100\right) + \max\left(0, 100 - \left| \frac{R_i - TR_i}{TR_i} \right| \times 100\right)}{2}$$

### C. Acoustic Sibilance Probes
Targeting a meditative, unvoiced frequency ceiling of $\le 2900 \text{ Hz}$. Hitting or dropping below this ceiling yields a perfect $100\%$ score:
$$\text{Convergence}_i(E) = \begin{cases} 100.0\% & \text{if } M_i \le 2900 \text{ Hz} \\ \max\left(0, 100 - \frac{M_i - 2900}{2900} \times 100\right) & \text{if } M_i > 2900 \text{ Hz} \end{cases}$$

### D. Acoustic Plosive Probes
Targeting a low-end sub-bass rumble ratio of $\le 0.150$. Hitting or dropping below this ceiling yields a perfect $100\%$ score:
$$\text{Convergence}_i(E) = \begin{cases} 100.0\% & \text{if } M_i \le 0.150 \\ \max\left(0, 100 - \frac{M_i - 0.150}{0.150} \times 100\right) & \text{if } M_i > 0.150 \end{cases}$$

---

## 4. Google Sheet & Convergence Chart Setup

When the notebook's Step 7 cell launches, it will securely establish **`SuttaPlayer_UAT_Convergence`** as a Google Sheet in your Drive [sutta-piper-trainer-v6.ipynb.txt].

### To Create Your Live Auto-Updating Line Chart:
1. Open your generated Google Sheet.
2. Select all columns starting from **`epoch`** (Column B) through **`probe_punct_09_braces`** (Column O).
3. Click **Insert** $\rightarrow$ **Chart**.
4. In the Chart Editor panel:
   * **Chart Type**: Set to **Line Chart**.
   * **X-axis**: Select **`epoch`** (Column B).
   * **Y-axis**: The 13 probe series will automatically register as individual lines sharing the same $0\% - 100\%$ scale!
5. As the background TMUX training process appends rows to Drive and the async Colab cell pushes them to Sheets, this chart will dynamically update, letting you visually trace the meditative convergence of SuttaPlayer's neural voice in real-time [user query]!
