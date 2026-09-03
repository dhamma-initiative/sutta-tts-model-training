#!/bin/bash
# scripts/3.launch-piper-training.sh
# =====================================================================
# SuttaPlayer Custom CLI-Only Training Execution Script
# Bypasses problematic YAML parsing logic by directly mapping properties
# =====================================================================

# Ensure Python search path can locate our integrated train_sutta_voice callback
export PYTHONPATH="./sutta-tts-model-training/scripts:$PYTHONPATH"

echo "🚀 Launching SuttaPlayer Piper1 VITS Fine-Tuning..."
echo "  • Dataset: ./metadata.csv"
echo "  • Audio directory: ./wavs"
echo "  • Symbol ID Map: ./phoneme_map.json"
echo "  • Base model: ./en_GB-northern_english_male-medium.ckpt"
echo "---------------------------------------------------------------------"

python -m piper.train fit \
  --data.voice_name "en_gb-suttaplayer-medium" \
  --data.csv_path ./metadata.csv \
  --data.audio_dir ./wavs \
  --data.cache_dir ./tmp/cache \
  --data.config_path ./tmp/config.json \
  --data.phoneme_type text \
  --data.phonemes_path ./phoneme_map.json \
  --data.num_symbols 256 \
  --data.batch_size 2 \
  --data.espeak_voice "en-gb" \
  --model.sample_rate 22050 \
  --model.mel_channels 80 \
  --model.mel_fmin 0 \
  --model.mel_fmax 8000 \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision 16-mixed \
  --trainer.callbacks.class_path "train_sutta_voice.SuttaVoiceUatCallback" \
  --ckpt_path ./en_GB-northern_english_male-medium.ckpt
