# =================================================================
# SUTTAPLAYER ONNX EXPORT & QUANTIZATION PIPELINE (v5.0.0)
# =================================================================
# This script automates the complete model deployment pipeline:
# 1. Exports PyTorch Lightning checkpoints to full-precision FP32 ONNX graphs.
# 2. Performs dynamic INT8 quantization to output lightweight 38MB PWA graphs.
# 3. Generates the required sidecar JSON configuration files for both runtimes.
# =================================================================

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# Try to import onnxruntime quantization
try:
    import onnxruntime
    from onnxruntime.quantization import QuantType, quantize_dynamic
except ImportError as e:
    print(f"❌ Error: Failed to import onnxruntime quantization: {e}")
    print("Please run: pip install onnx onnxruntime jsonargparse docstring-parser")
    sys.exit(1)

def run_command(cmd, env=None):
    """Executes a system command and streams output in real-time."""
    print(f"🏃 Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line, end=\"\", flush=True)
    process.wait()
    return process.returncode == 0

def generate_piper_json_sidecar(config_path, output_json_path, phoneme_map_path=None):
    """Generates the required Piper .onnx.json metadata file."""
    print(f"📖 Parsing training config: {config_path}")
    if not os.path.exists(config_path):
        print(f"⚠️ Warning: Config file not found at {config_path}. Creating fallback metadata.")
        vits_config = {}
    else:
        with open(config_path, \"r\", encoding=\"utf-8\") as f:
            vits_config = json.load(f)

    # Initialize standard Piper metadata structure
    piper_config = {
        \"audio\": {
            \"sample_rate\": vits_config.get(\"data\", {}).get(\"sampling_rate\", 22050)
        },
        \"espeak\": {
            \"voice\": vits_config.get(\"data\", {}).get(\"espeak_voice\", \"en-gb\")
        },
        \"inference\": {
            \"noise_scale\": 0.667,   # Optimized for non-rhotic sibilance control
            \"length_scale\": 1.1,    # Matches natural unhurried reading cadence
            \"noise_w\": 0.8,         # Standard sibilance suppression
        },
        \"phoneme_type\": vits_config.get(\"data\", {}).get(\"phoneme_type\", \"text\"),
        \"phoneme_map\": {},
        \"phoneme_id_map\": {},
        \"num_symbols\": 0,
        \"num_speakers\": vits_config.get(\"data\", {}).get(\"n_speakers\", 1),
        \"speaker_id_map\": {},
    }

    # Populate phoneme_id_map using your custom map if provided, otherwise fall back to training config symbols
    pmap = None
    if phoneme_map_path and os.path.exists(phoneme_map_path):
        print(f\"🗺️ Loading custom phoneme map from: {phoneme_map_path}\")
        with open(phoneme_map_path, \"r\", encoding=\"utf-8\") as f:
            pmap = json.load(f)
    elif \"symbols\" in vits_config:
        print(\"📋 Using symbols list from training config.\")
        pmap = {s: i for i, s in enumerate(vits_config[\"symbols\"])}
    elif \"phoneme_id_map\" in vits_config:
        print(\"📋 Extracting phoneme_id_map directly from training config.\")
        pmap = vits_config[\"phoneme_id_map\"]

    if pmap:
        # Standard Piper ONNX expects phoneme strings mapped to a list containing their single ID
        piper_config[\"phoneme_id_map\"] = {
            k: [v] if isinstance(v, int) else v for k, v in pmap.items()
        }
        piper_config[\"num_symbols\"] = len(pmap)
    else:
        print(\"⚠️ Warning: No phoneme symbols or maps could be resolved.\")

    with open(output_json_path, \"w\", encoding=\"utf-8\") as f:
        json.dump(piper_config, f, indent=2)

    print(f\"✅ Created Piper metadata file: {output_json_path}\")

def main():
    parser = argparse.ArgumentParser(description=\"SuttaPlayer Model ONNX Export & Quantization Pipeline\")
    parser.add_argument(
        \"--checkpoint\",
        type=str,
        required=True,
        help=\"Path to your trained PyTorch checkpoint (.ckpt) file\"
    )
    parser.add_argument(
        \"--config\",
        type=str,
        required=True,
        help=\"Path to the training config .json file (e.g. en_gb-suttaplayer-medium.json)\"
    )
    parser.add_argument(
        \"--phoneme-map\",
        type=str,
        default=None,
        help=\"Path to your custom phoneme_map.json (optional)\"
    )
    parser.add_argument(
        \"--output-dir\",
        type=str,
        default=\"/content/drive/MyDrive/piper_training/onnx_models\",
        help=\"Directory to save exported ONNX models\"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fp32_onnx_path = os.path.join(args.output_dir, \"pali_vits_fp32.onnx\")
    quant_onnx_path = os.path.join(args.output_dir, \"pali_vits_quant.onnx\")
    
    fp32_json_path = fp32_onnx_path + \".json\"
    quant_json_path = quant_onnx_path + \".json\"

    print(\"=================================================\")
    print(\"🎙️ SUTTAPLAYER ONNX EXPORT & QUANTIZATION PIPELINE\")
    print(\"=================================================\")
    print(f\"Checkpoint: {args.checkpoint}\")
    print(f\"Config:     {args.config}\")
    print(f\"Output Dir: {args.output_dir}\\n\")

    # 1. Export standard FP32 model
    print(\"📦 STEP 1: Exporting full precision FP32 ONNX model...\")
    
    # 🚀 EXPLICIT RESOLUTION: Bypass sys.executable and use the Colab-managed launcher binary path!
    python_exec = \"/usr/local/bin/python\"
    if not os.path.exists(python_exec):
        python_exec = \"python\" # Fallback to path resolution
        
    export_cmd = [
        python_exec, \"-m\", \"piper.train.export_onnx\",
        \"--checkpoint\", args.checkpoint,
        \"--output-file\", fp32_onnx_path
    ]
    # Set PYTHONPATH to include local repo scripts
    env = os.environ.copy()
    env[\"PYTHONPATH\"] = f\"/content/drive/MyDrive/sutta-tts-model-training/scripts:{env.get('PYTHONPATH', '')}\"
    
    success = run_command(export_cmd, env=env)
    if not success:
        print(\"❌ Error: FP32 ONNX export failed.\")
        sys.exit(1)
    print(f\"✅ Full precision FP32 model exported to: {fp32_onnx_path}\\n\")

    # 2. Dynamic INT8 quantization
    print(\"⚡ STEP 2: Creating dynamic INT8 quantized model for mobile/PWA runtimes...\")
    try:
        quantize_dynamic(
            model_input=fp32_onnx_path,
            model_output=quant_onnx_path,
            weight_type=QuantType.QUInt8
        )
        print(f\"✅ Quantized model created at: {quant_onnx_path}\\n\")
    except Exception as e:
        print(f\"❌ Error during quantization: {e}\")
        sys.exit(1)

    # 3. Generate sidecar JSON metadata for both models
    print(\"📋 STEP 3: Generating metadata sidecar files...\")
    generate_piper_json_sidecar(args.config, fp32_json_path, args.phoneme_map)
    generate_piper_json_sidecar(args.config, quant_json_path, args.phoneme_map)

    print(\"\\n=================================================\")
    print(\"🎉 ONNX EXPORT COMPLETE!\")
    print(f\"   FP32 (Desktop/High Resources): {fp32_onnx_path} (~116MB)\")
    print(f\"   INT8 (PWA/Low Resources):     {quant_onnx_path} (~38MB)\")
    print(\"=================================================\")

if __name__ == \"__main__\":
    main()
