#!/usr/bin/env python3
import os
import argparse
from pydub import AudioSegment
from pydub.silence import detect_silence

def strip_carrier(input_wav, output_wav, min_silence_len=100, silence_thresh=-45, keep_silence=100):
    """
    Intelligent carrier signal stripper (v3) that programmatically analyzes the acoustic
    structure of synthesized WAV files. It automatically identifies whether a file contains
    a head carrier, a tail carrier, both, or none, and performs targeted trimming while
    preserving the natural leading/trailing silences.
    """
    try:
        # Load audio
        audio = AudioSegment.from_wav(input_wav)
        total_duration = len(audio)
        
        # Find all silent intervals (using lower min_silence_len to detect tight gaps)
        silent_ranges = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        
        # Heuristic 1: Identify Head Carrier Gap (silent gap following a prefix spoken in the first 2.5 seconds)
        head_gap = None
        for r in silent_ranges:
            start, end = r
            if 700 <= end <= 2500:
                head_gap = r
                break  # First matching gap from left
                
        # Heuristic 2: Identify Tail Carrier Gap (silent gap before a suffix spoken in the final 2.5 seconds)
        tail_gap = None
        for r in reversed(silent_ranges):
            start, end = r
            rem_from_start = total_duration - start
            rem_from_end = total_duration - end
            # Gap must start within 2.5s of the end, and have at least 150ms of active speech following it
            if 600 <= rem_from_start <= 2500 and rem_from_end > 150:
                # Ensure tail gap does not overlap with or precede head gap
                if head_gap and start <= head_gap[1]:
                    continue
                tail_gap = r
                break  # First matching gap from right
                
        # Slicing bounds calculation
        slice_start = 0
        slice_end = total_duration
        actions = []
        
        if head_gap:
            slice_start = max(0, head_gap[1] - keep_silence)
            actions.append(f"stripped head (trimmed first {slice_start}ms)")
        if tail_gap:
            slice_end = min(total_duration, tail_gap[0] + keep_silence)
            actions.append(f"stripped tail (trimmed after {slice_end}ms)")
            
        # Safety net: If trims overlap or are invalid, abort and preserve original
        if slice_start >= slice_end or (slice_end - slice_start) < 150:
            print(f"⚠️ Trimming bounds collapsed for {os.path.basename(input_wav)} ([{slice_start}:{slice_end}]). Copying original.")
            audio.export(output_wav, format="wav")
            return False
            
        # Export processed audio
        trimmed_audio = audio[slice_start:slice_end]
        trimmed_audio.export(output_wav, format="wav")
        
        if actions:
            print(f"✂️ {os.path.basename(input_wav)}: {' and '.join(actions)} (Original: {total_duration}ms, Cleaned: {len(trimmed_audio)}ms)")
            return True
        else:
            print(f"✅ {os.path.basename(input_wav)}: No carrier detected. Copying original (Duration: {total_duration}ms).")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(input_wav)}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-strip head and/or tail carrier signals from synthesized WAVs (v3).")
    parser.add_argument("-i", "--input", required=True, help="Input WAV file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output WAV file or directory")
    parser.add_argument("-s", "--silence", type=int, default=100, help="Minimum silence length in ms (default: 100)")
    parser.add_argument("-t", "--threshold", type=int, default=-45, help="Silence threshold in dB (default: -45)")
    parser.add_argument("-k", "--keep", type=int, default=100, help="Silence padding to keep in ms (default: 100)")
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = [f for f in os.listdir(args.input) if f.endswith(".wav")]
        print(f"Processing {len(files)} files in folder: {args.input}...")
        processed = 0
        for f in sorted(files):
            success = strip_carrier(
                os.path.join(args.input, f), 
                os.path.join(args.output, f),
                min_silence_len=args.silence,
                silence_thresh=args.threshold,
                keep_silence=args.keep
            )
            if success:
                processed += 1
        print(f"🎉 Complete! Successfully trimmed {processed}/{len(files)} files.")
    else:
        strip_carrier(
            args.input, 
            args.output,
            min_silence_len=args.silence,
            silence_thresh=args.threshold,
            keep_silence=args.keep
        )
