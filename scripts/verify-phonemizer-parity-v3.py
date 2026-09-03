#!/usr/bin/env python3
# scripts/verify_phonemizer_parity.py
import os
import re
import json
import sys
import unicodedata
import argparse

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title.upper():^66}")
    print("="*70)

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON file at '{path}': {e}")
        sys.exit(1)

def load_text_lines(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\r\n") for line in f]
    except Exception as e:
        print(f"❌ Error loading text file at '{path}': {e}")
        sys.exit(1)

def num_to_words(n):
    if n == 0:
        return "zero"
    
    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", 
             "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    def expand_under_1000(val):
        parts = []
        h = val // 100
        rem = val % 100
        if h > 0:
            parts.append(units[h] + " hundred")
        if rem > 0:
            if rem < 20:
                parts.append(units[rem])
            else:
                t = rem // 10
                u = rem % 10
                if u > 0:
                    parts.append(f"{tens[t]} {units[u]}")
                else:
                    parts.append(tens[t])
        return " ".join(parts)

    if n >= 1000:
        th = n // 1000
        rem = n % 1000
        th_part = units[th] + " thousand" if th < 20 else expand_under_1000(th) + " thousand"
        if rem > 0:
            return th_part + " " + expand_under_1000(rem)
        return th_part
    else:
        return expand_under_1000(n)

def expand_numbers_in_text(text):
    # Step 1: Strip commas from numbers (e.g., 1,250 -> 1250)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    
    # Step 2: Convert remaining numbers like 35 -> thirty five, 16 -> sixteen
    text = re.sub(r'\b\d+\b', lambda m: num_to_words(int(m.group(0))), text)
    
    return text

def clean_up_redundant_punctuation(result):
    result = re.sub(r'([.,;:!?])[^\S\r\n]+(?=[.,;:!?])', r'\1', result)
    result = re.sub(r'([.,;:!?])[.,;:!?]+', r'\1', result)
    result = re.sub(r'[^\S\r\n]+([.,;:!?])', r'\1', result)
    result = re.sub(r'([.,;:!?])(?=\w)', r'\1 ', result)
    result = re.sub(r'[^\S\r\n]{2,}', ' ', result)
    result = re.sub(r'^[.,;:!?\t ]+', '', result, flags=re.M)
    lines = [line.strip() for line in result.split('\n')]
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description="SuttaPlayer Python Phonemizer Parity Verifier v3")
    parser.add_argument("-p", "--phoneme_map", default="./config/en[gb]_pi[si]-suttaplayer-phoneme-map.json", help="Path to phoneme map")
    parser.add_argument("-e", "--english_dict", default="./config/pho_en[gb]-to-espeak-v1.51-ipa.json", help="Path to English dictionary")
    parser.add_argument("-pi", "--pali_dict", default="./config/pho_pi[si]-to-espeak-v1.51-ipa.json", help="Path to Pali dictionary")
    parser.add_argument("-c", "--corpus_raw", default="./corpus-preperation/notebooklm-created-colab-corpus.piper", help="Path to raw corpus")
    parser.add_argument("-cc", "--corpus_clean", default="./corpus-preperation/notebooklm-created-colab-corpus.piper.cleaned", help="Path to cleaned reference corpus")
    
    args = parser.parse_args()

    print_header("SuttaPlayer Python Phonemizer Parity Verifier v3")

    print("📂 Ingesting local validation resources...")
    english_dict_raw = load_json(args.english_dict)
    pali_dict_raw = load_json(args.pali_dict)
    raw_lines = load_text_lines(args.corpus_raw)
    expected_lines = load_text_lines(args.corpus_clean)

    print(f"  ✅ English Dictionary Loaded: {len(english_dict_raw)} entries.")
    print(f"  ✅ Pali Dictionary Loaded: {len(pali_dict_raw)} entries.")
    print(f"  ✅ Raw Corpus Loaded: {len(raw_lines)} sentences.")
    print(f"  ✅ Cleaned Reference Corpus Loaded: {len(expected_lines)} sentences.\n")

    # 1. Normalize dictionaries to NFC and Lowercase, stripping ZWJ
    english_dict = {unicodedata.normalize("NFC", k.lower()): v.replace("\u200D", "").replace("\u200d", "") for k, v in english_dict_raw.items()}
    pali_dict = {unicodedata.normalize("NFC", k.lower()): v.replace("\u200D", "").replace("\u200d", "") for k, v in pali_dict_raw.items()}

    # 2. Compile preprocessing rules
    # Case sensitive (JSON array of length 2)
    case_sensitive_rules = [
        (re.compile(r'-'), ' '),
        (re.compile(r'&'), 'and'),
        (re.compile(r'(DHAMMAPADA|ITIVUTTAKA|KHUDDAKAPĀṬHA)'), r'\1.'),
        (re.compile(r'(SUTTA NIPĀTA|THERAGĀTHĀ|THERĪGĀTHĀ|UDĀNA)\s+([:\d]+)'), r'\1. \2.'),
        (re.compile(r'\s+((AṄGUTTARA|DĪGHA|MAJJHIMA|SAṀYUTTA)\s+NIKĀYA)'), r'. \1.')
    ]
    
    # Case insensitive (JSON array of length 3 with "gmiu" flags)
    case_insensitive_rules = [
        (re.compile(r"\bblessed(\s+one)\b", re.IGNORECASE), r"bless'ed\1"),
        (re.compile(r"\b(in|past|their|future|previous|former|in|of|their|entire|the)\s+lives\b", re.IGNORECASE), r"\1 lyves")
    ]

    # Pattern to capture words (matching letters, diacritics, and apostrophes)
    word_pattern = re.compile(r"[a-zA-ZāīūṅñṭḍṇḷṁĀĪŪṄÑṬḌṆḶṀ]+(?:['’][a-zA-ZāīūṅñṭḍṇḷṁĀĪŪṄÑṬḌṆḶṀ]+)*")

    def phonemize_word(match):
        word = match.group(0)
        word_clean = word.lower().replace("’", "'")
        word_clean = unicodedata.normalize("NFC", word_clean)

        # Dictionary Lookup (English takes priority, then Pali)
        if word_clean in english_dict:
            return english_dict[word_clean]
        elif word_clean in pali_dict:
            return pali_dict[word_clean]
        else:
            raise KeyError(f"Term [{word}] (normalized: [{word_clean}]) was not found in either EN or PI dictionaries!")

    print("🧐 Auditing phonemizer logic against reference dataset...")
    
    mismatches = []
    processed_count = 0
    
    for i, (raw, expected) in enumerate(zip(raw_lines, expected_lines), 1):
        try:
            # Step A: Unicode normalization to NFC
            text = unicodedata.normalize("NFC", raw)
            
            # Step B: Strict Case-Sensitive Preprocessing Rules
            for pattern, repl in case_sensitive_rules:
                text = pattern.sub(repl, text)
                
            # Step C: Case-Insensitive Preprocessing Rules
            for pattern, repl in case_insensitive_rules:
                text = pattern.sub(repl, text)
                
            # Step D: Dynamic Number-to-Words Expansion
            text = expand_numbers_in_text(text)
                
            # Step E: Word replacement via callback preserving punctuation and spaces
            generated = word_pattern.sub(phonemize_word, text)
            
            # Step F: Final cleanup of carriage returns and whitespaces
            generated = generated.strip()
            generated = clean_up_redundant_punctuation(generated)
            
            # Verify exact NFC matches
            expected_nfc = unicodedata.normalize("NFC", expected).strip().replace("\u200D", "").replace("\u200d", "")
            generated_nfc = unicodedata.normalize("NFC", generated).replace("\u200D", "").replace("\u200d", "")
            
            if generated_nfc != expected_nfc:
                mismatches.append({
                    "line": i,
                    "raw": raw,
                    "generated": generated_nfc,
                    "expected": expected_nfc
                })
            
            processed_count += 1
            
        except Exception as err:
            mismatches.append({
                "line": i,
                "raw": raw,
                "error": str(err)
            })

    # Summary Report
    print_header("Parity Audit Results Summary")
    total_lines = len(raw_lines)
    pass_count = total_lines - len(mismatches)
    pass_pct = (pass_count / total_lines) * 100 if total_lines > 0 else 0.0

    print(f"  • Total Sentences Checked  : {total_lines}")
    print(f"  • Exact Matching Parity    : {pass_count}")
    print(f"  • Mismatches / Errors      : {len(mismatches)}")
    print(f"  • Mathematical Equivalence : {pass_pct:.2f}%")
    print("-" * 70)

    if len(mismatches) == 0:
        print("\n🎉 SUCCESS: 100% PERFECT PARITY ACHIEVED!")
        print("Your Deno/TS phonemizer and our Python verification engine produce")
        print("the exact same character-for-character cleaned output. PARITY COMPLETE! 🚀\n")
        sys.exit(0)
    else:
        print(f"\n❌ ALERT: {len(mismatches)} PARITY GAP(S) DETECTED!")
        print("Review the mismatched lines below to align regex boundaries or mapping keys:\n")
        
        for idx, m in enumerate(mismatches[:5], 1):
            print(f"[{idx}] --- Line {m['line']} ---")
            print(f"  Raw Text:  {m['raw']}")
            if "error" in m:
                print(f"  🚨 ERROR:  {m['error']}")
            else:
                print(f"  Generated: {m['generated']}")
                print(f"  Expected:  {m['expected']}")
                # Character delta highlight
                diff_chars = []
                for g, e in zip(m['generated'], m['expected']):
                    if g == e:
                        diff_chars.append(g)
                    else:
                        diff_chars.append(f"\033[91m{g}\033[0m")
                print(f"  Diff Map:  {''.join(diff_chars)}")
            print("-" * 70)
            
        if len(mismatches) > 5:
            print(f"... and {len(mismatches) - 5} more mismatches. See output files.")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
