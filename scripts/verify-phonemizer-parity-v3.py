#!/usr/bin/env python3
"""
======================================================================
           SUTTAPLAYER PYTHON PHONEMIZER PARITY VERIFIER V3         
======================================================================
Verifies 100% mathematical and character-for-character parity between
the Deno DtSuttaPhonemizer / 2.phonemize-sources-for-tts pipeline and
the Python verification engine.
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path

# Unicode letter range including Latin extended, IPA extensions (\u0250-\u02AF),
# and IPA modifier letters (\u02B0-\u02FF) such as stress marks (ˈ, ˌ)
LETTER_CLASS = r"a-zA-Z\u00C0-\u02FF\u1E00-\u1EFF"

def num_to_words(n: int) -> str:
    """Pure Python standalone number-to-words converter (en-US)."""
    if n == 0:
        return "zero"
    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
             "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    scales = ["", "thousand", "million", "billion"]

    def _convert_group(val: int) -> str:
        words = []
        if val >= 100:
            words.append(units[val // 100] + " hundred")
            val %= 100
        if val >= 20:
            words.append(tens[val // 10])
            if val % 10:
                words.append(units[val % 10])
        elif val > 0:
            words.append(units[val])
        return " ".join(words)

    parts = []
    scale_idx = 0
    while n > 0:
        val = n % 1000
        if val > 0:
            grp_words = _convert_group(val)
            if scales[scale_idx]:
                grp_words += " " + scales[scale_idx]
            parts.append(grp_words)
        n //= 1000
        scale_idx += 1
    return " ".join(reversed(parts))

def find_file(filename: str, search_paths: list[str]) -> str:
    """Finds a file across multiple candidate relative search paths."""
    for p in search_paths:
        path_obj = Path(p)
        if path_obj.is_file():
            return str(path_obj.resolve())
    return filename

def load_json(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv_lines(filepath: str) -> list[tuple[str, str]]:
    """Loads pipe-delimited CSV lines as (id_or_file, text_or_phonemes)."""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            parts = line_str.split("|")
            if len(parts) >= 2:
                file_id = parts[0].strip()
                val = "|".join(parts[1:])  # Preserve exact whitespace/punctuation for phonemes
                entries.append((file_id, val))
            else:
                entries.append(("", line_str))
    return entries

def convert_js_flags_to_python(flags_str: str) -> int:
    """Converts JS regex flag strings (e.g. 'gmiu') to Python re flags."""
    py_flags = 0
    if "i" in flags_str:
        py_flags |= re.IGNORECASE
    if "m" in flags_str:
        py_flags |= re.MULTILINE
    if "s" in flags_str or "u" in flags_str:
        py_flags |= re.UNICODE
    return py_flags

def compile_regex_list(regex_list: list) -> list[tuple[re.Pattern, str]]:
    compiled = []
    for item in regex_list:
        if len(item) < 2:
            continue
        find_pat = item[0]
        repl_pat = item[1].replace("$1", r"\1").replace("$2", r"\2").replace("$3", r"\3")
        flags_str = item[2] if len(item) >= 3 else "gmu"
        py_flags = convert_js_flags_to_python(flags_str)
        try:
            pattern = re.compile(find_pat, py_flags)
            compiled.append((pattern, repl_pat))
        except re.error as e:
            print(f"  ⚠️ Warning: Could not compile regex '{find_pat}': {e}")
    return compiled

def process_text_replacements(text: str, regex_rules: list[tuple[re.Pattern, str]]) -> str:
    for pattern, repl in regex_rules:
        text = pattern.sub(repl, text)
    return text

def clean_up_redundant_punctuation(text: str) -> str:
    """Replicates Deno cleanUpRedundantPunctuation exactly."""
    text = re.sub(r"([.,;:!?])[^\S\r\n]+(?=[.,;:!?])", r" \1", text)
    text = re.sub(r"([.,;:!?])[.,;:!?]+", r"\1", text)
    text = re.sub(r"[^\S\r\n]+([.,;:!?])", r" \1", text)
    text = re.sub(rf"([.,;:!?])(?=[{LETTER_CLASS}0-9])", r"\1 ", text)
    text = re.sub(r"[^\S\r\n]{2,}", " ", text)
    text = re.sub(r"^[.,;:!?[^\S\r\n]]+", "", text, flags=re.MULTILINE)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)

class PythonSuttaPhonemizer:
    def __init__(self, english_dict: dict, pali_dict: dict, config_data: dict):
        self.english_dict = english_dict
        self.pali_dict = pali_dict
        
        pre_list = config_data.get("preProcessRegExFindReplaceList", [])
        post_list = config_data.get("postProcessRegExFindReplList", [])
        patch_list = config_data.get("patchProcessRegExFindReplList", [])
        
        self.pre_regexes = compile_regex_list(pre_list)
        self.post_regexes = compile_regex_list(post_list)
        self.patch_regexes = compile_regex_list(patch_list)
        
        self.word_split_regex = re.compile(
            rf"(?<![{LETTER_CLASS}])[a-zāīūṁṃṇṅñṣṭḍḷḥ’']+(?![{LETTER_CLASS}])",
            re.IGNORECASE | re.UNICODE
        )
        self.number_split_regex = re.compile(r"\b(\d+(?:,\d+)*)\b", re.IGNORECASE)

    def process(self, raw_text: str) -> str:
        # 1. Pre-processing
        text = process_text_replacements(raw_text, self.pre_regexes)
        lower_text = text.lower()
        
        # 2. Extract words & numbers (strip surrounding single quotes/apostrophes)
        matched_words = set()
        for m in self.word_split_regex.finditer(lower_text):
            w = m.group(0).strip("'’")
            if w:
                matched_words.add(w)
            
        numbers_map = {}
        for m in self.number_split_regex.finditer(lower_text):
            num_raw = m.group(0)
            num_clean = int(num_raw.replace(",", ""))
            words_str = num_to_words(num_clean)
            numbers_map[num_raw] = words_str
            for w in words_str.split():
                if w:
                    matched_words.add(w)
                    
        # 3. Replace numbers with words
        phonemised_text = lower_text
        for num_raw, words_str in numbers_map.items():
            phonemised_text = re.sub(rf"\b{re.escape(num_raw)}\b", words_str, phonemised_text)
            
        # 4. Dictionary lookup
        dictionary_map = {}
        for term in matched_words:
            repl = self.english_dict.get(term)
            if not repl:
                repl = self.pali_dict.get(term)
            if repl:
                dictionary_map[term] = repl
                
        if dictionary_map:
            # Sort terms descending by length for longest-prefix matching
            terms = sorted(dictionary_map.keys(), key=len, reverse=True)
            escaped_terms = [re.escape(t) for w in terms for t in [w]]
            pattern_str = rf"(?<![{LETTER_CLASS}])(" + "|".join(escaped_terms) + rf")(?![{LETTER_CLASS}])"
            pattern = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
            
            phonemised_text = pattern.sub(
                lambda m: dictionary_map.get(m.group(0).lower(), m.group(0)),
                phonemised_text
            )
            
        # 5. Post-processing
        phonemised_text = process_text_replacements(phonemised_text, self.post_regexes)
        phonemised_text = clean_up_redundant_punctuation(phonemised_text)
        
        # 6. Patch processing (strips ZWJ, ZWNJ, combining dot, tilde, tie bar \u0361 + enforces leading space)
        phonemised_text = process_text_replacements(phonemised_text, self.patch_regexes)
        
        return phonemised_text

def create_diff_map(gen: str, exp: str) -> str:
    """Generates a visual diff highlighting mismatched characters."""
    max_len = max(len(gen), len(exp))
    diff = []
    for i in range(max_len):
        c1 = gen[i] if i < len(gen) else "∅"
        c2 = exp[i] if i < len(exp) else "∅"
        if c1 == c2:
            diff.append(c1)
        else:
            diff.append("x")
    return "".join(diff[:80])

def main():
    parser = argparse.ArgumentParser(description="SuttaPlayer Python Phonemizer Parity Verifier V3")
    parser.add_argument("--config", default="2.phonemize-sources-for-tts-config.json", help="Path to config JSON")
    parser.add_argument("--english", default="config/pho_en[gb]-to-espeak-v1.51-ipa.json", help="Path to English IPA dict")
    parser.add_argument("--pali", default="config/pho_pi[si]-to-aksharamukha-2.1.0-ipa.json", help="Path to Pali IPA dict")
    parser.add_argument("--raw", default="corpus-preperation/metadata-text.csv", help="Path to raw text CSV")
    parser.add_argument("--reference", default="corpus-preperation/metadata-phonemes.csv", help="Path to reference phonemes CSV")
    args = parser.parse_args()

    print("======================================================================")
    print("           SUTTAPLAYER PYTHON PHONEMIZER PARITY VERIFIER V3         ")
    print("======================================================================")
    print("📂 Ingesting local validation resources...")

    config_path = find_file(args.config, [
        args.config,
        "2.phonemize-sources-for-tts-config.json",
        "corpus-preperation/2.phonemize-sources-for-tts-config.json",
        "../2.phonemize-sources-for-tts-config.json"
    ])
    english_path = find_file(args.english, [
        args.english,
        "config/pho_en[gb]-to-espeak-v1.51-ipa.json",
        "tts.word-lists/pho_en[gb]-to-espeak-v1.51-ipa.json",
        "../config/pho_en[gb]-to-espeak-v1.51-ipa.json"
    ])
    pali_path = find_file(args.pali, [
        args.pali,
        "config/pho_pi[si]-to-aksharamukha-2.1.0-ipa.json",
        "tts.word-lists/pho_pi[si]-to-aksharamukha-2.1.0-ipa.json",
        "../config/pho_pi[si]-to-aksharamukha-2.1.0-ipa.json"
    ])
    raw_path = find_file(args.raw, [
        args.raw,
        "corpus-preperation/metadata-text.csv",
        "metadata-text.csv",
        "../corpus-preperation/metadata-text.csv"
    ])
    ref_path = find_file(args.reference, [
        args.reference,
        "corpus-preperation/metadata-phonemes.csv",
        "metadata-phonemes.csv",
        "../corpus-preperation/metadata-phonemes.csv"
    ])

    config_data = load_json(config_path)
    english_dict = load_json(english_path)
    pali_dict = load_json(pali_path)
    raw_lines = load_csv_lines(raw_path)
    ref_lines = load_csv_lines(ref_path)

    print(f"  ✅ English Dictionary Loaded: {len(english_dict)} entries.")
    print(f"  ✅ Pali Dictionary Loaded: {len(pali_dict)} entries.")
    print(f"  ✅ Raw Corpus Loaded: {len(raw_lines)} sentences.")
    print(f"  ✅ Cleaned Reference Corpus Loaded: {len(ref_lines)} sentences.\n")

    phonemizer = PythonSuttaPhonemizer(english_dict, pali_dict, config_data)

    print("🧐 Auditing phonemizer logic against reference dataset...\n")

    matches = 0
    mismatches = 0
    total = min(len(raw_lines), len(ref_lines))
    mismatch_details = []

    for i in range(total):
        raw_id, raw_text = raw_lines[i]
        ref_id, ref_text = ref_lines[i]

        generated_ipa = phonemizer.process(raw_text)

        if generated_ipa == ref_text:
            matches += 1
        else:
            mismatches += 1
            if len(mismatch_details) < 10:
                mismatch_details.append((i + 1, raw_text, generated_ipa, ref_text))

    parity_pct = (matches / total * 100) if total > 0 else 0.0

    print("======================================================================")
    print("                     PARITY AUDIT RESULTS SUMMARY                   ")
    print("======================================================================")
    print(f"  • Total Sentences Checked  : {total}")
    print(f"  • Exact Matching Parity    : {matches}")
    print(f"  • Mismatches / Errors      : {mismatches}")
    print(f"  • Mathematical Equivalence : {parity_pct:.2f}%")
    print("----------------------------------------------------------------------\n")

    if mismatches == 0:
        print("🎉 SUCCESS: 100% PERFECT PHONEMIZER PARITY ACHIEVED!")
        print("Your Python phonemizer engine generates identical token streams to Deno.\n")
    else:
        print(f"❌ ALERT: {mismatches} PARITY GAP(S) DETECTED!")
        print("Review the mismatched lines below to align regex boundaries or mapping keys:\n")
        for line_num, raw_t, gen_t, exp_t in mismatch_details:
            print(f"[{line_num}] --- Line {line_num} ---")
            print(f"  Raw Text:  {raw_t}")
            print(f"  Generated: {gen_t}")
            print(f"  Expected:  {exp_t}")
            print(f"  Diff Map:  {create_diff_map(gen_t, exp_t)}")
            print("----------------------------------------------------------------------")
        if mismatches > 10:
            print(f"... and {mismatches - 10} more mismatches.\n")

if __name__ == "__main__":
    main()
