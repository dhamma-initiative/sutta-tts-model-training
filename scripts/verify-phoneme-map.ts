// scripts/verify-phoneme-map.ts
import { parseArgs } from "jsr:@std/cli/parse-args";
import { join } from "jsr:@std/path";

interface CharactersetJson {
  metadata: {
    total_characters_scanned: number;
    unique_character_count: number;
  };
  characters: Array<{
    char: string;
    code_point: string;
    count: number;
    category: string;
  }>;
}

async function loadJson<T>(path: string): Promise<T> {
  try {
    const txt = await Deno.readTextFile(path);
    return JSON.parse(txt) as T;
  } catch (err) {
    console.error(`❌ Failed to read or parse JSON file: ${path}`);
    throw err;
  }
}

async function main() {
  const flags = parseArgs(Deno.args, {
    string: ["map", "english", "pali", "json"],
    alias: { p: "map", e: "english", pi: "pali", j: "json" },
    default: {
      map: "config/en[gb]_pi[si]-suttaplayer-phoneme-map.json",
      english: "config/pho_en[gb]-to-espeak-v1.51-ipa.json",
      pali: "config/pho_pi[si]-to-espeak-v1.51-ipa.json",
      json: "config/ALL-suttas-text-characterset-report.json"
    }
  });

  console.log("====================================================");
  console.log("    SUTTAPLAYER PHONEME ID MAP COMPLIANCE AUDITOR   ");
  console.log("====================================================\n");

  // 1. Load resources
  console.log("📂 Loading audit resources...");
  const phonemeMap = await loadJson<Record<string, number>>(flags.map);
  const englishDict = await loadJson<Record<string, string>>(flags.english);
  const paliDict = await loadJson<Record<string, string>>(flags.pali);
  const charsetReport = await loadJson<CharactersetJson>(flags.json);
  console.log(`  ✅ Phoneme ID Map loaded: ${Object.keys(phonemeMap).length} symbols.`);
  console.log(`  ✅ English IPA Dictionary loaded: ${Object.keys(englishDict).length} words.`);
  console.log(`  ✅ Pali IPA Dictionary loaded: ${Object.keys(paliDict).length} terms.`);
  console.log(`  ✅ Characterset Report loaded: ${charsetReport.characters.length} unique characters.\n`);

  // Track compliance flags
  let isCompliant = true;
  const missingEnglishSymbols = new Set<string>();
  const missingPaliSymbols = new Set<string>();
  const missingPunctuation = new Set<string>();

  // 2. Audit English IPA Dictionary Coverage
  console.log("🔍 Auditing English Dictionary IPA symbol coverage...");
  for (const [word, ipa] of Object.entries(englishDict)) {
    const normalizedIpa = ipa.normalize("NFD"); // Decompose combining characters if necessary
    for (const char of normalizedIpa) {
      // Ignore spaces and common boundary controls
      if (char === " " || char === "/" || char === "#") continue;
      
      const nfcChar = char.normalize("NFC");
      if (!(nfcChar in phonemeMap)) {
        isCompliant = false;
        missingEnglishSymbols.add(nfcChar);
      }
    }
  }

  // 3. Audit Pali IPA Dictionary Coverage
  console.log("🔍 Auditing Pali Dictionary IPA symbol coverage...");
  for (const [word, ipa] of Object.entries(paliDict)) {
    const normalizedIpa = ipa.normalize("NFD");
    for (const char of normalizedIpa) {
      if (char === " " || char === "/" || char === "#") continue;
      
      const nfcChar = char.normalize("NFC");
      if (!(nfcChar in phonemeMap)) {
        isCompliant = false;
        missingPaliSymbols.add(nfcChar);
      }
    }
  }

  // 4. Audit Sutta Persistent Punctuation Coverage
  console.log("🔍 Auditing Sutta persistent punctuation coverage...");
  // Extract all characters of category "Punctuation" or "Symbol" that occurred in raw suttas
  const punctuationFromSuttas = charsetReport.characters.filter(
    c => c.category === "Punctuation" || c.category === "Symbol"
  );

  for (const item of punctuationFromSuttas) {
    // Ampersand is explicitly skipped in SuttaPlayer Technical Specifications
    if (item.char === "&") continue;
    
    if (!(item.char in phonemeMap)) {
      isCompliant = false;
      missingPunctuation.add(item.char);
    }
  }

  // 5. Final Report & Mitigation Generator
  if (isCompliant) {
    console.log("====================================================");
    console.log("🎉 SUCCESS: YOUR PHONEME MAP IS 100% COMPLIANT!");
    console.log("====================================================");
    console.log("  No missing IPA symbols or punctuation tokens detected.");
    console.log("  The training config is secured against [UNK] token crashes.");
    console.log("====================================================");
    Deno.exit(0);
  } else {
    console.error("====================================================");
    console.error("🔥 COMPLIANCE FAILURE: PHONEME MAP IS NOT COMPLIANT!");
    console.error("====================================================\n");

    if (missingEnglishSymbols.size > 0) {
      console.error(`❌ Missing English IPA Symbols (${missingEnglishSymbols.size}):`);
      for (const char of missingEnglishSymbols) {
        console.error(`  - '${char}' (Unicode: U+${char.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")})`);
      }
      console.error("");
    }

    if (missingPaliSymbols.size > 0) {
      console.error(`❌ Missing Pali IPA Symbols (${missingPaliSymbols.size}):`);
      for (const char of missingPaliSymbols) {
        console.error(`  - '${char}' (Unicode: U+${char.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")})`);
      }
      console.error("");
    }

    if (missingPunctuation.size > 0) {
      console.error(`❌ Missing Persistent Sutta Punctuation (${missingPunctuation.size}):`);
      for (const char of missingPunctuation) {
        console.error(`  - '${char}' (Unicode: U+${char.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")})`);
      }
      console.error("");
    }

    // Determine highest existing token index
    let maxId = 0;
    for (const id of Object.values(phonemeMap)) {
      if (id > maxId) maxId = id;
    }

    // Compile combined missing characters
    const allMissing = new Set([...missingEnglishSymbols, ...missingPaliSymbols, ...missingPunctuation]);
    const sortedMissing = Array.from(allMissing).sort();

    console.error("----------------------------------------------------");
    console.error("💡 STRATEGIC MITIGATION ACTION PLAN:");
    console.error("----------------------------------------------------");
    console.error(`1. Your current phoneme map has a maximum index of ${maxId}.`);
    console.error(`2. To prevent training crashes, append these missing symbols`);
    console.error(`   to 'config/en[gb]_pi[si]-suttaplayer-phoneme-map.json' starting from ID ${maxId + 1}:\n`);
    
    console.error("Copy and paste this JSON block before the closing brace in your map:");
    
    const patchObj: Record<string, number> = {};
    let nextId = maxId + 1;
    for (const char of sortedMissing) {
      patchObj[char] = nextId++;
    }
    
    const patchJson = JSON.stringify(patchObj, null, 2)
      .trim()
      .replace(/^\{/, "")
      .replace(/\}$/, "")
      .trim();
      
    console.error(`, \n${patchJson}`);
    console.error("----------------------------------------------------\n");
    Deno.exit(1);
  }
}

if (import.meta.main) {
  main();
}
