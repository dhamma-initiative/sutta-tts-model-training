// analyze_chars.ts
// Run with: deno run --allow-read analyze_chars.ts <filename>

import { exists } from "https://deno.land/std@0.192.0/fs/exists.ts";

const filename = Deno.args[0];

if (!filename) {
  console.error("Usage: deno run --allow-read analyze-sutta-chars <file.txt>");
  Deno.exit(1);
}

if (!await exists(filename)) {
  console.error(`File not found: ${filename}`);
  Deno.exit(1);
}

const text = await Deno.readTextFile(filename);

// Use a Set to store unique characters
const uniqueChars = new Set<string>();
const charCounts = new Map<string, number>();

// Iterate over the string. JavaScript/TypeScript iterates by UTF-16 code units,
// but for most common Pali/Sanskrit chars this works. 
// For full grapheme cluster support, we'd need a library like 'grapheme-splitter',
// but simple iteration usually suffices for identifying the set.
// To be strictly safe with combining marks, we normalize first.
const normalizedText = text.normalize('NFC');

for (const char of normalizedText) {
  if (!uniqueChars.has(char)) {
    uniqueChars.add(char);
    charCounts.set(char, 1);
  } else {
    charCounts.set(char, (charCounts.get(char) || 0) + 1);
  }
}

// Convert to array and sort by code point
const sortedChars = Array.from(uniqueChars).sort((a, b) => a.codePointAt(0)! - b.codePointAt(0)!);

console.log(`\nTotal unique characters found: ${sortedChars.length}\n`);
console.log("Char | Code Point | Name | Category | Count");
console.log("-----|------------|------|----------|------");

for (const char of sortedChars) {
  const code = char.codePointAt(0)!;
  const hex = "U+" + code.toString(16).toUpperCase().padStart(4, '0');
  
  // Determine category
  let category = "Other";
  if (/\s/.test(char)) category = "Whitespace";
  else if (/\p{P}/u.test(char)) category = "Punctuation";
  else if (/\p{L}/u.test(char)) category = "Letter";
  else if (/\p{N}/u.test(char)) category = "Number";
  else if (/\p{C}/u.test(char)) category = "Control/Invisible";

  // Get character name (basic implementation, Deno doesn't have built-in charnames)
  // We will just display the hex for precision
  const name = "See Unicode Table"; 

  // Display representation
  const display = char === ' ' ? 'SPACE' : char === '\n' ? 'LF' : char === '\t' ? 'TAB' : char === '\r' ? 'CR' : char;

  console.log(`${display.padEnd(4)} | ${hex} | ${name.padEnd(15)} | ${category.padEnd(8)} | ${charCounts.get(char)}`);
}

// Specific report on potential problem characters
console.log("\n--- Potential Issues (Whitespace & Punctuation) ---");
const issues = sortedChars.filter(c => /\s/.test(c) || /\p{P}/u.test(c));
if (issues.length === 0) {
  console.log("No special whitespace or punctuation issues detected.");
} else {
  for (const char of issues) {
    const code = char.codePointAt(0)!;
    const hex = "U+" + code.toString(16).toUpperCase().padStart(4, '0');
    const display = char === ' ' ? 'SPACE' : char === '\n' ? 'LF' : char === '\t' ? 'TAB' : char === '\r' ? 'CR' : char;
    console.log(`Found: '${display}' (${hex})`);
  }
}   