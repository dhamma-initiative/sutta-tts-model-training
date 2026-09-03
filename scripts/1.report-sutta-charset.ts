// scripts/1.report-sutta-charset.ts
import { expandGlob } from "jsr:@std/fs/expand-glob";
import { join, basename } from "jsr:@std/path";
import { parseArgs } from "jsr:@std/cli/parse-args";

interface CharInfo {
  char: string;
  codePoint: string;
  count: number;
  categories: string[];
}

function getCharCategory(char: string): string {
  const code = char.charCodeAt(0);
  if (code >= 0x0000 && code <= 0x001F) return "Control Character";
  if (char === " " || char === "\r" || char === "\n" || char === "\t") return "Whitespace";
  if (/[0-9]/.test(char)) return "Numeric";
  if (/[\p{P}]/u.test(char)) return "Punctuation";
  if (/[\p{S}]/u.test(char)) return "Symbol";
  if (/[\p{L}]/u.test(char)) return "Letter";
  return "Other";
}

async function main() {
  const flags = parseArgs(Deno.args, {
    string: ["input", "output", "json"],
    alias: { i: "input", o: "output", j: "json" },
    default: {
      input: "text/sutta-books",
      output: "config/ALL-suttas-text-characterset-report.md",
      json: "config/ALL-suttas-text-characterset-report.json"
    }
  });

  const inputDir = flags.input;
  const mdPath = flags.output;
  const jsonPath = flags.json;

  console.log(`📊 Scanning Sutta Sourced Directory: ${inputDir}`);
  const charMap = new Map<string, number>();
  let totalChars = 0;
  let fileCount = 0;

  for await (const file of expandGlob(join(inputDir, "*.txt"))) {
    fileCount++;
    console.log(`  -> Reading file: ${basename(file.path)}`);
    const text = await Deno.readTextFile(file.path);
    const normalized = text.normalize("NFC");
    for (const char of normalized) {
      charMap.set(char, (charMap.get(char) || 0) + 1);
      totalChars++;
    }
  }

  if (fileCount === 0) {
    console.error(`❌ Error: No text files found in ${inputDir}`);
    Deno.exit(1);
  }

  // Compile individual character profiles
  const characters: CharInfo[] = [];
  for (const [char, count] of charMap.entries()) {
    const codePoint = `U+${char.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")}`;
    const category = getCharCategory(char);
    characters.push({
      char,
      codePoint,
      count,
      categories: [category]
    });
  }

  // Sort by count descending
  characters.sort((a, b) => b.count - a.count);

  // Separate into Logical Subgroups for MD Report
  const letters = characters.filter(c => c.categories.includes("Letter"));
  const punctuation = characters.filter(c => c.categories.includes("Punctuation") || c.categories.includes("Symbol"));
  const whitespace = characters.filter(c => c.categories.includes("Whitespace") || c.categories.includes("Control Character"));

  // 1. Write structured JSON Manifest
  const jsonReport = {
    metadata: {
      total_characters_scanned: totalChars,
      unique_character_count: characters.length,
      scanned_directory: inputDir,
      timestamp: new Date().toISOString()
    },
    characters: characters.map(c => ({
      char: c.char,
      code_point: c.codePoint,
      count: c.count,
      category: c.categories[0]
    }))
  };

  await Deno.mkdir(join(jsonPath, ".."), { recursive: true }).catch(() => {});
  await Deno.writeTextFile(jsonPath, JSON.stringify(jsonReport, null, 2));
  console.log(`✅ Structured JSON Character Set Manifest written to: ${jsonPath}`);

  // 2. Write Beautiful Human-Readable MD Report
  const mdLines: string[] = [
    `# Sutta Text Corpus Character Set Audit Report`,
    `Generated on: ${new Date().toLocaleDateString()} | Scanned Directory: \`${inputDir}\``,
    `\n## Executive Summary`,
    `*   **Total Characters Scanned:** ${totalChars.toLocaleString()}`,
    `*   **Unique Symbols Found:** ${characters.length}`,
    `*   **Structured Metadata Source:** \`${basename(jsonPath)}\``,
    `\n## 1. Letters & Diacritics (Count: ${letters.length})`,
    `| Character | Unicode Code Point | Occurrence Count |`,
    `| :---: | :--- | :--- |`
  ];

  for (const c of letters) {
    mdLines.push(`| **${c.char}** | \`${c.codePoint}\` | ${c.count.toLocaleString()} |`);
  }

  mdLines.push(
    `\n## 2. Persistent Punctuation & Boundary Symbols (Count: ${punctuation.length})`,
    `*These markers must have dedicated representations in the SuttaPlayer phoneme map to ensure appropriate pauses.*`,
    `\n| Character | Unicode Code Point | Occurrence Count | Description / Role |`,
    `| :---: | :--- | :--- | :--- |`
  );

  for (const c of punctuation) {
    let desc = "Punctuation Mark";
    if (c.char === "…") desc = "Ellipsis (Long Pause)";
    else if (c.char === "—" || c.char === "–") desc = "Em/En Dash (Boundary Pause)";
    else if (c.char === "[" || c.char === "]") desc = "Brackets (Acoustic Side-Pause)";
    else if (c.char === "•") desc = "Bullet Point (Segment Marker)";
    mdLines.push(`| **${c.char}** | \`${c.codePoint}\` | ${c.count.toLocaleString()} | ${desc} |`);
  }

  mdLines.push(
    `\n## 3. Whitespace & Control Characters (Count: ${whitespace.length})`,
    `| Character Escape | Unicode Code Point | Occurrence Count | Category |`,
    `| :---: | :--- | :--- | :--- |`
  );

  for (const c of whitespace) {
    const displayChar = c.char === " " ? "SPACE" : c.char === "\n" ? "LF (\\n)" : c.char === "\r" ? "CR (\\r)" : "CONTROL";
    mdLines.push(`| \`${displayChar}\` | \`${c.codePoint}\` | ${c.count.toLocaleString()} | ${c.categories[0]} |`);
  }

  await Deno.mkdir(join(mdPath, ".."), { recursive: true }).catch(() => {});
  await Deno.writeTextFile(mdPath, mdLines.join("\n"));
  console.log(`✅ Human-readable Markdown Audit Report written to: ${mdPath}`);
}

if (import.meta.main) {
  main();
}
