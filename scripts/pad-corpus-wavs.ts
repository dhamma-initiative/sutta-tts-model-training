// pad-corpus-wavs.ts
// Automated WAV Silence Padding & Punctuation Alignment Tool for Deno
// Formats 16-bit PCM Mono WAV files to contain exact target millisecond silences at punctuation boundaries.

import { parseArgs } from "jsr:@std/cli/parse-args";
import { ensureDir } from "jsr:@std/fs/ensure-dir";
import { join, dirname } from "jsr:@std/path";

// Silence duration mapping for punctuation targets (in ms)
const PUNCTUATION_PAUSES: Record<string, number> = {
  ",": 250,
  ";": 350,
  ":": 400,
  "—": 450,
  "–": 300,
  "…": 600,
  ".": 500,
  "!": 500,
  "?": 500,
  "•": 420,
};

const BRACKET_PAUSES: Record<string, { leadMs: number; trailMs: number }> = {
  "[": { leadMs: 220, trailMs: 0 },
  "]": { leadMs: 0, trailMs: 420 },
  "(": { leadMs: 220, trailMs: 0 },
  ")": { leadMs: 0, trailMs: 280 },
  "{": { leadMs: 220, trailMs: 0 },
  "}": { leadMs: 0, trailMs: 420 },
};

interface TextClause {
  text: string;
  leadMs: number;
  trailMs: number;
}

function parseTextIntoClauses(text: string): TextClause[] {
  const regex = /([{}[](),;:—-.…!?•])|([^{}[](),;:—-.…!?•]+)/g;
  const tokens: string[] = [];
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match[0].trim().length > 0) {
      tokens.push(match[0]);
    }
  }

  const clauses: TextClause[] = [];
  let currentText = "";
  let currentLeadMs = 0;

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];

    if (BRACKET_PAUSES[token] && BRACKET_PAUSES[token].leadMs > 0) {
      if (currentText.trim().length > 0) {
        clauses.push({ text: currentText.trim(), leadMs: currentLeadMs, trailMs: 0 });
        currentText = "";
      }
      currentLeadMs = BRACKET_PAUSES[token].leadMs;
      currentText += token;
    } else if (BRACKET_PAUSES[token] && BRACKET_PAUSES[token].trailMs > 0) {
      currentText += token;
      clauses.push({ text: currentText.trim(), leadMs: currentLeadMs, trailMs: BRACKET_PAUSES[token].trailMs });
      currentText = "";
      currentLeadMs = 0;
    } else if (PUNCTUATION_PAUSES[token]) {
      currentText += token;
      clauses.push({ text: currentText.trim(), leadMs: currentLeadMs, trailMs: PUNCTUATION_PAUSES[token] });
      currentText = "";
      currentLeadMs = 0;
    } else {
      currentText += token;
    }
  }

  if (currentText.trim().length > 0) {
    clauses.push({ text: currentText.trim(), leadMs: currentLeadMs, trailMs: 0 });
  }

  return clauses;
}

function createSilenceBuffer(sampleRate: number, durationMs: number): Float32Array {
  const numSamples = Math.floor((sampleRate * durationMs) / 1000);
  return new Float32Array(numSamples);
}

function pcm16ToFloat32(int16Array: Int16Array): Float32Array {
  const float32 = new Float32Array(int16Array.length);
  for (let i = 0; i < int16Array.length; i++) {
    float32[i] = int16Array[i] / (int16Array[i] < 0 ? 0x8000 : 0x7fff);
  }
  return float32;
}

function floatTo16BitPCM(float32Array: Float32Array): Int16Array {
  const int16 = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

function createWavHeader(sampleRate: number, dataByteLength: number): Uint8Array {
  const buffer = new ArrayBuffer(44);
  const view = new DataView(buffer);
  view.setUint8(0, 0x52); view.setUint8(1, 0x49); view.setUint8(2, 0x46); view.setUint8(3, 0x46); // RIFF
  view.setUint32(4, 36 + dataByteLength, true);
  view.setUint8(8, 0x57); view.setUint8(9, 0x41); view.setUint8(10, 0x56); view.setUint8(11, 0x45); // WAVE
  view.setUint8(12, 0x66); view.setUint8(13, 0x6d); view.setUint8(14, 0x74); view.setUint8(15, 0x20); // fmt
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  view.setUint8(36, 0x64); view.setUint8(37, 0x61); view.setUint8(38, 0x74); view.setUint8(39, 0x61); // data
  view.setUint32(40, dataByteLength, true);
  return new Uint8Array(buffer);
}

// Locate silence intervals in Float32 audio using sliding RMS
function findSpokenSegments(
  samples: Float32Array,
  sampleRate: number,
  thresholdDb = -32,
  minSilenceMs = 180
): Array<{ start: number; end: number }> {
  const hopSize = 128;
  const numHops = Math.floor(samples.length / hopSize);
  const isSilent: boolean[] = new Array(numHops);

  for (let h = 0; h < numHops; h++) {
    let sumSq = 0;
    for (let i = 0; i < hopSize; i++) {
      const s = samples[h * hopSize + i];
      sumSq += s * s;
    }
    const rms = Math.sqrt(sumSq / hopSize);
    const db = 20 * Math.log10(rms + 1e-6);
    isSilent[h] = db < thresholdDb;
  }

  const minSilenceHops = Math.floor((sampleRate * (minSilenceMs / 1000)) / hopSize);
  const spoken: Array<{ start: number; end: number }> = [];

  let inSpeech = false;
  let speechStartHop = 0;
  let silenceHopCount = 0;

  for (let h = 0; h < numHops; h++) {
    if (!isSilent[h]) {
      if (!inSpeech) {
        inSpeech = true;
        speechStartHop = Math.max(0, h - 2); // 15ms lead cushion
      }
      silenceHopCount = 0;
    } else {
      if (inSpeech) {
        silenceHopCount++;
        if (silenceHopCount >= minSilenceHops) {
          inSpeech = false;
          const speechEndHop = Math.min(numHops, h - silenceHopCount + 2); // 15ms trail cushion
          spoken.push({
            start: speechStartHop * hopSize,
            end: speechEndHop * hopSize,
          });
        }
      }
    }
  }

  if (inSpeech) {
    spoken.push({
      start: speechStartHop * hopSize,
      end: samples.length,
    });
  }

  return spoken;
}

async function processWavFile(
  inputWavPath: string,
  outputWavPath: string,
  text: string,
  sampleRate = 22050
) {
  const bytes = await Deno.readFile(inputWavPath);
  // Extract 16-bit PCM payload (skip 44-byte header if present)
  const headerOffset = bytes.length > 44 ? 44 : 0;
  const pcm16 = new Int16Array(bytes.buffer, bytes.byteOffset + headerOffset, Math.floor((bytes.byteLength - headerOffset) / 2));
  const samples = pcm16ToFloat32(pcm16);

  const clauses = parseTextIntoClauses(text);
  const spoken = findSpokenSegments(samples, sampleRate);

  const audioChunks: Float32Array[] = [];
  let totalSamples = 0;

  // Leading anchor silence
  const leadCushion = createSilenceBuffer(sampleRate, 100);
  audioChunks.push(leadCushion);
  totalSamples += leadCushion.length;

  for (let i = 0; i < clauses.length; i++) {
    const clause = clauses[i];

    if (clause.leadMs > 0) {
      const leadSil = createSilenceBuffer(sampleRate, clause.leadMs);
      audioChunks.push(leadSil);
      totalSamples += leadSil.length;
    }

    // Match clause to corresponding spoken audio segment
    if (i < spoken.length) {
      const seg = spoken[i];
      const chunk = samples.subarray(seg.start, seg.end);
      audioChunks.push(chunk);
      totalSamples += chunk.length;
    } else if (spoken.length > 0) {
      // Fallback: use last spoken chunk if mismatch
      const seg = spoken[spoken.length - 1];
      const chunk = samples.subarray(seg.start, seg.end);
      audioChunks.push(chunk);
      totalSamples += chunk.length;
    }

    if (clause.trailMs > 0) {
      const trailSil = createSilenceBuffer(sampleRate, clause.trailMs);
      audioChunks.push(trailSil);
      totalSamples += trailSil.length;
    }
  }

  // Trailing anchor silence
  const trailCushion = createSilenceBuffer(sampleRate, 100);
  audioChunks.push(trailCushion);
  totalSamples += trailCushion.length;

  const merged = new Float32Array(totalSamples);
  let offset = 0;
  for (const chunk of audioChunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  const outPcm16 = floatTo16BitPCM(merged);
  const outBytes = new Uint8Array(outPcm16.buffer, outPcm16.byteOffset, outPcm16.byteLength);
  const header = createWavHeader(sampleRate, outBytes.byteLength);

  await ensureDir(dirname(outputWavPath));
  const outFile = await Deno.open(outputWavPath, { write: true, create: true, truncate: true });
  await outFile.write(header);
  await outFile.write(outBytes);
  outFile.close();
}

async function main() {
  const flags = parseArgs(Deno.args, {
    string: ["input", "output", "csv"],
    alias: { i: "input", o: "output", c: "csv" },
  });

  if (!flags.input || !flags.output || !flags.csv) {
    console.log("Usage: deno run --allow-all pad-corpus-wavs.ts -i ./wavs -o ./wavs_padded -c metadata-text.csv");
    Deno.exit(1);
  }

  const csvContent = await Deno.readTextFile(flags.csv);
  const lines = csvContent.split(/\r?\n/).filter((l) => l.trim().length > 0);

  console.log(`\n🚀 Starting Automated WAV Silence Padding...`);
  console.log(`Input WAVs:  ${flags.input}`);
  console.log(`Output WAVs: ${flags.output}`);
  console.log(`Corpus CSV:  ${flags.csv}`);
  console.log(`Total Files: ${lines.length}\n`);

  let count = 0;
  for (const line of lines) {
    const parts = line.split("|");
    if (parts.length < 2) continue;

    let fileId = parts[0].trim();
    if (!fileId.endsWith(".wav")) fileId += ".wav";
    const text = parts[1].trim();

    const inPath = join(flags.input, fileId);
    const outPath = join(flags.output, fileId);

    try {
      await processWavFile(inPath, outPath, text);
      count++;
      if (count % 100 === 0 || count === lines.length) {
        console.log(`  ✅ Processed [${count}/${lines.length}] WAV files`);
      }
    } catch (err) {
      console.error(`  ❌ Failed for ${fileId}: ${(err as Error).message}`);
    }
  }

  console.log(`\n🎉 Complete! Successfully padded ${count} WAV files to ${flags.output}\n`);
}

if (import.meta.main) {
  main();
}
