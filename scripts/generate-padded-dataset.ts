import { parseArgs } from "jsr:@std/cli/parse-args";
import { TtsReaderPhonemizer } from "./tts-reader-phonemizer.ts";

interface TTSConfig {
  audio: { sample_rate: number };
  phoneme_id_map: Record<string, number[]>;
}

// Silence duration mapping for punctuation targets (in ms)
const PUNCTUATION_PAUSES: Record<string, number> = {
  ",": 250,
  ";": 350,
  ":": 400,
  "—": 450,
  "–": 300,
//   "-": 250,
  "…": 600,
  ".": 500,
  "!": 500,
  "?": 500,
  "•": 420,
};

// Lead and Trail pause rules for brackets, parentheses, and braces (probe_punct_06, 08, 09)
const BRACKET_PAUSES: Record<string, { leadMs: number; trailMs: number }> = {
  "[": { leadMs: 220, trailMs: 0 },
  "]": { leadMs: 0, trailMs: 420 },
  "(": { leadMs: 220, trailMs: 0 },
  ")": { leadMs: 0, trailMs: 280 },
  "{": { leadMs: 220, trailMs: 0 },
  "}": { leadMs: 0, trailMs: 420 },
};

function createSilenceBuffer(sampleRate: number, durationMs: number): Float32Array {
  const numSamples = Math.floor((sampleRate * durationMs) / 1000);
  return new Float32Array(numSamples);
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
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  view.setUint8(36, 0x64); view.setUint8(37, 0x61); view.setUint8(38, 0x74); view.setUint8(39, 0x61); // data
  view.setUint32(40, dataByteLength, true);
  return new Uint8Array(buffer);
}

interface Segment {
  text: string;
  leadSilenceMs: number;
  trailSilenceMs: number;
}

function tokenizeTextWithPauses(text: string): Segment[] {
  // Regex matches:
  // Group 1: Delimiters (brackets, parentheses, braces, punctuation marks)
  // Group 2: Non-delimiter text (phoneme sequences)
  const regex = /([{}[\](),;:—\-\.…!?•])|([^{}[\](),;:—\-\.…!?•]+)/g;
  
  const tokens: string[] = [];
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match[0].trim().length > 0) {
      tokens.push(match[0]);
    }
  }

  const segments: Segment[] = [];
  let currentText = "";
  let currentLeadMs = 0;

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];

    // Case 1: Opening bracket/parenthesis/brace (triggers LEAD pause before the inner text)
    if (BRACKET_PAUSES[token] && BRACKET_PAUSES[token].leadMs > 0) {
      if (currentText.trim().length > 0) {
        segments.push({
          text: currentText.trim(),
          leadSilenceMs: currentLeadMs,
          trailSilenceMs: 0,
        });
        currentText = "";
      }
      currentLeadMs = BRACKET_PAUSES[token].leadMs;
      currentText += token;
    } 
    // Case 2: Closing bracket/parenthesis/brace (triggers TRAIL pause after bracket closes)
    else if (BRACKET_PAUSES[token] && BRACKET_PAUSES[token].trailMs > 0) {
      currentText += token;
      segments.push({
        text: currentText.trim(),
        leadSilenceMs: currentLeadMs,
        trailSilenceMs: BRACKET_PAUSES[token].trailMs,
      });
      currentText = "";
      currentLeadMs = 0;
    } 
    // Case 3: Punctuation mark (commas, colons, em-dashes, ellipses)
    else if (PUNCTUATION_PAUSES[token]) {
      currentText += token;
      const pauseMs = PUNCTUATION_PAUSES[token];
      segments.push({
        text: currentText.trim(),
        leadSilenceMs: currentLeadMs,
        trailSilenceMs: pauseMs,
      });
      currentText = "";
      currentLeadMs = 0;
    } 
    // Case 4: Phoneme text content
    else {
      currentText += token;
    }
  }

  // Push any remaining trailing text segment
  if (currentText.trim().length > 0) {
    segments.push({
      text: currentText.trim(),
      leadSilenceMs: currentLeadMs,
      trailSilenceMs: 0,
    });
  }

  return segments;
}

async function main() {
  const args = parseArgs(Deno.args, {
    string: ["model", "csv", "outDir"],
    alias: { m: "model", c: "csv", o: "outDir" },
    default: { outDir: "padded/wavs" }
  });

  if (!args.model || !args.csv) {
    console.error("Usage: deno run --allow-all generate-padded-dataset.ts -m <model.onnx> -c <metadata.csv> -o <outDir>");
    Deno.exit(1);
  }

  const outDir = args.outDir;
  const configPath = `${args.model}.json`;
  const config: TTSConfig = JSON.parse(await Deno.readTextFile(configPath));
  const phonemizer = new TtsReaderPhonemizer(config.phoneme_id_map, false);

  const workerUrl = new URL("./tts-reader-worker.ts", import.meta.url).href;
  const worker = new Worker(workerUrl, { type: "module" });

  await new Promise<void>((resolve) => {
    worker.onmessage = (e) => {
      if (e.data.type === "READY") resolve();
    };
    worker.postMessage({ type: "INIT", payload: { modelPath: args.model, configData: config } });
  });

  const inferChunk = (id: number, phonemeIds: number[][]): Promise<Float32Array> => {
    return new Promise((resolve) => {
      const handler = (e: MessageEvent) => {
        if (e.data.type === "RESULT" && e.data.id === id) {
          worker.removeEventListener("message", handler);
          resolve(e.data.audioData);
        }
      };
      worker.addEventListener("message", handler);
      worker.postMessage({ type: "INFERENCE", payload: { id, phonemeIds } });
    });
  };

  await Deno.mkdir(outDir, { recursive: true });
  const csvContent = await Deno.readTextFile(args.csv);
  const lines = csvContent.split(/\r?\n/).filter((l) => l.trim().length > 0);

  console.log(`Processing ${lines.length} corpus items for padded audio generation into: ${outDir}`);

  let fileIdx = 0;
  for (const line of lines) {
    fileIdx++;
    const parts = line.split("|");
    const rawFilename = parts.length > 1 ? parts[0] : `${fileIdx - 1}.wav`;
    const text = parts.length > 1 ? parts[1] : line;
    const filename = rawFilename.endsWith(".wav") ? rawFilename : `${rawFilename}.wav`;

    const segments = tokenizeTextWithPauses(text);
    const audioChunks: Float32Array[] = [];
    let totalSamples = 0;

    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];

      if (seg.leadSilenceMs > 0) {
        const leadSilence = createSilenceBuffer(config.audio.sample_rate, seg.leadSilenceMs);
        audioChunks.push(leadSilence);
        totalSamples += leadSilence.length;
      }

      const phonemeIds = phonemizer.convertPhonemesToIds(seg.text);
      const audio = await inferChunk(i, phonemeIds);

      audioChunks.push(audio);
      totalSamples += audio.length;

      if (seg.trailSilenceMs > 0) {
        const trailSilence = createSilenceBuffer(config.audio.sample_rate, seg.trailSilenceMs);
        audioChunks.push(trailSilence);
        totalSamples += trailSilence.length;
      }
    }

    const mergedAudio = new Float32Array(totalSamples);
    let offset = 0;
    for (const chunk of audioChunks) {
      mergedAudio.set(chunk, offset);
      offset += chunk.length;
    }

    const pcm16 = floatTo16BitPCM(mergedAudio);
    const wavBytes = new Uint8Array(pcm16.buffer, pcm16.byteOffset, pcm16.byteLength);
    const wavHeader = createWavHeader(config.audio.sample_rate, wavBytes.byteLength);

    const outPath = `${outDir}/${filename}`;
    const outFile = await Deno.open(outPath, { write: true, create: true, truncate: true });
    await outFile.write(wavHeader);
    await outFile.write(wavBytes);
    outFile.close();

    console.log(`[${fileIdx}/${lines.length}] Generated padded WAV: ${outPath} (${(mergedAudio.length / config.audio.sample_rate).toFixed(2)}s)`);
  }

  worker.terminate();
  console.log("Synthetic padded dataset generation complete!");
}

if (import.meta.main) {
  main().catch((err) => {
    console.error(err);
    Deno.exit(1);
  });
}