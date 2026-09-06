// sutta-training-manager.ts
// Version: 13.0.0
// Tmux-first orchestrator. CPU/GPU aware, configurable batch size, override checkpoint path.
// keep-alive is DEPRECATED here -> use a persistent notebook cell instead.

import { parseArgs } from "jsr:@std/cli/parse-args";
import { ensureDir } from "jsr:@std/fs/ensure-dir";
import { join } from "jsr:@std/path";

const VERSION = "13.0.0";

// === Paths ===
const DRIVE_BASE = "/content/drive/MyDrive/sutta-tts-model-training";
const PIPER_TRAINING = "/content/drive/MyDrive/piper_training";
const LOCAL_CACHE = "/content/piper_cache";
const REPO_DIR = "/content/piper1-gpl";
const METADATA_CSV = join(DRIVE_BASE, "/corpus-preperation/metadata-phonemes.csv");
const PHONEME_MAP = join(DRIVE_BASE, "/config/en[gb]_pi[si]-suttaplayer-phoneme-map.json");
const AUDIO_DIR = join(PIPER_TRAINING, "wavs");
const DEFAULT_BASE_CKPT = join(PIPER_TRAINING, "en_GB-northern_english_male-medium.ckpt");
const LOCAL_LOGS = join(REPO_DIR, "lightning_logs");
const DRIVE_CKPTS = join(PIPER_TRAINING, "checkpoints");

const VOICE_NAME = "en_gb-suttaplayer-medium"; // FIXED per your instruction

// === Configurable defaults ===
const DEFAULT_BATCH_GPU = 8;   // dropped from 32 to avoid OOM on Colab GPU
const DEFAULT_BATCH_CPU = 2;   // safe CPU batch size
const KEEP_TOP_K = 3;
const CHECKPOINT_MONITOR_INTERVAL_MS = 120000;
const KEEP_ALIVE_INTERVAL_MS = 60000; // deprecated

// === Parse CLI arguments ===
const flags = parseArgs(Deno.args, {
  boolean: [
    "init", "train", "train-fg", "monitor", "dry-run", "diag-setup",
    "sync-sheets", "pip-restore", "cache-restore", "cpu", "keepalive",
  ],
  string: ["ckpt-path", "batch-size", "max-epochs"],
  alias: {
    i: "init", t: "train", f: "train-fg", m: "monitor", d: "dry-run",
    g: "diag-setup", s: "sync-sheets", c: "cpu", k: "keepalive",
  },
});

const isDryRun = flags["dry-run"] || false;
const isDiagSetup = flags["diag-setup"] || false;
const isCpu = flags["cpu"] || false;
const isKeepAlive = flags["keepalive"] || false; // deprecated
const ckptPath = flags["ckpt-path"] || DEFAULT_BASE_CKPT;
const maxEpochs = flags["max-epochs"]; // optional override

// batch size: explicit flag > cpu default > gpu default
const batchSize = flags["batch-size"]
  ? parseInt(flags["batch-size"], 10)
  : (isCpu ? DEFAULT_BATCH_CPU : DEFAULT_BATCH_GPU);

console.log(`=================================================`);
console.log(`🎙️ Sutta TTS Training Manager v${VERSION}`);
console.log(`   Mode: ${isCpu ? "CPU" : "GPU"} | Batch: ${batchSize} | ckpt: ${ckptPath}`);
console.log(`=================================================`);

// === Helpers ===
async function runCmd(cmd: string, args: string[], options: { cwd?: string } = {}) {
  if (isDryRun) {
    console.log(`[DRY-RUN] Would run: ${cmd} ${args.join(" ")}`);
    return { code: 0, success: true };
  }
  const command = new Deno.Command(cmd, {
    args, cwd: options.cwd, stdout: "inherit", stderr: "inherit",
  });
  return await command.spawn().status;
}

async function runCmdWithOutput(cmd: string, args: string[], options: { cwd?: string } = {}) {
  const output = await new Deno.Command(cmd, {
    args, cwd: options.cwd, stdout: "piped", stderr: "piped",
  }).output();
  return {
    code: output.code, success: output.success,
    stdout: new TextDecoder().decode(output.stdout),
    stderr: new TextDecoder().decode(output.stderr),
  };
}

async function statPath(path: string) {
  try { return await Deno.stat(path); } catch { return null; }
}

// === Smart checkpoint sync + prune (union of top-K by MOS + top-K by MEL) ===
async function syncAndPruneCheckpoints() {
  console.log("🔄 Syncing and pruning checkpoints (top 3 MOS + top 3 MEL)...");
  if (!isDryRun) {
    await new Deno.Command("rsync", {
      args: ["-av", "--include=*/", "--include=*.ckpt", "--exclude=*",
             `${LOCAL_LOGS}/`, `${DRIVE_CKPTS}/`],
    }).output();
    console.log("  ✅ Local → Drive sync complete.");
  } else {
    console.log(`  [DRY-RUN] Would rsync ${LOCAL_LOGS}/ -> ${DRIVE_CKPTS}/`);
  }
  await pruneCheckpoints(LOCAL_LOGS, KEEP_TOP_K);
  await pruneCheckpoints(DRIVE_CKPTS, KEEP_TOP_K);
}

async function pruneCheckpoints(dir: string, keepTopK: number) {
  if (isDryRun) {
    console.log(`  [DRY-RUN] Would prune ${dir} (keep top ${keepTopK} MOS + ${keepTopK} MEL + last)`);
    return;
  }
  const parsed: Array<{ name: string; mos?: number; mel?: number }> = [];
  try {
    for await (const entry of Deno.readDir(dir)) {
      if (entry.isFile && entry.name.endsWith(".ckpt") && entry.name !== "last.ckpt") {
        const mos = entry.name.match(/val_mos=([\d.]+)/);
        const mel = entry.name.match(/val_mel=([\d.]+)/);
        parsed.push({
          name: entry.name,
          mos: mos ? parseFloat(mos[1]) : undefined,
          mel: mel ? parseFloat(mel[1]) : undefined,
        });
      }
    }
  } catch { return; } // dir may not exist yet

  const mosRanked = parsed.filter(p => p.mos !== undefined).sort((a, b) => b.mos! - a.mos!);
  const melRanked = parsed.filter(p => p.mel !== undefined).sort((a, b) => a.mel! - b.mel!);

  const keep = new Set<string>(["last.ckpt"]);
  mosRanked.slice(0, keepTopK).forEach(p => keep.add(p.name));
  melRanked.slice(0, keepTopK).forEach(p => keep.add(p.name));

  for (const p of parsed.filter(p => !keep.has(p.name))) {
    try {
      await Deno.remove(join(dir, p.name));
      console.log(`  🗑️ Pruned: ${p.name}`);
    } catch (err) {
      console.error(`  ⚠️ Failed to prune ${p.name}: ${err.message}`);
    }
  }
}

// === DEPRECATED keep-alive (use notebook cell instead) ===
function startKeepAliveLoop() {
  console.log("⚠️  DEPRECATED: keep-alive in Deno does NOT reset Colab's idle timer.");
  console.log("   Use a persistent notebook cell instead. Continuing for legacy parity...");
  if (isDryRun) return;
  let ticks = 0;
  setInterval(async () => {
    ticks++;
    const minutes = (ticks * KEEP_ALIVE_INTERVAL_MS) / 60000;
    console.log(`PING [${new Date().toLocaleTimeString()}] (${minutes.toFixed(1)} min)`);
  }, KEEP_ALIVE_INTERVAL_MS);
}

// === Training launcher ===
function getTrainingScriptContent(): string {
  const accel = isCpu
    ? "--trainer.accelerator cpu --trainer.devices 1 --trainer.precision 32"
    : "--trainer.accelerator gpu --trainer.devices 1 --trainer.precision 16-mixed";
  const epochsArg = maxEpochs ? ` \\\n  --trainer.max_epochs ${maxEpochs}` : "";

  return `#!/bin/bash
cd ${REPO_DIR}
PYTHONPATH=${REPO_DIR}:$PYTHONPATH \\
python3 -m piper.train fit \\
  --data.voice_name "${VOICE_NAME}" \\
  --data.csv_path "${METADATA_CSV}" \\
  --data.phoneme_type text \\
  --data.phonemes_path "${PHONEME_MAP}" \\
  --data.audio_dir "${AUDIO_DIR}" \\
  --model.sample_rate 22050 \\
  --data.espeak_voice "en-gb" \\
  --data.cache_dir "${LOCAL_CACHE}/cache" \\
  --data.config_path "${LOCAL_CACHE}/config.json" \\
  --data.batch_size ${batchSize} \\
  ${accel} \\
  --trainer.callbacks.class_path "train_sutta_voice.SuttaVoiceUatCallback" \\
  --ckpt_path "${ckptPath}" \\
  --model.mel_fmin 0 \\
  --model.mel_fmax 8000${epochsArg}
`;
}

async function startTrainingSession(tmuxMode = true) {
  const scriptPath = "/content/run_training.sh";
  const trainingCmdScript = getTrainingScriptContent();

  if (isDryRun) {
    console.log(`\n[DRY-RUN] Would write ${scriptPath}:`);
    console.log("----");
    console.log(trainingCmdScript);
    console.log("----");
    return;
  }

  await ensureDir(LOCAL_CACHE);
  await Deno.writeTextFile(scriptPath, trainingCmdScript);
  await new Deno.Command("chmod", { args: ["+x", scriptPath] }).output();

  if (tmuxMode) {
    const out = await new Deno.Command("tmux", { args: ["ls"] }).output();
    const list = new TextDecoder().decode(out.stdout);
    if (!list.includes("piper_train")) {
      // Log training stdout to a file so the notebook can tail it
      await new Deno.Command("tmux", {
        args: ["new-session", "-d", "-s", "piper_train",
               `bash ${scriptPath} 2>&1 | tee /content/train.log`],
      }).output();
      console.log("🎉 Training launched in tmux 'piper_train' (logging to /content/train.log).");
    } else {
      console.log("⚠️ tmux 'piper_train' already running. Skipping launch.");
    }
  } else {
    console.log("🔥 Launching training in foreground...");
    await runCmd("bash", [scriptPath]);
  }
}

// === Restores ===
async function pipRestore() {
  const req = "/content/drive/MyDrive/piper_env_requirements.txt";
  const json = "/content/drive/MyDrive/piper_env_pip_list.json";

  // Check if JSON exists (the "Exact" backup)
  if (await statPath(json)) {
    console.log("📦 Restoring Pip Environment (EXACT MODE from JSON)...");
    if (!isDryRun) {
      // 1. Uninstall everything first to avoid conflicts
      console.log("  🗑️  Flushing existing packages...");
      await runCmd("pip", ["uninstall", "-y", "-q", "pip", "setuptools", "wheel"]); // Basic cleanup
      // Note: We can't easily uninstall *all* packages without a list, 
      // so we will just force-reinstall the exact list.
      
      // 2. Force install exact versions from JSON
      // This command parses the JSON and installs each package with == and --no-deps
      // to prevent pip from trying to upgrade/downgrade dependencies.
      const cmd = `python3 -c "
import json
import subprocess
import sys

with open('${json}', 'r') as f:
    packages = json.load(f)

for pkg in packages:
    name = pkg['name']
    version = pkg['version']
    # Force install exact version, ignore dependencies to prevent conflicts
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', f'{name}=={version}'])
    print(f'  ✅ Installed {{name}}=={{version}}')
"`;
      await runCmd("bash", ["-c", cmd]);
      console.log("✅ Exact environment restored from JSON.");
    } else {
      console.log(`[DRY-RUN] Would parse ${json} and force-install exact versions.`);
    }
  } else if (await statPath(req)) {
    console.log("⚠️  JSON backup not found. Falling back to standard requirements.txt...");
    console.log("   Note: This may not match your original environment exactly.");
    if (!isDryRun) { await runCmd("pip", ["install", "-r", req]); }
    else { console.log(`[DRY-RUN] pip install -r ${req}`); }
  } else {
    console.error("❌ Missing both JSON and requirements files. Cannot restore.");
  }
}

async function piperCacheRestore() {
  const archive = "/content/drive/MyDrive/piper_cache.tgz";
  if (!await statPath(archive)) { console.error("❌ Missing:", archive); return; }
  console.log("📦 Restoring piper_cache...");
  if (!isDryRun) {
    await ensureDir(LOCAL_CACHE);
    await runCmd("tar", ["-xzf", archive, "-C", "/content"]);
  } else {
    console.log(`[DRY-RUN] tar -xzf ${archive} -C /content`);
  }
}

// === Diagnostics ===
async function runDiagSetup() {
  console.log("=== ENVIRONMENT DIAGNOSTIC ===");
  const paths = [
    { l: "Drive Base", p: DRIVE_BASE, r: true },
    { l: "Local Cache", p: LOCAL_CACHE, r: true },
    { l: "Metadata CSV", p: METADATA_CSV, r: true },
    { l: "WAVs Dir", p: AUDIO_DIR, r: true },
    { l: "Phoneme Map", p: PHONEME_MAP, r: true },
    { l: "Base ckpt", p: ckptPath, r: true },
  ];
  for (const p of paths) {
    const info = await statPath(p.p);
    console.log(info ? `  [OK]   ${p.l.padEnd(15)} ${p.p}`
                     : `  [FAIL] ${p.l.padEnd(15)} ${p.r ? "REQUIRED" : "optional"}: ${p.p}`);
  }
  console.log("\nBinaries:");
  for (const b of ["tmux", "rsync", "git", "python3", "deno"]) {
    const { code, stdout } = await runCmdWithOutput("which", [b]);
    console.log(code === 0 ? `  [OK]   ${b.padEnd(8)} ${stdout.trim()}`
                           : `  [FAIL] ${b.padEnd(8)} missing`);
  }
  const out = await new Deno.Command("tmux", { args: ["ls"] }).output();
  const list = new TextDecoder().decode(out.stdout);
  console.log(`\ntmux: ${list.includes("piper_train") ? "piper_train RUNNING" : "no piper_train session"}`);
}

// === Pipeline ===
async function runPipeline() {
  if (isDiagSetup) return runDiagSetup();
  if (flags["train"]) return startTrainingSession(true);
  if (flags["train-fg"]) return startTrainingSession(false);
  if (flags.monitor) { await syncAndPruneCheckpoints(); if (isKeepAlive) startKeepAliveLoop(); return; }
  if (flags["pip-restore"]) return pipRestore();
  if (flags["cache-restore"]) return piperCacheRestore();
  if (flags["sync-sheets"]) { console.log("Run the Sheets sync Python cell in the notebook."); return; }

  console.log(`SuttaPlayer Training Manager v${VERSION}`);
  console.log("Usage: deno run --allow-all sutta-training-manager.ts \\");
  console.log("  [--init|--train|--train-fg|--monitor|--pip-restore|--cache-restore|--sync-sheets] \\");
  console.log("  [--cpu] [--batch-size N] [--ckpt-path PATH] [--max-epochs N] [--dry-run] [--diag-setup]");
}

runPipeline().catch((err) => {
  console.error("FATAL ERROR running automated manager:", err.message);
  Deno.exit(1);
});