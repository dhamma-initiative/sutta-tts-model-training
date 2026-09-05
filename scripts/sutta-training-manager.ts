// sutta-training-manager.ts
// Version: 12.0.0
// Tmux-first orchestrator with smart checkpoint sync, pip_restore, piper_cache_restore, Sheets sync helper.

import { parseArgs } from "jsr:@std/cli/parse-args";
import { ensureDir } from "jsr:@std/fs/ensure-dir";
import { join } from "jsr:@std/path";

const VERSION = "12.0.0";

// === Standard Path Specifications ===
const DRIVE_BASE = "/content/drive/MyDrive/sutta-tts-model-training";
const PIPER_TRAINING = "/content/drive/MyDrive/piper_training";
const LOCAL_CACHE = "/content/piper_cache";
const REPO_DIR = "/content/piper1-gpl";
const METADATA_CSV = join(DRIVE_BASE, "/corpus-preperation/metadata-phonemes.csv");
const PHONEME_MAP = join(DRIVE_BASE, "/config/en[gb]_pi[si]-suttaplayer-phoneme-map.json");
const AUDIO_DIR = join(PIPER_TRAINING, "wavs");
const BASE_CKPT = join(PIPER_TRAINING, "en_GB-northern_english_male-medium.ckpt");
const LOCAL_LOGS = join(REPO_DIR, "lightning_logs");
const DRIVE_CKPTS = join(PIPER_TRAINING, "checkpoints");
const KEEP_ALIVE_INTERVAL_MS = 60000;
const CHECKPOINT_MONITOR_INTERVAL_MS = 120000;
const KEEP_TOP_K = 3;

// === Parse CLI arguments ===
const args = Deno.args;
const flags = parseArgs(args, {
  boolean: [
    "init",
    "train",
    "train-fg",
    "monitor",
    "dry-run",
    "diag-setup",
    "sync-sheets",
    "pip-restore",
    "cache-restore"
  ],
  alias: {
    i: "init",
    t: "train",
    f: "train-fg",
    m: "monitor",
    d: "dry-run",
    g: "diag-setup",
    s: "sync-sheets",
    pr: "pip-restore",
    cr: "cache-restore",
  },
});

const isDryRun = flags["dry-run"] || false;
const isDiagSetup = flags["diag-setup"] || false;
const isSyncSheets = flags["sync-sheets"] || false;
const isPipRestore = flags["pip-restore"] || false;
const isCacheRestore = flags["cache-restore"] || false;

console.log(`=================================================`);
console.log(`🎙️ Sutta TTS Training Manager v${VERSION}`);
console.log(`=================================================`);

// === Write execution manifest ===
async function writeExecutionManifest() {
  const manifestPath = "/content/sutta_training_manifest.json";
  const manifest = {
    version: VERSION,
    timestamp: new Date().toISOString(),
    drive_base: DRIVE_BASE,
    local_cache: LOCAL_CACHE,
    status: "active"
  };
  try {
    if (!isDryRun) {
      await Deno.writeTextFile(manifestPath, JSON.stringify(manifest, null, 2));
    }
  } catch (err) {
    console.error("⚠️ Failed to write execution manifest:", err.message);
  }
}

// === Run helpers ===
async function runCmd(cmd: string, args: string[], options: { cwd?: string } = {}) {
  if (isDryRun) {
    console.log(`[DRY-RUN] Would run: ${cmd} ${args.join(" ")}`);
    return { code: 0, success: true };
  }
  const command = new Deno.Command(cmd, {
    args,
    cwd: options.cwd,
    stdout: "inherit",
    stderr: "inherit",
  });
  const process = command.spawn();
  const status = await process.status;
  return status;
}

async function runCmdWithOutput(cmd: string, args: string[], options: { cwd?: string } = {}) {
  const command = new Deno.Command(cmd, {
    args,
    cwd: options.cwd,
    stdout: "piped",
    stderr: "piped",
  });
  const output = await command.output();
  const stdout = new TextDecoder().decode(output.stdout);
  const stderr = new TextDecoder().decode(output.stderr);
  return { code: output.code, success: output.success, stdout, stderr };
}

async function statPath(path: string) {
  try {
    return await Deno.stat(path);
  } catch {
    return null;
  }
}

// === Smart checkpoint sync + pruning (keep top K by MOS + top K by MEL) ===
async function syncAndPruneCheckpoints() {
  console.log("🔄 Syncing and pruning checkpoints (top 3 MOS + top 3 MEL)...");

  // 1. Sync local to Drive
  if (!isDryRun) {
    const syncCmd = new Deno.Command("rsync", {
      args: [
        "-av",
        "--include=*/",
        "--include=*.ckpt",
        "--exclude=*",
        `${LOCAL_LOGS + "/"}`,
        `${DRIVE_CKPTS + "/"}`,
      ],
    });
    await syncCmd.output();
    console.log("  ✅ Local → Drive sync complete.");
  } else {
    console.log(`  [DRY-RUN] Would run: rsync -av --include=*/ --include=*.ckpt --exclude=* ${LOCAL_LOGS}/ ${DRIVE_CKPTS}/`);
  }

  // 2. Prune both local and Drive (union of top K by each metric)
  await pruneCheckpoints(LOCAL_LOGS, KEEP_TOP_K);
  await pruneCheckpoints(DRIVE_CKPTS, KEEP_TOP_K);
}

async function pruneCheckpoints(dir: string, keepTopK: number) {
  if (!isDryRun) {
    const files = await Deno.readDir(dir);
    const parsed: Array<{ name: string; mos?: number; mel?: number }> = [];
    for await (const entry of files) {
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

    const mosRanked = parsed.filter((p) => p.mos !== undefined).sort((a, b) => b.mos! - a.mos!);
    const melRanked = parsed.filter((p) => p.mel !== undefined).sort((a, b) => a.mel! - b.mel!);

    const keep = new Set<string>(["last.ckpt"]);
    mosRanked.slice(0, keepTopK).forEach((p) => keep.add(p.name));
    melRanked.slice(0, keepTopK).forEach((p) => keep.add(p.name));

    const doomed = parsed.map((p) => p.name).filter((n) => !keep.has(n));
    for (const name of doomed) {
      const path = join(dir, name);
      try {
        await Deno.remove(path);
        console.log(`  🗑️ Pruned: ${name}`);
      } catch (err) {
        console.error(`  ⚠️ Failed to prune ${name}: ${err.message}`);
      }
    }
  } else {
    console.log(`  [DRY-RUN] Would prune checkpoints in ${dir} (keep top ${keepTopK} by MOS + MEL)`);
  }
}

// === Keep-alive loop (pings + metrics) ===
function startKeepAliveLoop() {
  console.log("💓 Keep-alive loop activated.");
  if (isDryRun) {
    console.log(`[DRY-RUN] Would ping keep-alive every ${KEEP_ALIVE_INTERVAL_MS}ms.`);
    return;
  }
  let ticks = 0;
  setInterval(async () => {
    ticks++;
    const minutes = (ticks * KEEP_ALIVE_INTERVAL_MS) / 60000;
    let metricsLog = "";
    try {
      const latestMetricFile = join(LOCAL_LOGS, "version_0", "metrics.csv");
      const fileInfo = await statPath(latestMetricFile);
      if (fileInfo && fileInfo.isFile) {
        const text = await Deno.readTextFile(latestMetricFile);
        const lines = text.trim().split("\n");
        if (lines.length > 1) {
          metricsLog = " | Last Metrics: " + lines[lines.length - 1];
        }
      }
    } catch {}
    console.log(`PING [${new Date().toLocaleTimeString()}] - Training Active (${minutes.toFixed(1)} mins elapsed)${metricsLog}`);
  }, KEEP_ALIVE_INTERVAL_MS);
}

// === Start training in tmux (preferred) or foreground ===
async function startTrainingSession(tmuxMode = true) {
  const scriptPath = "/content/run_training.sh";
  const trainingCmdScript = `#!/bin/bash
cd ${REPO_DIR}
python3 -m piper.train fit \\
  --data.voice_name "au_male_57" \\
  --data.csv_path "${METADATA_CSV}" \\
  --data.phoneme_type text \\
  --data.phonemes_path "${PHONEME_MAP}" \\
  --data.audio_dir "${AUDIO_DIR}" \\
  --model.sample_rate 22050 \\
  --data.espeak_voice "en-gb" \\
  --data.cache_dir "${LOCAL_CACHE}/cache" \\
  --data.config_path "${LOCAL_CACHE}/config.json" \\
  --data.batch_size 32 \\
  --trainer.accelerator gpu \\
  --trainer.devices 1 \\
  --trainer.precision 16-mixed \\
  --trainer.callbacks.class_path "train_sutta_voice.SuttaVoiceUatCallback" \\
  --ckpt_path "${BASE_CKPT}" \\
  --model.mel_fmin 0 \\
  --model.mel_fmax 8000
`;

  if (isDryRun) {
    console.log(`[DRY-RUN] Would write launcher to ${scriptPath}`);
    console.log("-------------------------------------------------");
    console.log(trainingCmdScript);
    console.log("-------------------------------------------------");
    return;
  }

  await Deno.writeTextFile(scriptPath, trainingCmdScript);
  const chmod = new Deno.Command("chmod", { args: ["+x", scriptPath] });
  await chmod.output();

  if (tmuxMode) {
    const tmuxListCmd = new Deno.Command("tmux", { args: ["ls"] });
    const out = await tmuxListCmd.output();
    const tmuxList = new TextDecoder().decode(out.stdout);

    if (!tmuxList.includes("piper_train")) {
      console.log("Creating new tmux session 'piper_train'...");
      const startTmux = new Deno.Command("tmux", {
        args: ["new-session", "-d", "-s", "piper_train", "bash", scriptPath],
      });
      await startTmux.output();
      console.log("🎉 Training launched in tmux session 'piper_train'!");
    } else {
      console.log("⚠️ tmux session 'piper_train' already exists. Skipping launch.");
    }
  } else {
    console.log("🔥 Launching training in foreground...");
    await runCmd("bash", [scriptPath]);
  }
}

// === Pip restore (from /content/drive/MyDrive/piper_env_requirements.txt) ===
async function pipRestore() {
  const reqPath = "/content/drive/MyDrive/piper_env_requirements.txt";
  const exists = await statPath(reqPath);
  if (!exists) {
    console.error("❌ Requirements file not found:", reqPath);
    return;
  }
  console.log("📦 Restoring pip environment from requirements.txt...");
  if (!isDryRun) {
    await runCmd("pip", ["install", "-r", reqPath]);
    console.log("✅ Pip restore complete.");
  } else {
    console.log(`[DRY-RUN] Would run: pip install -r ${reqPath}`);
  }
}

// === Piper cache restore (from /content/drive/MyDrive/piper_cache.tgz) ===
async function piperCacheRestore() {
  const archivePath = "/content/drive/MyDrive/piper_cache.tgz";
  const exists = await statPath(archivePath);
  if (!exists) {
    console.error("❌ Cache archive not found:", archivePath);
    return;
  }
  console.log("📦 Restoring piper_cache from archive...");
  if (!isDryRun) {
    await ensureDir(LOCAL_CACHE);
    await runCmd("tar", ["-xzf", archivePath, "-C", "/content"]);
    console.log("✅ piper_cache restore complete.");
  } else {
    console.log(`[DRY-RUN] Would extract ${archivePath} to /content`);
  }
}

// === Diagnostics ===
async function runDiagSetup() {
  console.log("=== SUTTAPLAYER ENVIRONMENT DIAGNOSTIC REPORT ===");
  console.log(`Local Time: ${new Date().toLocaleString()}`);
  console.log(`Working Directory: ${Deno.cwd()}`);
  console.log(`Manifest Version: ${VERSION}`);
  console.log("=================================================");

  console.log("\n1. Auditing Filesystem Paths...");
  const paths = [
    { label: "Drive Base Path", path: DRIVE_BASE, required: true },
    { label: "Local Cache Path", path: LOCAL_CACHE, required: true },
    { label: "Metadata CSV", path: METADATA_CSV, required: true },
    { label: "WAVs Directory", path: AUDIO_DIR, required: true },
    { label: "Phoneme Map JSON", path: PHONEME_MAP, required: true },
    { label: "Base Checkpoint", path: BASE_CKPT, required: true },
  ];
  for (const p of paths) {
    const info = await statPath(p.path);
    if (info) {
      const typeStr = info.isDirectory ? "Dir" : "File";
      const sizeStr = info.size ? ` (${(info.size / (1024 * 1024)).toFixed(2)} MB)` : "";
      console.log(`  [OK]   ${p.label.padEnd(25)} | Present: ${p.path} (${typeStr}${sizeStr})`);
    } else {
      const statusStr = p.required ? "[MISSING - REQUIRED]" : "[MISSING - OPTIONAL]";
      console.log(`  [FAIL] ${p.label.padEnd(25)} | ${statusStr}: ${p.path}`);
    }
  }

  console.log("\n2. Auditing System Binaries...");
  const binaries = ["tmux", "rsync", "inotify-tools", "git", "python3", "apt-get"];
  for (const bin of binaries) {
    try {
      const command = new Deno.Command("which", { args: [bin] });
      const { code, stdout } = await command.output();
      if (code === 0) {
        const path = new TextDecoder().decode(stdout).trim();
        console.log(`  [OK]   ${bin.padEnd(25)} | Installed at: ${path}`);
      } else {
        console.log(`  [FAIL] ${bin.padEnd(25)} | Missing from system PATH`);
      }
    } catch {
      console.log(`  [FAIL] ${bin.padEnd(25)} | Audit command failed`);
    }
  }

  console.log("\n3. Checking tmux sessions...");
  const tmuxListCmd = new Deno.Command("tmux", { args: ["ls"] });
  const out = await tmuxListCmd.output();
  const tmuxList = new TextDecoder().decode(out.stdout);
  if (tmuxList.includes("piper_train")) {
    console.log("  [OK]   tmux session 'piper_train' is running.");
  } else {
    console.log("  [INFO] No 'piper_train' tmux session found.");
  }

  console.log("=================================================");
}

// === Pipeline ===
async function runPipeline() {
  await writeExecutionManifest();

  if (isDiagSetup) {
    await runDiagSetup();
    return;
  }

  if (flags.init) {
    // Optional: you can keep init if you still need it (clone repo, install deps).
    // But since you said you did manual setup, this can be a no-op or skipped.
    console.log("ℹ️ --init skipped (manual setup complete).");
    return;
  }

  if (flags["train"]) {
    await startTrainingSession(true); // tmux
    return;
  }

  if (flags["train-fg"]) {
    await startTrainingSession(false); // foreground
    return;
  }

  if (flags.monitor) {
    console.log("🔄 Checkpoint Sync daemon active.");
    syncAndPruneCheckpoints(); // one-shot; not a long-running interval
    startKeepAliveLoop();
    return;
  }

  if (isSyncSheets) {
    console.log("📊 Google Sheets sync (Python helper required).");
    console.log("Run the Python cell in Colab to sync UAT metrics.");
    return;
  }

  if (isPipRestore) {
    await pipRestore();
    return;
  }

  if (isCacheRestore) {
    await piperCacheRestore();
    return;
  }

  console.log("SuttaPlayer Training Manager (v12)");
  console.log("Usage: deno run --allow-all sutta-training-manager.ts [--init | --train | --train-fg | --monitor | --sync-sheets | --pip-restore | --cache-restore] [--dry-run] [--diag-setup]");
}

runPipeline().catch((err) => {
  console.error("FATAL ERROR running automated manager:", err.message);
  Deno.exit(1);
});