// sutta-training-manager.ts
// Version: 11.0.0
// Pure versionless orchestrator with self-healing, automated symlinking, and foreground training modes.

import { parseArgs } from "jsr:@std/cli/parse-args";
import { ensureDir } from "jsr:@std/fs/ensure-dir";
import { join } from "jsr:@std/path";

const VERSION = "11.0.0";

// Standard Path Specifications
const DRIVE_BASE = "/content/drive/MyDrive/sutta-tts-model-training"; // Git clone source of truth
const PIPER_TRAINING = "/content/drive/MyDrive/piper_training"; // Training output directory
const LOCAL_CACHE = "/content/piper_cache";
const REPO_DIR = "/content/piper1-gpl";
const MAMBA_BIN = "bin/micromamba";

// Core Data Asset Paths
const METADATA_CSV = join(DRIVE_BASE, "/corpus-preperation/metadata-phonemes.csv");
const PHONEME_MAP = join(DRIVE_BASE, "/config/en[gb]_pi[si]-suttaplayer-phoneme-map.json");
const AUDIO_DIR = join(PIPER_TRAINING, "wavs");
const BASE_CKPT = join(PIPER_TRAINING, "en_GB-northern_english_male-medium.ckpt");

const LOCAL_LOGS = join(LOCAL_CACHE, "lightning_logs");
const DRIVE_CKPTS = join(PIPER_TRAINING, "checkpoints");

const KEEP_ALIVE_INTERVAL_MS = 60000; // 1 minute
const CHECKPOINT_MONITOR_INTERVAL_MS = 120000; // 2 minutes

// Parse CLI arguments
const args = Deno.args;
const flags = parseArgs(args, {
  boolean: ["init", "train", "train-fg", "monitor", "dry-run", "diag-setup"],
  alias: {
    i: "init",
    t: "train",
    f: "train-fg",
    m: "monitor",
    d: "dry-run",
    g: "diag-setup",
  },
});

const isDryRun = flags["dry-run"] || false;
const isDiagSetup = flags["diag-setup"] || false;

// Log version instantly to stdout on execution
console.log(`=================================================`);
console.log(`🎙️ Sutta TTS Training Manager v${VERSION}`);
console.log(`=================================================`);

// Write filesystem execution manifest
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

// Symlink configurations and modules from the Drive "Source of Truth" repo
async function createSourceSymlinks() {
  console.log("🔗 Verifying and establishing symlinks to Git Source of Truth...");
  if (isDryRun) {
    console.log(`[DRY-RUN] Would symlink files in ${DRIVE_BASE} to ${REPO_DIR}`);
    return;
  }

  const fileMappings = [
    { src: join(DRIVE_BASE, "train_sutta_voice.py"), dest: join(REPO_DIR, "train_sutta_voice.py") },
    { src: join(DRIVE_BASE, "train_sutta_voice.py"), dest: join(REPO_DIR, "src/piper/train/train_sutta_voice.py") },
    { src: join(DRIVE_BASE, "metadata-text.csv"), dest: "/content/metadata-text.csv" },
    { src: join(DRIVE_BASE, "phoneme_map.json"), dest: "/content/phoneme_map.json" }
  ];

  for (const mapping of fileMappings) {
    const srcStat = await statPath(mapping.src);
    if (!srcStat) {
      console.warn(`  ⚠️ Symlink source missing: ${mapping.src}`);
      continue;
    }

    // Clear existing dest symlink or file
    try {
      await Deno.remove(mapping.dest);
    } catch {
      // Ignored if file does not exist
    }

    try {
      await Deno.symlink(mapping.src, mapping.dest);
      console.log(`  ✅ Symlinked: ${mapping.dest} -> ${mapping.src}`);
    } catch (err) {
      console.error(`  ❌ Symlink failed for ${mapping.dest}:`, err.message);
    }
  }
}

// 1. Dynamic Environment Self-Healer
async function runSelfHealer(): Promise<boolean> {
  console.log("🩺 Checking Python environment for dependency health...");
  const checkCmd = [
    "run",
    "-r", "/root/micromamba",
    "-n", "py311",
    "python3", "-c", "import lightning; import setuptools; import torch; print('ENVIRONMENT_OK')"
  ];
  const check = await runCmdWithOutput(join(Deno.cwd(), "bin/micromamba"), checkCmd);
  if (check.success && check.stdout.includes("ENVIRONMENT_OK")) {
    console.log("  ✅ Python virtual environment is 100% healthy!");
    return true;
  }

  console.log("  ⚠️ Environment defect detected (missing setuptools or broken path hooks)!");
  console.log("  🛠️ Launching automated self-healing repair loop...");
  
  const healCmd = [
    "run",
    "-r", "/root/micromamba",
    "-n", "py311",
    "python3", "-m", "pip", "install", "--upgrade", "setuptools==81.0.0"
  ];
  
  const repair = await runCmd(join(Deno.cwd(), "bin/micromamba"), healCmd);
  if (repair.success) {
    console.log("  ✅ Self-healer successfully injected setuptools==81.0.0!");
    return true;
  } else {
    console.error("  ❌ Environment repair failed. A complete rebuilding via --init might be needed.");
    return false;
  }
}

async function verifyDependencies() {
  console.log("🔍 Verifying directory structure and base checkpoint existence...");
  const paths = [
    { label: "Drive Source of Truth Base", path: DRIVE_BASE, required: true },
    { label: "WAVs Directory", path: AUDIO_DIR, required: true },
    { label: "Metadata CSV", path: METADATA_CSV, required: true },
    { label: "Phoneme Map JSON", path: PHONEME_MAP, required: true },
    { label: "Piper Training Outputs Base", path: PIPER_TRAINING, required: true }
  ];

  let missingRequired = false;
  for (const p of paths) {
    const info = await statPath(p.path);
    if (info) {
      console.log(`  ✅ Found: ${p.path} (${info.isDirectory ? "Dir" : "File"})`);
    } else if (p.required) {
      console.error(`  ❌ Missing crucial training dependency: ${p.path}`);
      missingRequired = true;
    }
  }

  // Base checkpoint check and auto-recovery from Hugging Face if missing
  const ckptInfo = await statPath(BASE_CKPT);
  if (!ckptInfo && !isDryRun) {
    console.log(`\nℹ️ Base checkpoint 'en_GB-northern_english_male-medium.ckpt' is missing from ${DRIVE_BASE}.`);
    console.log("📥 Automatically downloading the pre-trained non-rhotic baseline model from Hugging Face...");
    try {
      await ensureDir(DRIVE_BASE);
      const url = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_GB/northern_english_male/medium/epoch%3D9029-step%3D2261720.ckpt";
      const download = new Deno.Command("curl", {
        args: ["-L", "-o", BASE_CKPT, url],
        stdout: "inherit",
        stderr: "inherit"
      });
      const dlStatus = await download.spawn().status;
      if (dlStatus.success) {
        console.log("  ✅ Base checkpoint downloaded successfully!");
      } else {
        throw new Error("Download failed");
      }
    } catch (err) {
      console.error("  ❌ Failed to download base checkpoint automatically:", err.message);
      Deno.exit(1);
    }
  }

  if (missingRequired && !isDryRun) {
    console.error("\n❌ Critical: Missing Google Drive folder dependencies. Please ensure paths are correct.");
    Deno.exit(1);
  }
}

async function installSystemTools() {
  console.log("🛠️ Installing system dependencies (tmux, rsync, inotify-tools)...");
  if (isDryRun) {
    console.log("[DRY-RUN] Would run: apt-get update -qq && apt-get install -y -qq tmux rsync inotify-tools");
    return;
  }
  await runCmd("apt-get", ["update", "-qq"]);
  await runCmd("apt-get", ["install", "-y", "-qq", "tmux", "rsync", "inotify-tools"]);
  console.log("✅ System dependencies installed.");
}

async function installPythonEnv() {
  console.log("🐍 Setting up Python 3.11.9 Sandboxed environment via micromamba...");
  const mambaExist = await statPath("bin/micromamba");
  if (!mambaExist) {
    console.log("Downloading micromamba...");
    if (!isDryRun) {
      const download = new Deno.Command("sh", {
        args: ["-c", "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba"]
      });
      await download.output();
    }
  }

  const envExist = await statPath("/root/micromamba/envs/py311");
  if (!envExist) {
    console.log("Creating virtual Python environment 'py311' with native setuptools...");
    // Inject setuptools directly at mamba-create level to bypass pkg_resources defects
    await runCmd(join(Deno.cwd(), MAMBA_BIN), [
      "create",
      "-y",
      "-r", "/root/micromamba",
      "-n", "py311",
      "python=3.11.9",
      "pip",
      "setuptools==81.0.0",
      "-c", "conda-forge"
    ]);
  } else {
    // If environment exists, trigger dynamic self-healer check
    await runSelfHealer();
  }
  console.log("✅ Python virtual environment established.");
}

async function cloneRepo() {
  const repoExist = await statPath(REPO_DIR);
  if (!repoExist) {
    console.log("📥 Cloning OHF-Voice/piper1-gpl training repository...");
    await runCmd("git", ["clone", "https://github.com/OHF-Voice/piper1-gpl.git", REPO_DIR]);
  }

  console.log("📦 Installing Python dependencies inside virtual env...");
  // Use absolute namespaced python pip calls to prevent system-path bleed
  const pipInstall = [
    "run",
    "-r", "/root/micromamba",
    "-n", "py311",
    "python3", "-m", "pip", "install",
    "torch==2.3.1",
    "onnx==1.15.0",
    "lightning==2.3.3",
    "tensorboard",
    "-e", "."
  ];
  await runCmd(join(Deno.cwd(), MAMBA_BIN), pipInstall, { cwd: REPO_DIR });

  console.log("⚙️ Compiling Cython Monotonic Alignment Search (MAS) C-Extension...");
  const buildAlign = [
    "run",
    "-r", "/root/micromamba",
    "-n", "py311",
    "bash", "build_monotonic_align.sh"
  ];
  await runCmd(join(Deno.cwd(), MAMBA_BIN), buildAlign, { cwd: REPO_DIR });
}

async function setupLocalCache() {
  console.log("📁 Preparing local SSD cache directory to protect VRAM overhead...");
  if (!isDryRun) {
    await ensureDir(LOCAL_CACHE);
    await ensureDir(join(LOCAL_CACHE, "cache"));
  }
  console.log(`  ✅ Local cache established at: ${LOCAL_CACHE}`);
}

function getTrainingScriptContent() {
  return `#!/bin/bash
cd ${REPO_DIR}
/content/bin/micromamba run -r /root/micromamba -n py311 python3 -m piper.train fit \
  --data.voice_name "au_male_57" \
  --data.csv_path "${METADATA_CSV}" \
  --data.audio_dir "${AUDIO_DIR}" \
  --data.cache_dir "${LOCAL_CACHE}/cache" \
  --data.config_path "${LOCAL_CACHE}/config.json" \
  --data.phoneme_type text \
  --data.phonemes_path "${PHONEME_MAP}" \
  --data.num_symbols 256 \
  --data.batch_size 32 \
  --data.espeak_voice "en-gb" \
  --model.sample_rate 22050 \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision 16-mixed \
  --trainer.callbacks.class_path "train_sutta_voice.SuttaVoiceUatCallback" \
  --ckpt_path "${BASE_CKPT}"
`;
}

async function startTrainingSession(tmuxMode = true) {
  await createSourceSymlinks();

  // Establish local checkpoints directory
  if (!isDryRun) {
    await ensureDir(LOCAL_LOGS);
    await ensureDir(DRIVE_CKPTS);
  }

  console.log("🔄 Syncing pre-existing checkpoints back from Drive to local cache...");
  if (!isDryRun) {
    try {
      const syncBack = new Deno.Command("rsync", {
        args: ["-av", "--ignore-existing", DRIVE_CKPTS + "/", LOCAL_LOGS + "/"]
      });
      await syncBack.output();
      console.log("  ✅ Checkpoint recovery sync complete.");
    } catch {
      console.log("  ℹ️ No existing checkpoints found on Google Drive. Starting fresh training session.");
    }
  }

  const scriptPath = "/content/run_training.sh";
  const trainingCmdScript = getTrainingScriptContent();

  if (isDryRun) {
    console.log(`\n[DRY-RUN] Would write launcher to ${scriptPath} with content:`);
    console.log("-------------------------------------------------");
    console.log(trainingCmdScript);
    console.log("-------------------------------------------------");
    return;
  }

  await Deno.writeTextFile(scriptPath, trainingCmdScript);
  const chmod = new Deno.Command("chmod", { args: ["+x", scriptPath] });
  await chmod.output();

  if (tmuxMode) {
    console.log("🚀 Initializing TMUX background session for training...");
    const tmuxListCmd = new Deno.Command("tmux", { args: ["ls"] });
    const out = await tmuxListCmd.output();
    const tmuxList = new TextDecoder().decode(out.stdout);

    if (!tmuxList.includes("piper_train")) {
      console.log("Creating new tmux session 'piper_train'...");
      const startTmux = new Deno.Command("tmux", {
        args: ["new-session", "-d", "-s", "piper_train", "bash", scriptPath]
      });
      await startTmux.output();
      console.log("🎉 Training successfully launched in tmux session 'piper_train'!");
    } else {
      console.log("⚠️ A tmux session named 'piper_train' is already running. Monitoring active training.");
    }
  } else {
    console.log("🔥 Launching training in the foreground of this cell...");
    await runCmd("bash", [scriptPath]);
  }
}

async function monitorCheckpoints() {
  console.log("🔄 Checkpoint Sync daemon active. Scanning for new checkpoint iterations...");
  if (isDryRun) {
    console.log(`[DRY-RUN] Would polling-sync checkpoints every ${CHECKPOINT_MONITOR_INTERVAL_MS}ms.`);
    return;
  }

  setInterval(async () => {
    try {
      const logsExist = await statPath(LOCAL_LOGS);
      if (!logsExist) return;

      const cmd = new Deno.Command("rsync", {
        args: [
          "-av",
          "--include=*/",
          "--include=*.ckpt",
          "--exclude=*",
          LOCAL_LOGS + "/",
          DRIVE_CKPTS + "/"
        ]
      });
      const { code } = await cmd.output();
      if (code === 0) {
        const findCmd = new Deno.Command("find", {
          args: [LOCAL_LOGS, "-name", "*.ckpt", "-mmin", "-2"]
        });
        const out = await findCmd.output();
        const latest = new TextDecoder().decode(out.stdout).trim();
        if (latest) {
          console.log(`✨ [Drive Sync] Newly calculated checkpoints synchronized to Drive:\n  ${latest}`);
        }
      }
    } catch (err) {
      console.error("Error running checkpoint rsync loop:", err.message);
    }
  }, CHECKPOINT_MONITOR_INTERVAL_MS);
}

function startKeepAliveLoop() {
  console.log("💓 Keep-alive loop activated. Suppressing socket timeouts...");
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
      const latestMetricFile = "/content/piper_cache/lightning_logs/version_0/metrics.csv";
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
    { label: "Base Checkpoint", path: BASE_CKPT, required: true }
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

  console.log("\n3. Auditing Environment Health...");
  await runSelfHealer();
  console.log("=================================================");
}

async function runPipeline() {
  await writeExecutionManifest();

  if (isDiagSetup) {
    await runDiagSetup();
    return;
  }

  if (flags.init) {
    await verifyDependencies();
    await installSystemTools();
    await installPythonEnv();
    await cloneRepo();
  } else if (flags.train) {
    await setupLocalCache();
    await startTrainingSession(true); // TMUX Background Mode
  } else if (flags["train-fg"]) {
    await setupLocalCache();
    await startTrainingSession(false); // Foreground Mode
  } else if (flags.monitor) {
    monitorCheckpoints();
    startKeepAliveLoop();
  } else {
    console.log("SuttaPlayer Training Manager (v11)");
    console.log("Usage: deno run --allow-all sutta-training-manager.ts [--init | --train | --train-fg | --monitor] [--dry-run] [--diag-setup]");
  }
}

runPipeline().catch((err) => {
  console.error("FATAL ERROR running automated manager:", err.message);
  Deno.exit(1);
});
