// sutta-training-manager-v9.ts
import { ensureDir } from "jsr:@std/fs/ensure-dir";
import { join, basename } from "jsr:@std/path";

const CHECKPOINT_MONITOR_INTERVAL_MS = 10000; // Poll every 10 seconds
const KEEP_ALIVE_INTERVAL_MS = 30000;        // Ping every 30 seconds

const DRIVE_BASE = "/content/drive/MyDrive/piper_training";
const LOCAL_CACHE = "/content/piper_cache";
const METADATA_CSV = join(DRIVE_BASE, "metadata.csv");
const AUDIO_DIR = join(DRIVE_BASE, "wavs");
const PHONEME_MAP = join(DRIVE_BASE, "phoneme_map.json");
const BASE_CKPT = join(DRIVE_BASE, "en_GB-northern_english_male-medium.ckpt");

const LOCAL_LOGS = join(LOCAL_CACHE, "lightning_logs");
const DRIVE_CKPTS = join(DRIVE_BASE, "checkpoints");

const args = Deno.args;
const isDryRun = args.includes("--dry-run");
const isDiagSetup = args.includes("--diag-setup");

const MAMBA_BIN = join(Deno.cwd(), "bin", "micromamba");

async function runCmd(cmd: string, argsList: string[], options: Deno.CommandOptions = {}) {
  const cmdStr = cmd + " " + argsList.join(" ");
  if (isDryRun) {
    console.log("[DRY-RUN] Would execute: " + cmdStr);
    return "[DRY-RUN OUTPUT]";
  }
  console.log("Executing: " + cmdStr);
  const command = new Deno.Command(cmd, { args: argsList, ...options });
  const { code, stdout, stderr } = await command.output();
  if (code !== 0) {
    const errorDecoder = new TextDecoder();
    console.error("Error executing " + cmd + ": " + errorDecoder.decode(stderr));
    throw new Error("Command failed with exit code " + code);
  }
  return new TextDecoder().decode(stdout);
}

async function statPath(path: string) {
  try {
    const info = await Deno.stat(path);
    return info;
  } catch {
    return null;
  }
}

async function downloadBaseCheckpoint() {
  const url = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_GB/northern_english_male/medium/epoch%3D9029-step%3D2261720.ckpt?download=true";
  if (isDryRun) {
    console.log("[DRY-RUN] Would download: " + url + " directly to " + BASE_CKPT);
    return;
  }
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error("Hugging Face server returned status " + response.status);
    }
    const file = await Deno.open(BASE_CKPT, { write: true, create: true });
    await response.body?.pipeTo(file.writable);
    console.log("  ✅ Base checkpoint downloaded successfully to: " + BASE_CKPT);
  } catch (err) {
    console.error("  ❌ Failed to download base checkpoint automatically: " + (err as Error).message);
    throw err;
  }
}

async function verifyDependencies() {
  console.log("🔍 Auditing Google Drive dependencies...");
  const paths = [\n    { label: "Metadata CSV", path: METADATA_CSV, required: true },
    { label: "WAVs Directory", path: AUDIO_DIR, required: true },
    { label: "Phoneme Map JSON", path: PHONEME_MAP, required: true },
    { label: "Base Checkpoint", path: BASE_CKPT, required: true }
  ];

  let missingRequired = false;

  for (const p of paths) {
    const info = await statPath(p.path);
    if (info) {
      console.log("  ✅ Found: " + p.path + " (" + (info.isDirectory ? "Dir" : "File") + ")");
    } else {
      if (p.required) {
        console.error("  ❌ Missing crucial training dependency: " + p.path);
        if (!isDryRun) {
          missingRequired = true;
        } else {
          console.log("         [DRY-RUN WARN] Missing but proceeding in dry-run mode.");
        }
      }
    }
  }

  const ckptInfo = await statPath(BASE_CKPT);
  if (!ckptInfo && !isDryRun) {
    console.log("\nℹ️ Base checkpoint is missing from Google Drive.");
    console.log("📥 Automatically downloading the pre-trained baseline model from Hugging Face directly to your Drive...");
    try {
      await downloadBaseCheckpoint();
    } catch {
      Deno.exit(1);
    }
  }

  if (missingRequired) {
    console.error("\n❌ Critical: Missing Google Drive folder dependencies. Please ensure '/MyDrive/piper_training/' is populated.");
    Deno.exit(1);
  }
}

async function installSystemTools() {
  console.log("🛠️ Installing system dependencies (tmux, rsync, inotify-tools)...");
  if (isDryRun) {
    console.log("[DRY-RUN] Would run: apt-get update -qq");
    console.log("[DRY-RUN] Would run: apt-get install -y -qq tmux rsync inotify-tools");
  } else {
    await runCmd("apt-get", ["update", "-qq"]);
    await runCmd("apt-get", ["install", "-y", "-qq", "tmux", "rsync", "inotify-tools"]);
    console.log("✅ System dependencies installed.");
  }
}

async function installPythonEnv() {
  console.log("🐍 Setting up Python 3.11.9 Sandboxed environment via micromamba...");
  
  const mambaExist = await Deno.stat("bin/micromamba").then(() => true).catch(() => false);
  if (!mambaExist) {
    console.log("Downloading micromamba...");
    if (isDryRun) {
      console.log("[DRY-RUN] Would run: curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba");
    } else {
      const cmd = new Deno.Command("sh", {
        args: ["-c", "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba"]
      });
      await cmd.output();
    }
  }

  const envExist = await Deno.stat("/root/micromamba/envs/py311").then(() => true).catch(() => false);
  if (!envExist) {
    console.log("Creating virtual Python environment 'py311' with native setuptools package...");
    if (isDryRun) {
      console.log("[DRY-RUN] Would run: " + MAMBA_BIN + " create -y -r /root/micromamba -n py311 python=3.11.9 pip setuptools -c conda-forge");
    } else {
      await runCmd(MAMBA_BIN, [
        "create",
        "-y",
        "-r", "/root/micromamba",
        "-n", "py311",
        "python=3.11.9",
        "pip",
        "setuptools",
        "-c", "conda-forge"
      ]);
    }
  }

  console.log("✅ Python virtual environment established.");
}

async function cloneRepo() {
  const repoDir = "/content/piper1-gpl";
  const repoExist = await Deno.stat(repoDir).then(() => true).catch(() => false);
  if (!repoExist) {
    console.log("📥 Cloning OHF-Voice/piper1-gpl training repository...");
    if (isDryRun) {
      console.log("[DRY-RUN] Would run: git clone https://github.com/OHF-Voice/piper1-gpl.git " + repoDir);
    } else {
      await runCmd("git", ["clone", "https://github.com/OHF-Voice/piper1-gpl.git", repoDir]);
    }
  }
  
  console.log("📦 Installing Python dependencies inside virtual env...");
  const pipInstall = [
    "run",
    "-r", "/root/micromamba",
    "-n", "py311",
    "pip", "install",\n    "setuptools",
    "torch==2.3.1",
    "onnx==1.15.0",
    "lightning==2.3.3",
    "tensorboard",
    "-e", "."
  ];
  if (isDryRun) {
    console.log("[DRY-RUN] Would run inside " + repoDir + ": " + MAMBA_BIN + " " + pipInstall.join(" "));
  } else {
    await runCmd(MAMBA_BIN, pipInstall, { cwd: repoDir });
  }

  console.log("⚙️ Compiling Cython Monotonic Alignment Search (MAS) C-Extension...");
  const buildAlign = [
    "run",
    "-r", "/root/micromamba",
    "-n", "py311",
    "bash", "build_monotonic_align.sh"
  ];
  if (isDryRun) {
    console.log("[DRY-RUN] Would run inside " + repoDir + ": " + MAMBA_BIN + " " + buildAlign.join(" "));
  } else {
    try {
      await runCmd(MAMBA_BIN, buildAlign, { cwd: repoDir });
      console.log("  ✅ Monotonic Alignment Search Compiled successfully!");
    } catch (err) {
      console.warn("  ⚠️ build_monotonic_align.sh failed or was missing. Attempting fallback setup.py build...");
      const fallbackBuild = [
        "run",
        "-r", "/root/micromamba",
        "-n", "py311",
        "python3", "setup.py", "build_ext", "--inplace"
      ];
      try {
        await runCmd(MAMBA_BIN, fallbackBuild, { cwd: join(repoDir, "src", "python") });
        console.log("  ✅ Fallback C-Extension Build completed successfully!");
      } catch (fallbackErr) {
        console.error("  ❌ Critical: Failed to build C-Extensions. Alignment training will run extremely slow!");
      }
    }
  }
}

async function setupLocalCache() {
  console.log("⚡ Setting up local lightning_logs cache directory...");
  if (isDryRun) {
    console.log("[DRY-RUN] Would ensure directory exists: " + LOCAL_CACHE);
    console.log("[DRY-RUN] Would ensure directory exists: " + DRIVE_CKPTS);
  } else {
    await ensureDir(LOCAL_CACHE);
    await ensureDir(DRIVE_CKPTS);
  }
  
  console.log("🔌 Registering train_sutta_voice.py UAT active validation callback inside training loop...");
  const uatCallbackSource = join(DRIVE_BASE, "train_sutta_voice.py");
  const uatCallbackDest = join("/content/piper1-gpl", "train_sutta_voice.py");
  if (isDryRun) {
    console.log("[DRY-RUN] Would copy file from " + uatCallbackSource + " to " + uatCallbackDest);
  } else {
    try {
      await Deno.copyFile(uatCallbackSource, uatCallbackDest);
      console.log("  ✅ train_sutta_voice.py integrated natively.");
    } catch (err) {
      console.error("  ❌ Failed to copy train_sutta_voice.py from Google Drive base folder to local repo!");
    }
  }

  console.log("🔄 Pulling existing checkpoints from Drive to prevent starting from epoch 0...");
  if (isDryRun) {
    console.log("[DRY-RUN] Would rsync from " + DRIVE_CKPTS + "/ to " + LOCAL_LOGS + "/");
  } else {
    try {
      const cmd = new Deno.Command("rsync", {
        args: ["-av", "--ignore-existing", DRIVE_CKPTS + "/", LOCAL_LOGS + "/"]
      });
      await cmd.output();
      console.log("  ✅ Back-sync from Google Drive complete.");
    } catch (err) {
      console.log("No existing checkpoints found on Google Drive. Starting fresh training session.");
    }
  }
}

async function startTrainingSession() {
  console.log("🚀 Initializing TMUX background session for training...");
  
  const trainingCmdScript = `#!/bin/bash
cd /content/piper1-gpl
# Launch training using direct command-line arguments to completely avoid YAML parsing failure modes
/content/bin/micromamba run -r /root/micromamba -n py311 python3 -m piper.train fit \\
  --data.voice_name "en_gb-suttaplayer-medium" \\
  --data.csv_path "${METADATA_CSV}" \\
  --data.audio_dir "${AUDIO_DIR}" \\
  --data.cache_dir "${LOCAL_CACHE}/cache" \\
  --data.config_path "${LOCAL_CACHE}/config.json" \\
  --data.phoneme_type text \\
  --data.phonemes_path "${PHONEME_MAP}" \\
  --data.num_symbols 256 \\
  --data.batch_size 32 \\
  --data.espeak_voice "en-gb" \\
  --model.sample_rate 22050 \\
  --trainer.accelerator gpu \\
  --trainer.devices 1 \\
  --trainer.precision 16-mixed \\
  --trainer.callbacks.class_path "train_sutta_voice.SuttaVoiceUatCallback" \\
  --ckpt_path "${BASE_CKPT}"
`;

  const scriptPath = "/content/run_training.sh";
  if (isDryRun) {
    console.log("\n[DRY-RUN] Would write script to " + scriptPath + " with content:");
    console.log("-------------------------------------------------");
    console.log(trainingCmdScript);
    console.log("-------------------------------------------------");
    console.log("[DRY-RUN] Would run: chmod +x " + scriptPath);
    console.log("[DRY-RUN] Would launch tmux session 'piper_train' with command: bash " + scriptPath);
  } else {
    await Deno.writeTextFile(scriptPath, trainingCmdScript);
    const cmd = new Deno.Command("chmod", { args: ["+x", scriptPath] });
    await cmd.output();

    const tmuxListCmd = new Deno.Command("tmux", { args: ["ls"] });\n    const out = await tmuxListCmd.output();\n    const tmuxList = new TextDecoder().decode(out.stdout);\n\n    if (!tmuxList.includes(\"piper_train\")) {\n      console.log(\"Creating new tmux session 'piper_train'...\");\n      const startTmux = new Deno.Command(\"tmux\", {\n        args: [\"new-session\", \"-d\", \"-s\", \"piper_train\", \"bash\", scriptPath]\n      });\n      await startTmux.output();\n      console.log(\"🎉 Training successfully launched in tmux session 'piper_train'!\");\n    } else {\n      console.log(\"⚠️ A tmux session named 'piper_train' is already running. Monitoring active training.\");\n    }\n  }\n}\n\nasync function monitorCheckpoints() {\n  console.log(\"🔄 Checkpoint Sync daemon active. Scanning for new checkpoint iterations...\");\n  if (isDryRun) {\n    console.log(\"[DRY-RUN] Would initiate checkpoint polling interval (rsync \" + LOCAL_LOGS + \"/ -> \" + DRIVE_CKPTS + \"/)\");\n    return;\n  }\n  \n  setInterval(async () => {\n    try {\n      const logsExist = await Deno.stat(LOCAL_LOGS).then(() => true).catch(() => false);\n      if (!logsExist) return;\n\n      const cmd = new Deno.Command(\"rsync\", {\n        args: [\n          \"-av\",\n          \"--include=*/\",\n          \"--include=*.ckpt\",\n          \"--exclude=*\",\n          LOCAL_LOGS + \"/\",\n          DRIVE_CKPTS + \"/\"\n        ]\n      });\n      const { code } = await cmd.output();\n      if (code === 0) {\n        const findCmd = new Deno.Command(\"find\", {\n          args: [LOCAL_LOGS, \"-name\", \"*.ckpt\", \"-mmin\", \"-2\"]\n        });\n        const out = await findCmd.output();\n        const latest = new TextDecoder().decode(out.stdout).trim();\n        if (latest) {\n          console.log(\"✨ [Drive Sync] Newly calculated checkpoints synchronized to Drive:\\n  \" + latest);\n        }\n      }\n    } catch (err) {\n      console.error(\"Error running checkpoint rsync loop:\", err);\n    }\n  }, CHECKPOINT_MONITOR_INTERVAL_MS);\n}\n\nfunction startKeepAliveLoop() {\n  console.log(\"💓 Keep-alive loop activated. Suppressing socket timeouts...\");\n  if (isDryRun) {\n    console.log(\"[DRY-RUN] Would initiate keep-alive polling interval.\");\n    return;\n  }\n  let ticks = 0;\n  setInterval(async () => {\n    ticks++;\n    const minutes = (ticks * KEEP_ALIVE_INTERVAL_MS) / 60000;\n    \n    let metricsLog = \"\";\n    try {\n      const latestMetricFile = \"/content/piper_cache/lightning_logs/version_0/metrics.csv\";\n      const fileInfo = await Deno.stat(latestMetricFile);\n      if (fileInfo.isFile) {\n        const text = await Deno.readTextFile(latestMetricFile);\n        const lines = text.trim().split(\"\\n\");\n        if (lines.length > 1) {\n          metricsLog = \" | Last Metrics: \" + lines[lines.length - 1];\n        }\n      }\n    } catch {}\n\n    console.log(\"PING [\" + new Date().toLocaleTimeString() + \"] - Training Active (\" + minutes.toFixed(1) + \" mins elapsed)\" + metricsLog);\n  }, KEEP_ALIVE_INTERVAL_MS);\n}\n\nasync function runDiagSetup() {\n  console.log(\"=== SUTTAPLAYER ENVIRONMENT DIAGNOSTIC REPORT ===\");\n  console.log(\"Local Time: \" + new Date().toLocaleString());\n  console.log(\"Working Directory: \" + Deno.cwd());\n  console.log(\"=================================================\");\n\n  console.log(\"\\n1. Auditing Filesystem Paths...\");\n  const paths = [\n    { label: \"Drive Base Path\", path: DRIVE_BASE, required: true },\n    { label: \"Local Cache Path\", path: LOCAL_CACHE, required: true },\n    { label: \"Metadata CSV\", path: METADATA_CSV, required: true },\n    { label: \"WAVs Directory\", path: AUDIO_DIR, required: true },\n    { label: \"Phoneme Map JSON\", path: PHONEME_MAP, required: true },\n    { label: \"Base Checkpoint\", path: BASE_CKPT, required: true }\n  ];\n\n  for (const p of paths) {\n    const info = await statPath(p.path);\n    if (info) {\n      const typeStr = info.isDirectory ? \"Dir\" : \"File\";\n      const sizeStr = info.size ? \" (\" + (info.size / (1024 * 1024)).toFixed(2) + \" MB)\" : \"\";\n      console.log(\"  [OK]   \" + p.label.padEnd(25) + \" | Present: \" + p.path + \" (\" + typeStr + sizeStr + \")\");\n    } else {\n      const statusStr = p.required ? \"[MISSING - REQUIRED]\" : \"[MISSING - OPTIONAL]\";\n      console.log(\"  [FAIL] \" + p.label.padEnd(25) + \" | \" + statusStr + \": \" + p.path);\n    }\n  }\n\n  console.log(\"\\n2. Auditing System Binaries...\");\n  const binaries = [\"tmux\", \"rsync\", \"inotify-tools\", \"git\", \"python3\", \"apt-get\"];\n  for (const bin of binaries) {\n    try {\n      const command = new Deno.Command(\"which\", { args: [bin] });\n      const { code, stdout } = await command.output();\n      if (code === 0) {\n        const path = new TextDecoder().decode(stdout).trim();\n        console.log(\"  [OK]   \" + bin.padEnd(25) + \" | Installed at: \" + path);\n      } else {\n        console.log(\"  [FAIL] \" + bin.padEnd(25) + \" | Missing from system PATH\");\n      }\n    } catch {\n      console.log(\"  [FAIL] \" + bin.padEnd(25) + \" | Audit command failed\");\n    }\n  }\n\n  console.log(\"\\n3. Auditing Micromamba & Python Environments...\");\n  const mambaExist = await Deno.stat(\"bin/micromamba\").then(() => true).catch(() => false);\n  console.log(\"  [OK]   \" + \"micromamba Binary\".padEnd(25) + \" | \" + (mambaExist ? \"Installed in bin/micromamba\" : \"Missing\"));\n\n  const envExist = await Deno.stat(\"/root/micromamba/envs/py311\").then(() => true).catch(() => false);\n  console.log(\"  [OK]   \" + \"py311 Virtual Env\".padEnd(25) + \" | \" + (envExist ? \"Established in /root/micromamba/envs/py311\" : \"Missing\"));\n\n  console.log(\"=================================================\");\n}\n\nasync function runPipeline() {\n  if (isDiagSetup) {\n    await runDiagSetup();\n    return;\n  }\n\n  if (args.includes(\"--init\")) {\n    await verifyDependencies();\n    await installSystemTools();\n    await installPythonEnv();\n    await cloneRepo();\n  } else if (args.includes(\"--train\")) {\n    await setupLocalCache();\n    await startTrainingSession();\n  } else if (args.includes(\"--monitor\")) {\n    monitorCheckpoints();\n    startKeepAliveLoop();\n  } else {\n    console.log(\"SuttaPlayer Training Manager (v5)\");\n    console.log(\"Usage: deno run --allow-all sutta-training-manager-v5.ts [--init | --train | --monitor] [--dry-run] [--diag-setup]\");\n  }\n}\n\nrunPipeline().catch((err) => {\n  console.error(\"FATAL ERROR running automated manager:\", err);\n  Deno.exit(1);\n});\n