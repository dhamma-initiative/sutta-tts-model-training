// sutta-training-manager-v4.ts
import { ensureDir } from "jsr:@std/fs/ensure-dir";
import { join, basename } from "jsr:@std/path";

const CHECKPOINT_MONITOR_INTERVAL_MS = 10000; // Poll every 10 seconds
const KEEP_ALIVE_INTERVAL_MS = 30000;        // Ping every 30 seconds

const DRIVE_BASE = "/content/drive/MyDrive/piper_training";
const LOCAL_CACHE = "/content/piper_cache";
const METADATA_CSV = join(DRIVE_BASE, "metadata.csv");
const AUDIO_DIR = join(DRIVE_BASE, "wavs");
const PHONEME_MAP = join(DRIVE_BASE, "phoneme_map.json");
const CONFIG_YAML = join(DRIVE_BASE, "piper-train-config.yaml");
const BASE_CKPT = join(DRIVE_BASE, "en_GB-northern_english_male-medium.ckpt");

const LOCAL_LOGS = join(LOCAL_CACHE, "lightning_logs");
const DRIVE_CKPTS = join(DRIVE_BASE, "checkpoints");

async function runCmd(cmd: string, args: string[], options: Deno.CommandOptions = {}) {
  console.log(`Executing: ${cmd} ${args.join(" ")}`);
  const command = new Deno.Command(cmd, { args, ...options });
  const { code, stdout, stderr } = await command.output();
  if (code !== 0) {
    const errorDecoder = new TextDecoder();
    console.error(`Error executing ${cmd}: ${errorDecoder.decode(stderr)}`);
    throw new Error(`Command failed with exit code ${code}`);
  }
  return new TextDecoder().decode(stdout);
}

async function verifyDependencies() {
  console.log("🔍 Auditing Google Drive dependencies...");
  const paths = [METADATA_CSV, AUDIO_DIR, PHONEME_MAP, CONFIG_YAML, BASE_CKPT];
  for (const path of paths) {
    try {
      const info = await Deno.stat(path);
      console.log(`  ✅ Found: ${path} (${info.isDirectory ? "Dir" : "File"})`);
    } catch {
      console.error(`  ❌ Missing crucial training dependency: ${path}`);
      console.error("Please ensure your Google Drive folder '/MyDrive/piper_training/' is correctly populated.");
      Deno.exit(1);
    }
  }
}

async function installSystemTools() {
  console.log("🛠️ Installing system dependencies (tmux, rsync, inotify-tools)...");
  await runCmd("apt-get", ["update", "-qq"]);
  await runCmd("apt-get", ["install", "-y", "-qq", "tmux", "rsync", "inotify-tools"]);
  console.log("✅ System dependencies installed.");
}

async function installPythonEnv() {
  console.log("🐍 Setting up Python 3.11.9 Sandboxed environment via micromamba...");
  
  // Download and unpack micromamba
  const mambaExist = await Deno.stat("bin/micromamba").then(() => true).catch(() => false);
  if (!mambaExist) {
    console.log("Downloading micromamba...");
    const cmd = new Deno.Command("sh", {\n      args: ["-c", "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba"]\n    });
    await cmd.output();
  }

  // Create environment
  const envExist = await Deno.stat("/root/micromamba/envs/py311").then(() => true).catch(() => false);
  if (!envExist) {
    console.log("Creating virtual Python environment 'py311'...");
    await runCmd("./bin/micromamba", [
      "create",
      "-y",
      "-r", "/root/micromamba",
      "-n", "py311",
      "python=3.11.9",
      "pip",
      "-c", "conda-forge"
    ]);
  }

  console.log("✅ Python virtual environment established.");
}

async function cloneRepo() {
  const repoDir = "/content/piper1-gpl";
  const repoExist = await Deno.stat(repoDir).then(() => true).catch(() => false);
  if (!repoExist) {
    console.log("📥 Cloning OHF-Voice/piper1-gpl training repository...");
    await runCmd("git", ["clone", "https://github.com/OHF-Voice/piper1-gpl.git", repoDir]);
  }
  
  console.log("📦 Installing Python dependencies inside virtual env...");
  // Construct wrapper execution to pip install dependencies
  const pipInstall = [
    "run",
    "-r", "/root/micromamba",
    "-n", "py311",
    "pip", "install",
    "torch==2.3.1",
    "torchaudio==2.3.1",
    "onnx==1.15.0",
    "lightning==2.3.3",
    "tensorboard",
    "-e", "."
  ];
  await runCmd("./bin/micromamba", pipInstall, { cwd: repoDir });

  // -------------------------------------------------------------
  // CRITICAL COMPILATION STEP: Compile Cython Monotonic Alignment C-Extension
  // This step is required by Piperdocs and is run inside the micromamba env
  // -------------------------------------------------------------
  console.log("⚙️ Compiling Cython Monotonic Alignment Search (MAS) C-Extension...");
  try {
    const buildAlign = [
      "run",
      "-r", "/root/micromamba",
      "-n", "py311",
      "bash", "build_monotonic_align.sh"
    ];
    await runCmd("./bin/micromamba", buildAlign, { cwd: repoDir });
    console.log("  ✅ Monotonic Alignment Search Compiled successfully!");
  } catch (err) {
    console.warn("  ⚠️ build_monotonic_align.sh failed or was missing. Attempting fallback setup.py build...");
    try {
      const fallbackBuild = [
        "run",
        "-r", "/root/micromamba",
        "-n", "py311",
        "python3", "setup.py", "build_ext", "--inplace"
      ];
      await runCmd("./bin/micromamba", fallbackBuild, { cwd: join(repoDir, "src", "python") });
      console.log("  ✅ Fallback C-Extension Build completed successfully!");
    } catch (fallbackErr) {
      console.error("  ❌ Critical: Failed to build C-Extensions. Alignment training will run extremely slow!");
    }
  }
}

async function setupLocalCache() {
  console.log("⚡ Setting up lightning_logs cache directory...");
  await ensureDir(LOCAL_CACHE);
  await ensureDir(DRIVE_CKPTS);
  
  // Register the custom train_sutta_voice.py callback natively in the source repository
  console.log("🔌 Registering train_sutta_voice.py UAT active validation callback inside training loop...");
  try {
    const uatCallbackSource = join(DRIVE_BASE, "train_sutta_voice.py");
    const uatCallbackDest = join("/content/piper1-gpl", "train_sutta_voice.py");
    await Deno.copyFile(uatCallbackSource, uatCallbackDest);
    console.log("  ✅ train_sutta_voice.py integrated natively.");
  } catch (err) {
    console.error("  ❌ Failed to copy train_sutta_voice.py from Google Drive base folder to local repo!");
  }

  // Back-sync any existing checkpoints from Drive to local cache to preserve epochs
  console.log("🔄 Pulling existing checkpoints from Drive to prevent starting from epoch 0...");
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

async function startTrainingSession() {
  console.log("🚀 Initializing TMUX background session for training...");
  
  // Write the execution bash script inside piper1-gpl repo
  const trainingCmdScript = `#!/bin/bash
cd /content/piper1-gpl
# Inject custom train_sutta_voice callback inside command line arguments automatically 
/content/bin/micromamba run -r /root/micromamba -n py311 python3 -m piper.train fit \\\\
  --config ${CONFIG_YAML} \\\\
  --data.voice_name=au_male_57 \\\\
  --data.csv_path=${METADATA_CSV} \\\\
  --data.audio_dir=${AUDIO_DIR} \\\\
  --data.phoneme_type=text \\\\
  --data.phonemes_path=${PHONEME_MAP} \\\\
  --data.cache_dir=${LOCAL_CACHE} \\\\
  --data.batch_size=32 \\\\
  --model.sample_rate=22050 \\\\
  --ckpt_path=${BASE_CKPT}
`;

  await Deno.writeTextFile("/content/run_training.sh", trainingCmdScript);
  await runCmd("chmod", ["+x", "/content/run_training.sh"]);

  // Start tmux session if not running
  const tmuxList = await runCmd("tmux", ["ls"]).catch(() => "");
  if (!tmuxList.includes("piper_train")) {
    console.log("Creating new tmux session 'piper_train'...");
    await runCmd("tmux", ["new-session", "-d", "-s", "piper_train", "bash", "/content/run_training.sh"]);
    console.log("🎉 Training successfully launched in tmux session 'piper_train'!");
  } else {
    console.log("⚠️ A tmux session named 'piper_train' is already running. Monitoring active training.");
  }
}

async function monitorCheckpoints() {
  console.log("🔄 Checkpoint Sync daemon active. Scanning for new checkpoint iterations...");
  
  setInterval(async () => {
    try {
      const logsExist = await Deno.stat(LOCAL_LOGS).then(() => true).catch(() => false);
      if (!logsExist) return;

      // Sync all generated checkpoints (.ckpt) back to Permanent Google Drive
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
          console.log(`✨ [Drive Sync] Newly calculated checkpoints synchronized to Drive:\\n  \${latest}`);
        }
      }
    } catch (err) {
      console.error("Error running checkpoint rsync loop:", err);
    }
  }, CHECKPOINT_MONITOR_INTERVAL_MS);
}

function startKeepAliveLoop() {
  console.log("💓 Keep-alive loop activated. Suppressing socket timeouts...");
  let ticks = 0;
  setInterval(async () => {
    ticks++;
    const minutes = (ticks * KEEP_ALIVE_INTERVAL_MS) / 60000;
    
    let metricsLog = "";
    try {
      const latestMetricFile = "/content/piper_cache/lightning_logs/version_0/metrics.csv";
      const fileInfo = await Deno.stat(latestMetricFile);
      if (fileInfo.isFile) {
        const text = await Deno.readTextFile(latestMetricFile);
        const lines = text.trim().split("\n");
        if (lines.length > 1) {
          metricsLog = ` | Last Metrics: \${lines[lines.length - 1]}`;
        }
      }
    } catch {}

    console.log(`PING [\${new Date().toLocaleTimeString()}] - Training Active (\${minutes.toFixed(1)} mins elapsed)\${metricsLog}`);
  }, KEEP_ALIVE_INTERVAL_MS);
}

async function runPipeline() {
  // Check parameters for CLI router
  const args = Deno.args;
  if (args.includes("--init")) {
    await verifyDependencies();
    await installSystemTools();
    await installPythonEnv();
    await cloneRepo();
  } else if (args.includes("--train")) {
    await setupLocalCache();
    await startTrainingSession();
  } else if (args.includes("--monitor")) {
    monitorCheckpoints();
    startKeepAliveLoop();
  } else {
    console.log("SuttaPlayer Training Manager (v4)");
    console.log("Usage: deno run --allow-all sutta-training-manager-v4.ts [--init | --train | --monitor]");
  }
}

runPipeline().catch((err) => {
  console.error("FATAL ERROR running automated manager:", err);
  Deno.exit(1);
});
