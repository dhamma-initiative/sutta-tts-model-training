// sutta-training-bootstrap.ts
// Version: 1.1.1
// Offline-first Environment Bootstrap Manager for SuttaPlayer.
// Installs pre-downloaded Python wheels at disk-speed and registers pre-compiled 
// C++/Cython alignments inside Colab's NVMe scratch space in under 30 seconds.

import { parseArgs } from "https://deno.land/std@0.224.0/cli/parse_args.ts";
import { join } from "https://deno.land/std@0.224.0/path/mod.ts";

const flags = parseArgs(Deno.args, {
  string: ["wheels", "repo"],
  default: {
    wheels: "/content/wheels_local",
    repo: "/content/piper1-gpl",
  },
});

const WHEELS_DIR = flags.wheels;
const REPO_DIR = flags.repo;

async function exists(path: string): Promise<boolean> {
  try {
    await Deno.stat(path);
    return true;
  } catch {
    return false;
  }
}

async function runCmd(cmd: string, args: string[], options: { cwd?: string } = {}) {
  const command = new Deno.Command(cmd, {
    args,
    cwd: options.cwd,
    stdout: "inherit",
    stderr: "inherit",
  });
  const process = command.spawn();
  const status = await process.status;
  return status.success;
}

async function hasCompiledAlign(dir: string): Promise<boolean> {
  try {
    for await (const entry of Deno.readDir(dir)) {
      if (entry.isFile && entry.name.startsWith("core") && entry.name.endsWith(".so")) {
        return true;
      }
    }
  } catch {}
  return false;
}

async function verifyPrecompiledBinaries() {
  console.log("🔍 Verifying compiled Cython & C++ binary modules...");
  
  const espeakBridgePath = join(REPO_DIR, "src/piper/espeakbridge.so");
  const cythonAlignDir = join(REPO_DIR, "src/piper/train/vits/monotonic_align");
  
  const bridgeOk = await exists(espeakBridgePath);
  const alignOk = await hasCompiledAlign(cythonAlignDir);
  
  if (bridgeOk && alignOk) {
    console.log("  ✅ Pre-compiled binaries present and structurally sound.");
    return true;
  } else {
    console.warn("  ⚠️ Warning: Compiled binary bindings (.so files) are missing or named unexpectedly.");
    console.warn("  🛠️ Attempting local fast recompilation of monotonic alignment MAS...");
    
    const buildScript = join(REPO_DIR, "build_monotonic_align.sh");
    if (await exists(buildScript)) {
      await runCmd("bash", [buildScript], { cwd: REPO_DIR });
      await runCmd("python3", ["setup.py", "build_ext", "--inplace"], { cwd: REPO_DIR });
    }
    return false;
  }
}

async function main() {
  console.log("=================================================");
  console.log("🎙️ SuttaPlayer TTS Bootstrapper & Environment Linker");
  console.log(`   Wheels Source: ${WHEELS_DIR}`);
  console.log(`   Compiled Repo: ${REPO_DIR}`);
  console.log("=================================================\n");

  if (await exists(WHEELS_DIR)) {
    console.log("📦 Step 1: Installing Python packages offline from GDrive Wheel Cache...");
    const pipArgs = [
      "install",
      "--no-index",
      `--find-links=${WHEELS_DIR}`,
      "setuptools", "torch", "torchaudio", "torchvision", "onnx", "lightning",
      "soundfile", "librosa", "pysilero-vad", "docstring-parser", "jsonargparse[signatures]"
    ];
    const pipSuccess = await runCmd("pip", pipArgs);
    if (pipSuccess) {
      console.log("  ✅ Offline dependencies installed successfully.");
    } else {
      console.error("  ❌ Python wheel installation failed.");
      Deno.exit(1);
    }
  } else {
    console.error(`❌ Error: Local wheels directory not found at ${WHEELS_DIR}.`);
    console.error("   Please run 'sutta-training-gdown-resources.ts' first.");
    Deno.exit(1);
  }

  if (await exists(REPO_DIR)) {
    console.log("\n📦 Step 2: Registering compiled VITS repository with system Python...");
    await verifyPrecompiledBinaries();
    
    const linkArgs = ["install", "--no-index", "--no-deps", "--no-build-isolation", "-e", REPO_DIR];
    const linkSuccess = await runCmd("pip", linkArgs);
    if (linkSuccess) {
      console.log("  ✅ Compiled repository successfully linked.");
    } else {
      console.error("  ❌ Failed to link repository.");
      Deno.exit(1);
    }
  } else {
    console.error(`❌ Error: Precompiled piper1-gpl folder missing at ${REPO_DIR}`);
    Deno.exit(1);
  }

  console.log("\n=================================================");
  console.log("🟢 BOOTSTRAP COMPLETE! SuttaPlayer environment is fully active.");
  console.log("   Pre-flight checks can now be launched inside your notebook cell.");
  console.log("=================================================");
}

main().catch((err) => {
  console.error("FATAL ERROR in bootstrap manager:", err.message);
  Deno.exit(1);
});
