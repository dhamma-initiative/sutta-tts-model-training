// sutta-training-pruner.ts
// Version: 1.1.0
// Smart Local-First Checkpoint Pruner & sync manager for SuttaPlayer.
// 1. Calculates top-3 MOS and top-3 MEL models directly on Colab's NVMe drive.
// 2. Permanently purges weak models locally with zero Google Drive Trash creation.
// 3. Executes rsync --delete to mirror only tier-1 survivors to Google Drive.

import { parseArgs } from "https://deno.land/std@0.224.0/cli/parse_args.ts";
import { join } from "https://deno.land/std@0.224.0/path/mod.ts";

const flags = parseArgs(Deno.args, {
  string: ["src", "dest"],
  default: {
    src: "/content/lightning_logs/version_0/checkpoints",
    dest: "/content/drive/MyDrive/piper_training/checkpoints",
  },
});

const LOCAL_LOGS = flags.src;
const DRIVE_CKPTS = flags.dest;
const KEEP_TOP_K = 3;

/**
 * Prunes checkpoints inside a specific directory.
 * Keeps only the top_k models based on val_mos (higher is better)
 * and top_k models based on val_mel (lower is better), plus last.ckpt.
 */
async function pruneCheckpoints(dir: string, keepTopK: number) {
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
  } catch {
    return; // Directory might not exist yet during startup
  }

  const mosRanked = parsed.filter(p => p.mos !== undefined).sort((a, b) => b.mos! - a.mos!);
  const melRanked = parsed.filter(p => p.mel !== undefined).sort((a, b) => a.mel! - b.mel!);

  const keep = new Set<string>(["last.ckpt"]);
  mosRanked.slice(0, keepTopK).forEach(p => keep.add(p.name));
  melRanked.slice(0, keepTopK).forEach(p => keep.add(p.name));

  for (const p of parsed.filter(p => !keep.has(p.name))) {
    try {
      await Deno.remove(join(dir, p.name));
      console.log(`  🗑️ Local Pruned: ${p.name}`);
    } catch (err) {
      console.error(`  ⚠️ Failed to prune local ${p.name}: ${err.message}`);
    }
  }
}

/**
 * Syncs and prunes checkpoints using a Local-First sequence:
 * 1. Prune locally on volatile NVMe storage (instant, zero GDrive trash created).
 * 2. Rsync survivors (top-3 MOS, top-3 MEL, last.ckpt) up to Google Drive.
 * 3. Use rsync's --delete flag to safely clear replaced high-tier models on GDrive.
 */
async function syncAndPruneCheckpoints() {
  console.log("🔄 Running Local-Prune-First Sync Cycle...");

  // Step 1: Prune local volatile directory first
  await pruneCheckpoints(LOCAL_LOGS, KEEP_TOP_K);

  // Step 2: Sync survivors and mirror deletions to Google Drive
  // Using rsync --delete ensures old high-tier checkpoints are removed from GDrive
  // only when they are replaced by an even stronger model locally.
  const rsync = new Deno.Command("rsync", {
    args: [
      "-av",
      "--delete",
      "--include=*/",
      "--include=*.ckpt",
      "--exclude=*",
      `${LOCAL_LOGS}/`,
      `${DRIVE_CKPTS}/`,
    ],
    stdout: "piped",
    stderr: "piped",
  });

  const output = await rsync.output();
  if (output.code === 0) {
    console.log("  ✅ Local → Drive synchronized successfully.");
  } else {
    console.error(`  ⚠️ Sync failed: ${new TextDecoder().decode(output.stderr)}`);
  }
}

async function main() {
  console.log("🚀 SuttaPlayer Checkpoint Pruner & GDrive Sync Active.");
  console.log(`   Local NVMe Source: ${LOCAL_LOGS}`);
  console.log(`   GDrive Destination: ${DRIVE_CKPTS}`);
  console.log(`   Retention Strategy: Top ${KEEP_TOP_K} MOS + Top ${KEEP_TOP_K} MEL`);
  console.log("   Press Ctrl+C to terminate.\n");

  while (true) {
    try {
      await syncAndPruneCheckpoints();
      console.log("⏳ Waiting 30 seconds before next cycle...\n");
      await new Promise((res) => setTimeout(res, 30000));
    } catch (err) {
      console.error(`❌ Error: ${err.message}`);
      console.log("⏳ Retrying in 30 seconds...\n");
      await new Promise((res) => setTimeout(res, 30000));
    }
  }
}

// Graceful shutdown
Deno.addSignalListener("SIGINT", () => {
  console.log("\n🛑 Shutting down gracefully...");
  Deno.exit(0);
});

main();
