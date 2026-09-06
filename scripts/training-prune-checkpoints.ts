// training-prune-checkpoints.ts
import { parseArgs } from "jsr:@std/cli/parse-args";
import { join } from "https://deno.land/std@0.224.0/path/mod.ts";

const flags = parseArgs(Deno.args, {
  string: ["src", "dest"],
  default: {
    src: "/content/piper1-gpl/lightning_logs/version_0/checkpoints",
    dest: "/content/drive/MyDrive/piper_training/checkpoints",
  },
});

const LOCAL_LOGS = flags.src;
const DRIVE_CKPTS = flags.dest;
const KEEP_TOP_K = 3;

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
    return; // dir may not exist yet
  }

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

async function syncAndPruneCheckpoints() {
  console.log("🔄 Syncing and pruning checkpoints (top 3 MOS + top 3 MEL)...");

  const rsync = new Deno.Command("rsync", {
    args: [
      "-av",
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
    console.log("  ✅ Local → Drive sync complete.");
  } else {
    console.error(`  ⚠️ Sync failed: ${new TextDecoder().decode(output.stderr)}`);
  }

  await pruneCheckpoints(LOCAL_LOGS, KEEP_TOP_K);
  await pruneCheckpoints(DRIVE_CKPTS, KEEP_TOP_K);
}

async function main() {
  console.log("🚀 Continuous checkpoint sync & prune started.");
  console.log(`   Local: ${LOCAL_LOGS}`);
  console.log(`   Remote: ${DRIVE_CKPTS}`);
  console.log("   Press Ctrl+C to stop.\n");

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

// Graceful shutdown on Ctrl+C
Deno.addSignalListener("SIGINT", () => {
  console.log("\n🛑 Shutting down gracefully...");
  Deno.exit(0);
});

main();