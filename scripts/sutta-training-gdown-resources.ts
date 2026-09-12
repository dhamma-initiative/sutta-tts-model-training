// sutta-training-gdown-resources.ts
// Version: 1.1.0
// Programmatic Cloud Resource Puller for SuttaPlayer training pipeline.
// Ingests training-stage-manifest.csv to pull large datasets, wheels, and checkpoints
// from Google Drive shared IDs, with automated decompression and permission warning helpers.

import { parseArgs } from "https://deno.land/std@0.224.0/cli/parse_args.ts";
import { ensureDir } from "https://deno.land/std@0.224.0/fs/ensure_dir.ts";
import { join } from "https://deno.land/std@0.224.0/path/mod.ts";

const flags = parseArgs(Deno.args, {
  string: ["i"],
  boolean: ["extract"],
  alias: { i: "input" },
});

if (!flags.input) {
  console.error("❌ Error: Missing input manifest parameter.");
  console.error("Usage: deno run --allow-all sutta-training-gdown-resources.ts -i <path-to-manifest.csv> [--extract]");
  Deno.exit(1);
}

const manifestPath = flags.input;

async function exists(path: string): Promise<boolean> {
  try {
    await Deno.stat(path);
    return true;
  } catch {
    return false;
  }
}

async function runCommand(cmd: string, args: string[]) {
  const command = new Deno.Command(cmd, {
    args,
    stdout: "inherit",
    stderr: "inherit",
  });
  const process = command.spawn();
  const status = await process.status;
  return status.success;
}

async function downloadResource(resource: string, gdriveId: string, destPath: string) {
  console.log(`📥 Downloading GDrive ID: ${gdriveId} to ${destPath}...`);
  
  // Try gdown first (native Colab utility) without --confirm
  const gdownSuccess = await runCommand("gdown", ["-O", destPath, gdriveId]);
  if (gdownSuccess) {
    console.log(`  ✅ gdown download completed successfully.`);
    return true;
  }

  console.warn(`  ⚠️ gdown failed or returned an error. Trying wget fallback...`);
  // Wget fallback with confirmation bypass for large files
  const wgetArgs = [
    "-O", destPath,
    `https://drive.usercontent.google.com/download?id=${gdriveId}&export=download&confirm=yes`
  ];
  const wgetSuccess = await runCommand("wget", wgetArgs);
  if (wgetSuccess) {
    console.log(`  ✅ wget download completed successfully.`);
    return true;
  }

  console.log("\n=================================================");
  console.log("🚨  GOOGLE DRIVE PERMISSION GATEWAY DETECTED!   ");
  console.log("=================================================");
  console.log(`❌ Failed to acquire: ${resource}`);
  console.log(`🔗 GDrive ID: ${gdriveId}`);
  console.log("\n👉 FIX ACTION Required:");
  console.log("   1. Log into the Google Account holding this file.");
  console.log("   2. Right-click the file on Google Drive.");
  console.log("   3. Select 'Share' -> 'Share'.");
  console.log("   4. Under 'General access', change 'Restricted' to:");
  console.log("      'Anyone with the link' (Viewer).");
  console.log("=================================================\n");
  return false;
}

async function main() {
  console.log("=================================================");
  console.log("🚀 SuttaPlayer GDown Cloud Resource Puller Active");
  console.log(`   Ingesting manifest: ${manifestPath}`);
  console.log("=================================================\n");

  if (!(await exists(manifestPath))) {
    console.error(`❌ Error: Manifest file not found at ${manifestPath}`);
    Deno.exit(1);
  }

  const content = await Deno.readTextFile(manifestPath);
  const lines = content.split(/\r?\n/);
  
  let rowIndex = 0;
  for (const line of lines) {
    rowIndex++;
    if (rowIndex === 1) continue; // Skip CSV headers
    
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith(",,,") || trimmed.replace(/,/g, "") === "") {
      continue; // Skip section gaps, blank lines, or trailing spacers
    }

    const parts = line.split(",").map(p => p.trim());
    if (parts.length < 4) continue;

    const [resource, destination, extractTo, gdriveId] = parts;
    if (!resource || !gdriveId || !destination) {
      console.warn(`  ⚠️ Skipping corrupt manifest row #${rowIndex}: ${line}`);
      continue;
    }

    const fullDestPath = join(destination, resource);
    await ensureDir(destination);

    console.log(`\n📦 Processing Resource: ${resource}`);
    console.log(`   Destination: ${fullDestPath}`);
    if (extractTo) console.log(`   Auto-Extract to: ${extractTo}`);

    // Check if resource already exists to prevent duplicate bandwidth downloads
    const filePresent = await exists(fullDestPath);
    let downloadOk = true;

    if (filePresent) {
      console.log(`  ✨ Resource already present. Skipping download.`);
    } else {
      downloadOk = await downloadResource(resource, gdriveId, fullDestPath);
    }

    if (!downloadOk) {
      console.error(`❌ Interrupted: Failed to acquire resource ${resource}`);
      Deno.exit(1);
    }

    // Auto-extract if defined and download was successful
    if (extractTo && flags.extract) {
      console.log(`📦 Decompressing ${resource} to ${extractTo}...`);
      await ensureDir(extractTo);
      
      const tarSuccess = await runCommand("tar", ["-zxf", fullDestPath, "-C", extractTo]);
      if (tarSuccess) {
        console.log(`  ✅ Decompression finished successfully.`);
      } else {
        console.error(`  ❌ Failed to extract ${resource}. File might be incomplete.`);
        Deno.exit(1);
      }
    }
  }

  console.log("\n🎯 All manifest items processed successfully. Pre-flight resources staged.");
}

main().catch((err) => {
  console.error("FATAL ERROR in resources manager:", err.message);
  Deno.exit(1);
});
