// sutta-training-gdown-resources.ts
// Version: 1.0.0
// Programmatic Cloud Resource Puller for SuttaPlayer training pipeline.
// Ingests training-stage-manifest.csv to pull large datasets, wheels, and checkpoints
// from Google Drive shared IDs, with automated decompression.

import { parseArgs } from "https://deno.land/std@0.224.0/cli/parse_args.ts";
import { ensureDir } from "https://deno.land/std@0.224.0/fs/ensure_dir.ts";
import { join } from "https://deno.land/std@0.224.0/path/mod.ts";

const flags = parseArgs(Deno.args, {
  string: ["i"],
  alias: { i: "input" },
});

if (!flags.input) {
  console.error("❌ Error: Missing input manifest parameter.");
  console.error("Usage: Deno run --allow-all sutta-training-gdown-resources.ts -i <path-to-manifest.csv>");
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

async function downloadResource(gdriveId: string, destPath: string) {
  console.log(`📥 Downloading GDrive ID: ${gdriveId} to ${destPath}...`);
  
  // Try gdown first (native Colab utility)
  const gdownSuccess = await runCommand("gdown", ["--id", gdriveId, "-O", destPath, "--confirm"]);
  if (gdownSuccess) {
    console.log(`  ✅ gdown download completed successfully.`);
    return true;
  }

  console.warn(`  ⚠️ gdown failed or missing. Falling back to wget...`);
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

  console.error(`  ❌ Failed to download resource with both gdown and wget.`);
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
  
  // Process lines (Skip headers, skip empty rows or section blocks like ,,,)
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
      downloadOk = await downloadResource(gdriveId, fullDestPath);
    }

    // Auto-extract if defined and download was successful
    if (downloadOk && extractTo) {
      console.log(`📦 Decompressing ${resource} to ${extractTo}...`);
      await ensureDir(extractTo);
      
      const tarSuccess = await runCommand("tar", ["-zxf", fullDestPath, "-C", extractTo]);
      if (tarSuccess) {
        console.log(`  ✅ Decompression finished successfully.`);
      } else {
        console.error(`  ❌ Failed to extract ${resource}. File might be incomplete.`);
      }
    }
  }

  console.log("\n🎯 All manifest items processed successfully. Pre-flight resources staged.");
}

main().catch((err) => {
  console.error("FATAL ERROR in resources manager:", err.message);
  Deno.exit(1);
});
