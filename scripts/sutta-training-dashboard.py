# =================================================================
# SUTTAPLAYER LIVE TELEMETRY & PLAYBACK DASHBOARD (v1.1.0)
# =================================================================
# This script monitors your uat_metrics.csv, syncs data to Google Sheets,
# and hosts the HTML5 local audio playback widgets in your browser.
# =================================================================

import os
import sys
import time
import pandas as pd
import gspread

# 🔐 Robust Multi-Environment Google Account Authentication
print("🔐 Authenticating Google Account for Sheets Sync...")
creds = None
try:
    # 1. Attempt to load cached Colab credentials from the VM's local disk
    from google.auth import default
    creds, _ = default()
    gc = gspread.authorize(creds)
    print("  🟢 [SUCCESS] Loaded cached Google credentials from VM environment.")
except Exception as cache_err:
    # 2. If cached credentials fail, check if we are in an interactive IPython cell to run popup auth
    print("  ⚠️ No cached credentials found on disk.")
    try:
        from google.colab import auth
        auth.authenticate_user()
        from google.auth import default
        creds, _ = default()
        gc = gspread.authorize(creds)
        print("  🟢 [SUCCESS] Interactive user authentication completed.")
    except Exception as cell_err:
        print("\n❌ [CRITICAL AUTH FAILURE]")
        print("   This script is running as a background subprocess and cannot trigger the login popup.")
        print("\n👉 FIX ACTION:")
        print("   Create a native Google Colab cell and execute this command first:")
        print("       from google.colab import auth")
        print("       auth.authenticate_user()")
        print("   Once authenticated in the cell, re-run this script in the terminal!")
        sys.exit(1)

# Paths
PIPER_TRAINING = "/content/drive/MyDrive/piper_training"
METRICS_CSV = f"{PIPER_TRAINING}/uat_metrics.csv"
sheet_name = "SuttaPlayer_UAT_Convergence"

try:
    # Open or create Google Sheet
    try:
        sh = gc.open(sheet_name)
    except Exception:
        sh = gc.create(sheet_name)
        print(f"  📊 Created new Google Sheet: '{sheet_name}'")

    ws = sh.get_worksheet(0)
    last_row = len(ws.col_values(1))

    print(f"\n📊 Keep-Alive and Sheets sync active! Listening to '{METRICS_CSV}'...")
    while True:
        if os.path.exists(METRICS_CSV):
            try:
                df = pd.read_csv(METRICS_CSV)
                if len(df) > last_row:
                    new_data = df.iloc[last_row:].values.tolist()
                    for row in new_data:
                        ws.append_row(row)
                        print(f"  [Sheet Sync] Logged Epoch {row[2]} to dashboard!")
                    last_row = len(df)
            except Exception as write_err:
                # Silently catch brief file-read conflicts during active validation writes
                pass

        time.sleep(15) # Quick sync loops
except KeyboardInterrupt:
    print("\n⏹️ Sync stopped gracefully.")
