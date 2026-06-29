# -*- coding: utf-8 -*-
"""
Script 0: BIDMC Dataset Downloader
Dataset : BIDMC PPG and Respiration Dataset (PhysioNet – open access, CC BY 4.0)
Purpose : Download all 53 BIDMC records once. Run this BEFORE 01_train_hypoxia_model.py.
Run on  : PC / Google Colab
Output  : ./bidmc_data/  (contains .dat and .hea files for all records)
"""

import os
import sys
import signal

# Ensure stdout accepts Unicode on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import wfdb

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CFG = {
    'BIDMC_DB'   : 'bidmc/1.0.0',   # used by get_record_list
    'BIDMC_DL'   : 'bidmc',          # used by dl_files (no version suffix)
    'DATA_DIR'   : './bidmc_data',
    'NUM_RECORDS': 53,
}

# ==============================================================================
# DOWNLOAD FUNCTION
# ==============================================================================
def download_bidmc():
    """
    Download all 53 BIDMC records individually with live progress.
    Uses wfdb.dl_files per-record (avoids the silent bulk-ZIP approach).
    Skips files that are already present in DATA_DIR.
    """
    os.makedirs(CFG['DATA_DIR'], exist_ok=True)
    print(f"\nDownloading BIDMC dataset -> '{CFG['DATA_DIR']}' ...")

    # Use a known record list instead of fetching from PhysioNet
    # (avoids hanging on slow network connections)
    records = [f"bidmc{i:02d}" for i in range(1, CFG['NUM_RECORDS'] + 1)]
    print(f"  Total records to check: {len(records)}")

    downloaded, skipped, already = 0, 0, 0
    for rname in records:
        # Check which files are already present
        expected_files = [f"{rname}.dat", f"{rname}.hea",
                          f"{rname}n.dat", f"{rname}n.hea"]
        all_present = all(
            os.path.exists(os.path.join(CFG['DATA_DIR'], f))
            for f in expected_files
        )
        if all_present:
            already += 1
            continue
        try:
            print(f"  Downloading {rname} ...", end=" ", flush=True)
            wfdb.dl_files(CFG['BIDMC_DL'], CFG['DATA_DIR'],
                          expected_files,
                          keep_subdirs=False)
            downloaded += 1
            print(f"OK  ({already + downloaded}/{len(records)})")
        except Exception as exc:
            skipped += 1
            print(f"SKIPPED: {exc}")

    total_ready = already + downloaded
    print(f"\n  Download complete.")
    print(f"    Already present : {already}")
    print(f"    Newly downloaded: {downloaded}")
    print(f"    Skipped (errors): {skipped}")
    print(f"    Total ready     : {total_ready}/{len(records)}")

    if total_ready == 0:
        print("\n  ERROR: No records available. Check internet connection.")
        sys.exit(1)


def verify_dataset():
    """Quick verification that enough records exist for training."""
    data_dir = CFG['DATA_DIR']
    count = 0
    for i in range(1, CFG['NUM_RECORDS'] + 1):
        rname = f"bidmc{i:02d}"
        hea_file = os.path.join(data_dir, f"{rname}.hea")
        if os.path.exists(hea_file):
            count += 1
    print(f"\n  Verification: {count}/{CFG['NUM_RECORDS']} records found in '{data_dir}'")
    return count


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    print("=" * 65)
    print("  BIDMC Dataset Downloader")
    print("  Dataset : BIDMC PPG & Respiration (PhysioNet, open-access)")
    print("=" * 65)

    download_bidmc()
    count = verify_dataset()

    print("\n" + "=" * 65)
    if count > 0:
        print(f"  DONE. {count} records ready. You can now run:")
        print("    python 01_train_hypoxia_model.py")
    else:
        print("  FAILED. No records downloaded. Check your connection.")
    print("=" * 65)
