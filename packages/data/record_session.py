#!/usr/bin/env python3
"""
Automated recording controller & downloader for Project Aria Gen2 glasses.

Flow:
  1. Sends 'aria_gen2 recording start' to glasses
  2. Prompts user in terminal to press ENTER when done
  3. Sends 'aria_gen2 recording stop'
  4. Auto-detects device IP & UUID, downloads the VRS file into data/

Recording only: post-processing is a separate step (post_process.py).
--data-dir is relative to the working directory (default data/).

Usage:
    python record_session.py --name demo14
    python record_session.py --name demo14 --data-dir ../../sample_data
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, check=True):
    print(f"\n[EXEC] {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if check and res.returncode != 0:
        print(f"Error: Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    return res.returncode


def get_device_ip():
    """Extract IP of the connected Aria device."""
    res = subprocess.run(["aria_gen2", "device", "list"], capture_output=True, text=True)
    if res.returncode == 0:
        match = re.search(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", res.stdout)
        if match:
            return match.group(0)
    return None


def get_uuid_for_name(name):
    """Query aria_gen2 recording list and extract the UUID matching name."""
    res = subprocess.run(["aria_gen2", "recording", "list"], capture_output=True, text=True)
    if res.returncode != 0:
        return None
    for line in res.stdout.splitlines():
        if name in line:
            match = re.search(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                line,
                re.IGNORECASE,
            )
            if match:
                return match.group(0)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="Session recording name (e.g. demo14)")
    ap.add_argument("--profile", default="profile8", help="Aria sensor profile (default: profile8)")
    ap.add_argument("--data-dir", default="data", help="Destination folder for raw VRS (default: data)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    existing_vrs = set(data_dir.glob(f"{args.name}*.vrs"))

    # --- Step 1: Start Recording ---
    print("\n=======================================================")
    print(f"  Starting Aria Recording: {args.name}")
    print("=======================================================")
    run_cmd(["aria_gen2", "recording", "start", "--profile", args.profile, "--recording-name", args.name])

    # --- Step 2: User Prompt to Stop ---
    print("\n-------------------------------------------------------")
    print(f"  Recording '{args.name}' is IN PROGRESS.")
    print("  Perform your demonstration on the table.")
    print("  Press [ENTER] in this terminal when finished.")
    print("-------------------------------------------------------")
    input()

    # --- Step 3: Stop Recording ---
    print("Stopping recording...")
    run_cmd(["aria_gen2", "recording", "stop"])

    print("Waiting 3 seconds for device to finalize storage write...")
    time.sleep(3)

    # --- Step 4: Download Recording to data/ ---
    print("\n=======================================================")
    print(f"  Downloading VRS File into '{data_dir}/'")
    print("=======================================================")

    uuid = get_uuid_for_name(args.name)
    device_ip = get_device_ip()

    if not uuid:
        sys.exit(f"Error: Could not resolve UUID for recording '{args.name}' from device list.")

    print(f"Found UUID for '{args.name}': {uuid}")
    dl_cmd = ["aria_gen2", "recording", "download", "-u", uuid, "-o", str(data_dir)]
    if device_ip:
        print(f"Using Wi-Fi device IP: {device_ip}")
        dl_cmd.extend(["--device-ip", device_ip])

    run_cmd(dl_cmd, check=True)

    new_vrs = sorted(set(data_dir.glob(f"{args.name}*.vrs")) - existing_vrs, key=lambda p: p.stat().st_mtime)
    if not new_vrs:
        new_vrs = sorted(data_dir.glob(f"{args.name}*.vrs"), key=lambda p: p.stat().st_mtime)

    if not new_vrs or new_vrs[-1].stat().st_size == 0:
        sys.exit(f"Error: Download finished, but no VRS matching '{args.name}*.vrs' was found in '{data_dir}/'.")

    vrs_path = new_vrs[-1]
    print(f"\nRecording saved successfully to: {vrs_path}")


if __name__ == "__main__":
    main()