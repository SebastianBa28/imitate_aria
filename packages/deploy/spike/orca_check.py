"""Milestone 0, step 4: can this machine talk to the ORCA hand? Reads joint states; moves nothing.

    python spike/orca_check.py [--port /dev/tty.usbserial-XXXX]
Needs calibration.yaml in config/orca/right/ (copy it from the rwr_system machine): without it, orca_core
cannot convert motor positions to joint angles, and this script stops instead of calibrating.
"""
import argparse
import glob
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "orca" / "right"

ap = argparse.ArgumentParser()
ap.add_argument("--port", default=None, help="serial port (default: the port in config.yaml, 'auto')")
args = ap.parse_args()

print("serial ports:", glob.glob("/dev/tty.usbserial*") + glob.glob("/dev/ttyUSB*") + glob.glob("/dev/cu.usbserial*") or "none")
if not (CONFIG_DIR / "calibration.yaml").exists():
    sys.exit(f"STOP: no calibration.yaml in {CONFIG_DIR}. Copy this hand's calibration there first.")

config_path = CONFIG_DIR / "config.yaml"
if args.port:  # don't edit the tracked config: use a temporary copy with the port set
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(CONFIG_DIR / "calibration.yaml", tmp / "calibration.yaml")
    conf = yaml.safe_load(config_path.read_text())
    conf["port"] = args.port
    (tmp / "config.yaml").write_text(yaml.safe_dump(conf))
    config_path = tmp / "config.yaml"

from deploy.hw.orca import OrcaHandV2  # noqa: E402

hand = OrcaHandV2(config_path)
hand.connect()
try:
    t0 = time.time()
    for _ in range(50):
        pos, vel = hand.read_joint_state()
    dt = time.time() - t0
    print(f"OK: {50 / dt:.0f} reads/s")
    for name, p in zip(hand.joint_ids, pos):
        print(f"  {name:11s} {p:7.1f} deg")
    print("unknown joints:", int(np.isnan(pos).sum()))
finally:
    hand.disconnect()
