#!/usr/bin/env python3
"""Gradmotion entry point for X1: CSV -> NPZ conversion then training.

Runs csv_to_npz.py (--robot x1) as a subprocess (it launches its own Isaac
Sim app), then hands off to rsl_rl/train.py via os.execv.

    python scripts/entry_x1_train.py \
        --task=Tracking-Flat-X1-v0 \
        --motion_csv motions/csv/dance1_subject3_x1.csv \
        --motion_file motions/dance1_subject3_x1.npz \
        --headless --num_envs 4096 --max_iterations 5000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def main() -> None:
    os.chdir(REPO_ROOT)

    # Split args: ours (--motion_csv/--npz_name) vs forwarded train args
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--motion_csv", required=True)
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--npz_name", default=None)
    parser.add_argument("--input_fps", type=int, default=30)
    parser.add_argument("--output_fps", type=int, default=50)
    known, train_args = parser.parse_known_args(argv)

    ext_path = os.path.join(REPO_ROOT, "source", "whole_body_tracking")
    print(f"[entry] pip install -e {ext_path} --no-deps")
    # --no-deps: the IsaacLab image already provides every runtime dependency;
    # letting pip resolve deps would upgrade numpy/scipy and break isaac-sim.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ext_path, "--no-deps"])

    name = known.npz_name or os.path.splitext(os.path.basename(known.motion_file))[0]
    convert_script = os.path.join(SCRIPT_DIR, "csv_to_npz.py")
    convert_cmd = [
        sys.executable, convert_script,
        "--robot", "x1",
        "--input_file", known.motion_csv,
        "--input_fps", str(known.input_fps),
        "--output_fps", str(known.output_fps),
        "--output_name", name,
        "--output_file", known.motion_file,
        "--no_wandb",
        "--headless",
    ]
    print(f"[entry] converting: {' '.join(convert_cmd)}")
    subprocess.check_call(convert_cmd)
    if not os.path.isfile(known.motion_file):
        raise FileNotFoundError(f"conversion did not produce {known.motion_file}")

    train_script = os.path.join(SCRIPT_DIR, "rsl_rl", "train.py")
    print(f"[entry] launching {train_script} {' '.join(train_args)}")
    os.execv(sys.executable, [sys.executable, train_script] + train_args)


if __name__ == "__main__":
    main()
