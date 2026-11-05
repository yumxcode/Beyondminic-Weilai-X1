#!/usr/bin/env python3
"""Gradmotion entry point for F1: install package then launch training.

The motion npz is prebuilt locally (tools_csv_to_npz_f1_local.py — the
container image has a numpy 1.x/2.x split that breaks the Isaac-side
converter, same as the X1 flow since v10).

    python scripts/entry_f1_train.py \
        --task=Tracking-Flat-F1-v0 \
        --motion_file motions/dance_f1_train.npz \
        --headless --num_envs 4096 --max_iterations 5000
"""

from __future__ import annotations

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def main() -> None:
    os.chdir(REPO_ROOT)

    ext_path = os.path.join(REPO_ROOT, "source", "whole_body_tracking")
    print(f"[entry] pip install -e {ext_path} --no-deps")
    # --no-deps: the IsaacLab image already provides every runtime dependency;
    # letting pip resolve deps would upgrade numpy/scipy and break isaac-sim.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ext_path, "--no-deps"])

    train_args = [a for a in sys.argv[1:]]
    if not any(a.startswith("--motion_file") for a in train_args):
        raise SystemExit("entry_f1_train.py requires --motion_file (prebuilt npz)")

    train_py = os.path.join(SCRIPT_DIR, "rsl_rl", "train.py")
    print(f"[entry] exec train.py {' '.join(train_args)}")
    os.execv(sys.executable, [sys.executable, train_py] + train_args)


if __name__ == "__main__":
    main()
