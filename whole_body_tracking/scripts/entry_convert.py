#!/usr/bin/env python3
"""Gradmotion entry point for CSV→NPZ motion conversion.

Installs the whole_body_tracking extension then hands off to
``csv_to_npz.py`` via :func:`os.execv`.

Usage::

    python scripts/entry_convert.py \
        --input_file motions/dance.csv --input_fps 30 \
        --output_name dance_zui \
        --output_file motions/dance_zui.npz \
        --no_wandb --headless
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
    print(f"[entry] pip install -e {ext_path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ext_path])

    convert_script = os.path.join(SCRIPT_DIR, "csv_to_npz.py")
    cli_args = sys.argv[1:]
    print(f"[entry] launching {convert_script} {' '.join(cli_args)}")
    os.execv(sys.executable, [sys.executable, convert_script] + cli_args)


if __name__ == "__main__":
    main()
