#!/usr/bin/env python3
"""Gradmotion entry point.

Installs the whole_body_tracking extension via ``pip install -e`` then hands
off to ``rsl_rl/train.py`` via :func:`os.execv` so that Isaac Lab's
``AppLauncher`` sees a clean process.

All CLI arguments are forwarded to train.py, e.g.::

    python scripts/entry_train.py \
        --task=Tracking-Flat-G1-v0 \
        --motion_file motions/dance_zui.npz \
        --headless --num_envs 4096
"""

from __future__ import annotations

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def main() -> None:
    # Change working directory to whole_body_tracking/ so relative paths resolve
    os.chdir(REPO_ROOT)

    # Step 1 — install the extension (creates an importable egg-link)
    ext_path = os.path.join(REPO_ROOT, "source", "whole_body_tracking")
    print(f"[entry] pip install -e {ext_path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ext_path])

    # Step 2 — hand off to train.py (replaces this process)
    train_script = os.path.join(SCRIPT_DIR, "rsl_rl", "train.py")
    cli_args = sys.argv[1:]
    print(f"[entry] launching {train_script} {' '.join(cli_args)}")
    os.execv(sys.executable, [sys.executable, train_script] + cli_args)


if __name__ == "__main__":
    main()
