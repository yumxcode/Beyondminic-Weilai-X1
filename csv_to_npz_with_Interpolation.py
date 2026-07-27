"""Compatibility entry point for the integrated BeyondMimic motion converter."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = (
        Path(__file__).resolve().parent
        / "training"
        / "whole_body_tracking"
        / "scripts"
        / "csv_to_npz.py"
    )
    runpy.run_path(str(script), run_name="__main__")
