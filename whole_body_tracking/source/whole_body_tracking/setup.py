"""Installation script for the 'whole_body_tracking' python package."""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation.
# EMPTY on purpose: the gradmotion IsaacLab image already provides every
# runtime dep (psutil, wandb 0.12.21, torch, ...), and the platform harness
# auto-runs `pip install -e` on this setup.py WITHOUT --no-deps — any entry
# here (e.g. onnxscript) cascades into a numpy 2.x upgrade that breaks
# isaac-sim's numpy 1.26.4 (TASK_20261105_423/427 both died this way).
# onnxscript is only needed for policy export, install it manually then.
INSTALL_REQUIRES = []

# Installation operation
setup(
    name="whole_body_tracking",
    packages=["whole_body_tracking"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.1",
        "Isaac Sim :: 4.0.0",
    ],
    zip_safe=False,
)
