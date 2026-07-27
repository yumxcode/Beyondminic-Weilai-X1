import os
from pathlib import Path


def _find_asset_root() -> Path:
    override = os.environ.get("BEYONDMIMIC_ASSET_DIR")
    if override:
        return Path(override).expanduser().resolve()
    package_assets = Path(__file__).resolve().parent
    if (package_assets / "unitree_description").is_dir():
        return package_assets
    for parent in package_assets.parents:
        if (parent / "unitree_description" / "urdf" / "g1" / "main.urdf").is_file():
            return parent
    return package_assets


# The integrated project keeps unitree_description at repository root. The
# environment variable remains available for a separately installed asset pack.
ASSET_DIR = os.fspath(_find_asset_root())
