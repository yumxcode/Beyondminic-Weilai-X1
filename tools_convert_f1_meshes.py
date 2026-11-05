#!/usr/bin/env python3
"""Convert F1 ASCII STL meshes to binary + decimate for MuJoCo/Isaac use.

MuJoCo requires binary STL with <=200k faces. F1 visual meshes are ASCII with
up to ~660k faces (66 MB). This script converts every visual STL to binary and
decimates to <= MAX_FACES, writing to F1_29DOF/meshes/sim/<name>.stl.
Vendor collision meshes (already binary) are copied over unchanged.
Upper-body visuals come from meshes/x1 (binary, MuJoCo-compatible) and are
referenced directly.

The mesh geometry itself is never modified beyond decimation (same frame).
"""
from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent
F1 = REPO / "F1_29DOF"
SIM = F1 / "meshes/sim"
MAX_FACES = 100_000


def decimate(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= max_faces:
        return mesh
    ratio = max_faces / len(mesh.faces)
    mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
    # fallback if fast path unavailable
    if len(mesh.faces) > max_faces:
        import fast_simplification
        v, f = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=1 - ratio)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return mesh


def main() -> None:
    SIM.mkdir(parents=True, exist_ok=True)
    report = []
    # leg visuals + base
    for stl in sorted((F1 / "meshes/visual").glob("*.stl")):
        m = trimesh.load(stl, process=False)
        n0 = len(m.faces)
        m2 = decimate(m, MAX_FACES)
        out = SIM / stl.name
        m2.export(out)
        report.append((stl.name, n0, len(m2.faces), stl.stat().st_size, out.stat().st_size))
    # collision meshes (binary already)
    for stl in sorted((F1 / "meshes/collision").rglob("*.stl")):
        out = SIM / stl.name
        shutil.copy(stl, out)
        m = trimesh.load(stl, process=False)
        report.append((stl.name, len(m.faces), len(m.faces), stl.stat().st_size, out.stat().st_size))
    print(f"{'mesh':44s} {'faces':>12s} {'->':>8s} {'MB':>8s} {'->':>7s}")
    for name, n0, n1, s0, s1 in report:
        print(f"{name:44s} {n0:12d} {n1:8d} {s0/1e6:8.2f} {s1/1e6:7.2f}")


if __name__ == "__main__":
    main()
