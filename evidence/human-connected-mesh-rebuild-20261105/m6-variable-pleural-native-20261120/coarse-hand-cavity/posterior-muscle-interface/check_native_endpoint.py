"""Check native skin/fat cell validity, mass and shell intersections at captured endpoints."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from audit import digest, rows
from audit_bed_layer_surfaces import audit, install_for_float32, role_surfaces
from audit_bed_motion_shell import terminal_positions
from extension import determinant, quality


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("run", "reference", "output"):
        p.add_argument(name, type=Path)
    cfg = p.parse_args()
    install_for_float32()
    nodes_file = cfg.run / "shell-mass.csv.nodes.csv"
    cells_file = cfg.run / "shell-mass.csv.tets.csv"
    log = cfg.run / "stdout.log"
    r = rows(cells_file)
    cells = np.array([[int(v[f"parent_node{i}"]) for i in range(4)] for v in r])
    layers = role_surfaces(r)

    def pose(path):
        return np.array([[float(row[f"current_{axis}_m"]) for axis in "xyz"]
                         for row in rows(path)], dtype=np.float32)

    start = pose(nodes_file)
    control = pose(cfg.reference / nodes_file.name)
    end = terminal_positions(log, start)
    layer_rows = len(start) // 3
    assert len(start) == 3 * layer_rows
    assert np.array_equal(start[:2 * layer_rows], control[:2 * layer_rows])
    assert np.array_equal(start[:, :2], control[:, :2])
    original = determinant(start.astype(float), cells)
    whole_shell = np.ones(len(start), dtype=bool)
    outcome = {
        "native_log_sha256": digest(log), "native_mesh_sha256": digest(nodes_file),
        "connectivity_sha256": digest(cells_file),
        "changed_deep_nodes": int(np.any(start != control, axis=1).sum()),
        "outer_and_bonded_interface_start_unchanged": True,
        "endpoints": {},
    }
    for condition, points in (("initial", start), ("terminal", end)):
        points = points.astype(float)
        det = determinant(points, cells)
        q = quality(points, cells, det)
        pairs, _ = audit(points, layers, whole_shell)
        outcome["endpoints"][condition] = {
            "inverted_or_degenerate_cells": int((det <= 0.).sum()),
            "cells_below_quality_floor": int((q < 1e-5).sum()),
            "minimum_normalized_determinant": float(q.min()),
            "minimum_determinant_ratio_to_start": float((det / original).min()),
            "shell_crossing_pairs": pairs,
        }
    with cfg.output.open("x") as f:
        json.dump(outcome, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(outcome["endpoints"]))


if __name__ == "__main__":
    main()
