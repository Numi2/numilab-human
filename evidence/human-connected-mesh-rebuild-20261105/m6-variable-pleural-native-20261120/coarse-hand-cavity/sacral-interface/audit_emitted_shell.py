"""Compare the candidate reference surfaces with the native physical shell endpoints."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from audit import digest, rows
from audit_bed_layer_surfaces import install_for_float32, role_surfaces
from audit_bed_motion_shell import terminal_positions
from fit_muscle_interface import exact_pairs, prepare_target


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("reference", "run", "candidate", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    install_for_float32()
    node_path = args.run / "shell-mass.csv.nodes.csv"
    cell_path = args.run / "shell-mass.csv.tets.csv"
    log_path = args.run / "stdout.log"
    layers = role_surfaces(rows(cell_path))
    with np.load(args.reference) as source:
        prepared = source["nodes_current"]
    native = np.array([[float(row[f"current_{axis}_m"]) for axis in "xyz"]
                       for row in rows(node_path)], dtype=np.float32)
    assert native.shape == prepared.shape
    terminal = terminal_positions(log_path, native).astype(np.float32)
    result = {
        "native_node_sha256": digest(node_path),
        "native_connectivity_sha256": digest(cell_path),
        "native_log_sha256": digest(log_path),
        "prepared_shell_sha256": digest(args.reference),
        "reference_gluteal_sha256": {},
        "maximum_prepared_to_initial_shell_distance_m": float(np.linalg.norm(
            native.astype(float) - prepared.astype(float), axis=1).max()),
        "maximum_native_endpoint_shell_displacement_m": float(np.linalg.norm(
            terminal.astype(float) - native.astype(float), axis=1).max()),
        "pairs": {},
    }
    for pose, points in (("native_initial", native), ("native_terminal", terminal)):
        domains = {}
        for layer, faces in sorted(layers.items()):
            indices = np.arange(len(faces))
            domains[layer] = indices, prepare_target(points, np.array(faces))
        by_side = {}
        for bank in (29, 30):
            path = args.candidate / f"gluteal-{bank}.npz"
            result["reference_gluteal_sha256"][bank] = digest(path)
            with np.load(path) as candidate:
                by_side[bank] = sorted(exact_pairs(
                    candidate["source_nodes"], candidate["source_faces"], domains))
        result["pairs"][pose] = by_side
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print({pose: {bank: len(witness) for bank, witness in banks.items()}
           for pose, banks in result["pairs"].items()})


if __name__ == "__main__":
    main()
