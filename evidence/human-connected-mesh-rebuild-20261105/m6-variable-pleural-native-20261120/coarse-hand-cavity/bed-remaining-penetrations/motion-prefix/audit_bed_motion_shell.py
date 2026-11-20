"""Audit every physical shell face at the retained native start and endpoint."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path

import numpy as np

from audit_bed_layer_surfaces import audit, install_for_float32, role_surfaces
from build_hand_trial import digest
from compare_bank_trials import emitted
from fit_hand_cavity import PARENT, table


def terminal_positions(log, before):
    samples = {}
    with log.open() as stream:
        for line in stream:
            if line.startswith("resting_FEM_node,object=BodyParts3D whole-body skin-fat bonded shell,"):
                fields = dict(field.split("=", 1) for field in next(csv.reader([line]))[2:])
                local = int(fields["local"])
                assert local not in samples
                samples[local] = [[float(fields[f"{frame}_{axis}"]) for axis in "xyz"]
                                  for frame in ("initial", "accepted")]
    assert list(sorted(samples)) == list(range(len(before)))
    start, end = np.asarray([samples[index] for index in range(len(before))],
                            dtype=np.float32).transpose(1, 0, 2)
    assert np.isfinite(end).all() and np.array_equal(start, before)
    return end.astype(np.float64)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    install_for_float32()
    cell_path = PARENT.with_name(PARENT.name + ".tets.csv")
    groups = role_surfaces(table(cell_path))
    start, _ = emitted(args.trial)
    log = args.trial / "stdout.log"
    end = terminal_positions(log, start)
    full_domain = np.ones(len(start), dtype=bool)
    start_pairs, face_counts = audit(start, groups, full_domain)
    end_pairs, _ = audit(end, groups, full_domain)
    new_pairs = {key: sorted(set(map(tuple, pairs)) - set(map(tuple, start_pairs.get(key, []))))
                 for key, pairs in end_pairs.items()}
    result = {
        "scope": "Exact Float32 physical shell self- and cross-layer triangle intersection "
                 "audit, initial and terminal accepted states",
        "native_log_sha256": digest(log),
        "connectivity_sha256": digest(cell_path),
        "faces_by_layer": face_counts,
        "initial_pairs_by_layer": start_pairs,
        "terminal_pairs_by_layer": end_pairs,
        "new_pairs_by_layer": new_pairs,
    }
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: {layer: len(pairs) for layer, pairs in result[key].items()}
                      for key in ("initial_pairs_by_layer", "terminal_pairs_by_layer", "new_pairs_by_layer")}))


if __name__ == "__main__":
    main()
