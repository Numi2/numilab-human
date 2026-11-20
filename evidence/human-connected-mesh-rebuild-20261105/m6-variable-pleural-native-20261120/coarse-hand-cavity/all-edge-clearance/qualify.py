"""Compare emitted material shapes, mass and accepted root certificates."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_bank_trials import emitted
from fit_hand_cavity import PARENT, quality, table

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    native = args.trial / "native-prefix"
    before, before_mass = emitted(args.baseline)
    after, after_mass = emitted(native)
    rows = table(PARENT.with_name(PARENT.name + ".tets.csv"))
    tets = np.array([[int(row[f"parent_node{k}"]) for k in range(4)] for row in rows])
    material_ids = np.array([int(row["material_index"]) for row in rows])
    before_quality = quality(before, tets)
    after_quality = quality(after, tets)
    by_material = {}
    for material in (15, 16):
        indices = np.flatnonzero(material_ids == material)
        prior = before_quality[indices]
        current = after_quality[indices]
        by_material[str(material)] = {
            "cells": len(indices),
            "baseline_minimum": float(prior.min()),
            "candidate_minimum": float(current.min()),
            "candidate_minimum_cell": int(indices[current.argmin()]),
            "cells_below_baseline_local_minimum": int((current < prior.min()).sum()),
            "min_relative_quality": float((current / prior).min()),
        }

    certificates = []
    with (native / "stderr.log").open() as stream:
        for line in stream:
            if line.startswith("matter_solver_domain_norms="):
                certificates.append(json.loads(line.partition("=")[2]))
    assert [r["root"] for r in certificates] == list(range(64))
    mechanical = max(r["nonvascular_norm"] / r["mechanical_threshold"] for r in certificates)
    vascular = max(r["vascular_norm"] / r["vascular_threshold"] for r in certificates)
    assert mechanical < 1 and vascular < 1
    assert math.isclose(math.fsum(before_mass), math.fsum(after_mass), rel_tol=2e-8)
    initial_matches = ((args.trial / "native/initial-skin-fem.csv").read_bytes() ==
                       (native / "initial-skin-fem.csv").read_bytes())
    assert initial_matches
    report = {
        "scope": "Emitted 28,632-cell paired skin and fat registration and 64 accepted roots.",
        "reference_tetrahedral_shape_metric": "six times absolute volume divided by cubed longest edge",
        "tet_shape_by_material": by_material,
        "skin_fat_initial_mapped_mass_kg": math.fsum(after_mass),
        "skin_fat_mass_delta_kg": math.fsum(after_mass) - math.fsum(before_mass),
        "matching_initial_skins_between_native_trials": initial_matches,
        "maximum_mechanical_residual_over_threshold": mechanical,
        "maximum_vascular_residual_over_threshold": vascular,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

if __name__ == "__main__":
    main()
