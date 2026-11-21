"""Measure why an altered shell cannot reuse exact refinement ancestry."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from audit import digest, rows
from cavity import determinant
from map_cavity import overlap


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("nodes", "proposal", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    x = np.array([[r[f"rest_{axis}_m"] for axis in "xyz"] for r in rows(a.nodes)], float)
    candidate = json.loads(a.proposal.read_text())
    report = []
    for edit in candidate["selected"]:
        for patch in edit["cavities"]:
            old = np.asarray(patch["original"])
            new = np.asarray(patch["replacement"])
            vertices = x[np.unique(old)]
            scale = float(np.ptp(vertices, axis=0).max())
            local = (x-vertices.mean(axis=0))/scale
            old_volume = determinant(local, old)/6.
            new_volume = determinant(local, new)/6.
            intersection = np.array([[overlap(local[o],local[n]) for n in new] for o in old])
            covered = intersection.sum(axis=1)
            covered_new = intersection.sum(axis=0)
            report.append({
                "diagonal": edit["edge"],
                "typed_material_donor_region": list(map(int, patch["typed_bank"])),
                "rows": patch["retired"],
                "old_reference_volume_m3": float(old_volume.sum()*scale**3),
                "new_reference_volume_m3": float(new_volume.sum()*scale**3),
                "net_volume_fraction_change": float(new_volume.sum()/old_volume.sum()-1.),
                "coverage_fractions_old": (covered/old_volume).tolist(),
                "coverage_fractions_new": (covered_new/new_volume).tolist(),
                "old_reference_volume_outside_new_shell_m3": float((old_volume-covered).sum()*scale**3),
                "new_reference_volume_outside_old_shell_m3": float((new_volume-covered_new).sum()*scale**3),
            })
    # The existing exact-refinement compiler requires per-parent 1e-6 volume
    # closure. For unequal aggregate domains there is no exact volume map.
    failures = sum(abs(r["net_volume_fraction_change"]) > 1.e-6 for r in report)
    result = {
        "proposal_sha256": digest(a.proposal),
        "reference_nodes_sha256": digest(a.nodes),
        "cavities": report,
        "boundary_changed_material_cavities": len(report),
        "volume_mismatch_exceeding_exact_refinement_tolerance": failures,
        "maximum_absolute_cavity_volume_change_fraction": max(abs(r["net_volume_fraction_change"]) for r in report),
        "new_reference_volume_outside_old_shell_m3": sum(r["new_reference_volume_outside_old_shell_m3"] for r in report),
        "old_reference_volume_outside_new_shell_m3": sum(r["old_reference_volume_outside_new_shell_m3"] for r in report),
    }
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "cavities"})


if __name__ == "__main__":
    main()
