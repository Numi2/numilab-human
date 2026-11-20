"""Map a stalled equilibrium solve back to the original material and donor."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from predict import ROOT, digest, read_table, read_model


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    _, _, world, rest, tets, _, _ = read_model()
    source = ROOT / "thickness-rebuild-034/compiled-reference/shell-mass.csv.tets.csv"
    rows = read_table(source)
    reference = rest[tets]
    edges = reference[:, 1:] - reference[:, :1]
    det = np.linalg.det(edges)
    size = np.linalg.norm(reference[:, :, None] - reference[:, None, :], axis=-1).max(axis=(1, 2))
    ratio = det / size**3
    summary = {}
    for run in args.runs:
        mesh = run / "seed.npz"
        with np.load(mesh, allow_pickle=False) as saved:
            deformation = saved["displacement_m"]
            jac = saved["deformation_j"]
            positions = world + deformation
            deformed = positions[tets]
            actual = np.linalg.det(deformed[:, 1:] - deformed[:, :1]) / det
            np.testing.assert_allclose(actual, jac, atol=1.e-10, rtol=0.)
            nodes = np.flatnonzero(saved["force_n"] > 0.)
        trace = json.loads((run / "trace.json").read_text())
        local = {}
        for direction, order in (("minimum", np.argsort(jac)[:5]),
                                 ("maximum", np.argsort(-jac)[:5])):
            local[direction] = [{"cell": int(i), "volume_nodes": tets[i].tolist(),
                                 "material": int(rows[i]["material_index"]),
                                 "donor": int(rows[i]["donor_body"]),
                                 "volume_ratio": float(actual[i]),
                                 "reference_det_over_diameter_cubed": float(ratio[i]),
                                 "reference_centroid_world_m": world[tets[i]].mean(axis=0).tolist(),
                                 "relative_node_motion_mm": (1.e3 * deformation[tets[i]]).tolist()}
                                for i in order]
        summary[run.name] = {
            "candidate_seed_sha256": digest(mesh),
            "stopped_at_load_fraction": trace[-1]["load_fraction"],
            "last_unmodified_force_moment_residual_max": trace[-1]["max_residual"],
            "current_shell_nodes_with_bed_reaction": len(nodes),
            "checkpoint": trace[-1]["outcome"],
            "volume_ratio_limits": local,
        }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
