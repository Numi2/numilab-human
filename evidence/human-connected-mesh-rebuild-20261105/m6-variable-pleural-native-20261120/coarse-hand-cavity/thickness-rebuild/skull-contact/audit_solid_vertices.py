"""Check exact point containment for moved shell boundary vertices."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import monotonic

import numpy as np

from rebuild import AUDIT, HERE, TRIANGLE_ROW, digest, install_for_float32, table
from solid_skull import ClosedBone


def vertex_states(points, other, vertex_ids, identity, triangles):
    lower = triangles.min(axis=(0, 1))
    upper = triangles.max(axis=(0, 1))
    candidates = vertex_ids[np.all((points[vertex_ids] >= lower) & (points[vertex_ids] <= upper), axis=1)
                            | np.all((other[vertex_ids] >= lower) & (other[vertex_ids] <= upper), axis=1)]
    if len(candidates) == 0:
        return None
    try:
        closed = ClosedBone(identity, triangles)
    except ValueError as error:
        return {"closed": False, "in_box_vertices": len(candidates), "reason": str(error)}
    return {"closed": True, "in_box_vertices": len(candidates),
            "original": closed.locations(points, candidates),
            "candidate": closed.locations(other, candidates)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--body", type=int, default=23)
    parser.add_argument("--all-surface", action="store_true")
    args = parser.parse_args()
    install_for_float32()
    start = monotonic()
    trial = json.loads((args.candidate / "receipt.json").read_text())
    path = args.candidate / "rebuild.npz"
    assert digest(path) == trial["candidate_sha256"]
    data = dict(np.load(path, allow_pickle=False))
    source = HERE / "compiled-reference/shell-mass.csv.nodes.csv"
    receipt = json.loads((HERE / "compiled-reference/execution.json").read_text())
    assert digest(source) == receipt[source.name]["sha256"]
    rows = table(source)
    initial = np.array([[float(row[f"current_{axis}_m"]) for axis in "xyz"] for row in rows])
    midpoint = initial[data["edge_parent"]].mean(axis=1).astype(np.float32).astype(np.float64)
    a, b = data["interior_edge_parent"]
    t = float(data["interior_edge_fraction"])
    edge = ((1. - t) * initial[a] + t * initial[b]).astype(np.float32).astype(np.float64)
    initial = np.vstack((initial, midpoint, edge))
    candidate = data["world"].astype(np.float64)
    boundary = np.unique(data["surface"])
    changed = boundary[np.any(candidate[boundary] != initial[boundary], axis=1)]
    bone = HERE / "inputs/physical-bone-initial.bin"
    prior = json.loads((AUDIT / "exact-startup-crossings.json").read_text())
    bone_sha = digest(bone)
    assert bone_sha == prior["source"][bone.name]
    topology = np.memmap(bone, dtype=TRIANGLE_ROW, mode="r")
    bank = topology[topology["body"] == args.body]
    assert len(bank) > 0
    tested = boundary if args.all_surface else changed
    measurements = {}
    counts = {"baseline_inside": 0, "candidate_inside": 0, "new_inside": 0,
              "baseline_boundary": 0, "candidate_boundary": 0, "indeterminate": 0}
    for identity in np.unique(bank["bone"]):
        selected = bank[bank["bone"] == identity]["vertices"]
        measured = vertex_states(initial, candidate, tested, identity, selected)
        if measured is None:
            continue
        measurements[int(identity)] = measured
        if not measured["closed"]:
            counts["indeterminate"] += measured["in_box_vertices"]
            continue
        before, after = measured["original"], measured["candidate"]
        counts["baseline_inside"] += sum(value == "inside" for value in before.values())
        counts["candidate_inside"] += sum(value == "inside" for value in after.values())
        counts["new_inside"] += sum(value == "inside" and before.get(node) != "inside"
                                    for node, value in after.items())
        counts["baseline_boundary"] += sum(value == "boundary" for value in before.values())
        counts["candidate_boundary"] += sum(value == "boundary" for value in after.values())
        counts["indeterminate"] += sum(value == "indeterminate"
                                       for table in (before, after) for value in table.values())
    print(json.dumps({
        "scope": "Exact shell boundary vertex classification against separate closed bones in one physical bank",
        "candidate_sha256": trial["candidate_sha256"], "bone_sha256": bone_sha,
        "body": args.body, "changed_surface_vertices": len(changed),
        "tested_surface_vertices": len(tested), "includes_unchanged": args.all_surface,
        **counts, "individual_bones": measurements, "seconds": monotonic() - start,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
