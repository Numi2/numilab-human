"""Locate boundary-preserving flips around measured layer-fit failures."""
from argparse import ArgumentParser
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from rebuild import HERE, determinant, digest, quality, table
from fit_hand_cavity import orient, replacements, typed_faces


def oriented_boundary(cells):
    facets = Counter()
    for tet in cells:
        for i, face in enumerate(combinations(tet, 3)):
            inversions = sum(a > b for a, b in combinations(face, 2))
            facets[tuple(sorted(face))] += (-1 if (i + inversions) % 2 else 1)
    return {face: sign for face, sign in facets.items() if sign != 0}


def positions(nodes, original_count, edges, frame):
    xyz = np.array([[float(row[f"{frame}_{axis}_m"]) for axis in "xyz"] for row in nodes])
    assert len(xyz) == original_count
    mids = xyz[edges].mean(axis=1)
    if frame == "current":
        mids = mids.astype(np.float32).astype(np.float64)
    return np.concatenate((xyz, mids))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    report = json.loads((args.trial / "receipt.json").read_text())
    path = args.trial / "rebuild.npz"
    assert digest(path) == report["candidate_sha256"]
    mesh = dict(np.load(path, allow_pickle=False))
    reference = HERE / "compiled-reference"
    nodes = table(reference / "shell-mass.csv.nodes.csv")
    labels = table(reference / "shell-mass.csv.labels.csv")
    parent = mesh["parent"]
    ancestry = np.array([[int(row[key]) for key in
                          ("material_index", "geometric_donor_body", "anatomical_region_id",
                           "stable_source_face_id")] for row in labels])
    centroid_region = ancestry[parent]
    _, owner = np.unique(centroid_region, axis=0, return_inverse=True)
    current = mesh["world"].astype(np.float64)
    rest = mesh["rest"]
    cells = [tuple(map(int, row)) for row in mesh["elements"]]
    original_pose = positions(nodes, len(nodes), mesh["edge_parent"], "current")
    original_rest = positions(nodes, len(nodes), mesh["edge_parent"], "rest")
    sampled = Counter()
    for entry in report["fit"]:
        for rejected in entry["geometric_witnesses"]:
            for values in rejected["frames"].values():
                for child, original, _ in values["worst_children"]:
                    assert parent[child] == original
                    sampled[child] += 1
    if not sampled:
        raise ValueError("requires a fit receipt with explicit limiting children")
    blocked_parents = Counter()
    for child, count in sampled.items():
        blocked_parents[int(parent[child])] += count

    retained = []
    rejected = Counter()
    for old, proposed in replacements(cells, owner, set(sampled)):
        old_cells = [cells[i] for i in old]
        children = orient(original_rest, proposed)
        if (not children or typed_faces(children) is None
                or oriented_boundary(children) != oriented_boundary(old_cells)):
            rejected["boundary"] += 1
            continue
        metrics = {}
        for key, pts in (("base_rest", original_rest), ("base_world", original_pose),
                         ("fit_rest", rest), ("fit_world", current)):
            before, after = determinant(pts, old_cells), determinant(pts, children)
            if after.min() <= 0 or not np.isclose(before.sum(), after.sum(), rtol=1.e-9, atol=1.e-20):
                rejected[key + "_volume"] += 1
                break
            quality_before, quality_after = quality(pts, old_cells).min(), quality(pts, children).min()
            if quality_after <= quality_before:
                rejected[key + "_shape"] += 1
                break
            metrics[key] = [float(quality_before), float(quality_after)]
        else:
            retained.append({"old_children": old, "old_parents": [int(parent[i]) for i in old],
                             "new_children": children, "geometry": metrics,
                             "witness_count": sum(sampled[i] for i in old),
                             "material_donor_region_face": centroid_region[old[0]].tolist()})
    retained.sort(key=lambda row: (-row["witness_count"], row["old_children"]))
    summary = {
        "input": report["candidate_sha256"], "source_label_sha256": digest(reference / "shell-mass.csv.labels.csv"),
        "typed_boundaries_preserved": True, "touched_outside_geometry": False,
        "screen_only": True, "limiting_children": sampled.most_common(), "limiting_parents": blocked_parents.most_common(),
        "rejected": dict(rejected), "retained": retained,
        "dominant_parent_covered": any(blocked_parents.most_common(1)[0][0] in row["old_parents"] for row in retained),
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
