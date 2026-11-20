"""Compare exact solid-location witnesses for the repaired gluteal volumes."""
from argparse import ArgumentParser
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from audit import closed_surface, digest, rows, shell_faces
from audit_bed_layer_surfaces import owner


def surface(points, faces):
    assert closed_surface(points, faces)
    return (owner.prepare_float32_point_location_meters(points, faces),
            points[faces].min(axis=(0, 1)), points[faces].max(axis=(0, 1)))


def classify(points, model):
    prepared, lo, hi = model
    locations = {}
    for i in np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1)):
        query = tuple(Fraction.from_float(float(x)) for x in points[i])
        location = owner.point_location_prepared_indexed(query, prepared)["location"]
        if location != "outside":
            locations[int(i)] = location
    return locations


def comparison(before, after):
    return dict(before=dict(Counter(before.values())), after=dict(Counter(after.values())),
                newly_inside=[i for i, state in after.items()
                              if state == "inside" and before.get(i) != "inside"],
                cleared=[i for i, state in before.items()
                         if state == "inside" and i not in after],
                boundary_or_indeterminate=[i for i, state in after.items() if state != "inside"])


def shell_surface(cook, cells):
    with np.load(cook) as data:
        shell = data["nodes_current"]
    faces, normals = shell_faces(shell.astype(float), cells, len(shell))
    verts = shell[faces].astype(float)
    cross = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
    flip = np.einsum("ij,ij->i", cross, normals) < 0.
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return shell, faces, surface(shell, faces)


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("reference", "repaired", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--paired-shell", type=Path)
    args = parser.parse_args()
    cook_path = args.reference / "native-cooked-float32.npz"
    tet_path = args.reference / "native-candidate-128" / "shell-mass.csv.tets.csv"
    cells = [[int(row[f"parent_node{i}"]) for i in range(4)] for row in rows(tet_path)]
    before_shell, before_faces, before_model = shell_surface(cook_path, cells)
    if args.paired_shell is None:
        after_shell, after_model = before_shell, before_model
    else:
        after_shell, after_faces, after_model = shell_surface(args.paired_shell, cells)
        assert np.array_equal(np.sort(before_faces, axis=1), np.sort(after_faces, axis=1))
    shell_vertex_ids = np.unique(before_faces)
    states, info = {}, {}
    for label, path, shell, skin in (
            ("before", args.reference, before_shell, before_model),
            ("after", args.repaired, after_shell, after_model)):
        shell_points = shell[shell_vertex_ids]
        data = {}
        for bank in (29, 30):
            asset_path = path / f"gluteal-{bank}.npz"
            with np.load(asset_path) as asset:
                nodes, source_nodes, source_faces = (asset[key] for key in
                                                    ("nodes", "source_nodes", "source_faces"))
            data[bank] = nodes, surface(source_nodes, source_faces)
            info[f"{label}_{bank}"] = digest(asset_path)
        locations = {}
        for bank, (nodes, muscle) in data.items():
            locations[f"{bank}_volume_node_in_shell"] = classify(nodes, skin)
            locations[f"shell_boundary_node_in_{bank}"] = {
                int(shell_vertex_ids[i]): state for i, state in classify(shell_points, muscle).items()}
            other = 30 if bank == 29 else 29
            locations[f"{bank}_volume_node_in_{other}"] = classify(nodes, data[other][1])
        states[label] = locations
    results = {name: comparison(before, states["after"][name])
               for name, before in states["before"].items()}
    result = dict(geometry_sha256=info, shell_sha256=digest(cook_path),
                  connectivity_sha256=digest(tet_path), witnesses=results)
    if args.paired_shell is not None:
        result["paired_shell_sha256"] = digest(args.paired_shell)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    for name, counts in results.items():
        print(name, counts["before"], counts["after"], "new_inside", len(counts["newly_inside"]))


if __name__ == "__main__":
    main()
