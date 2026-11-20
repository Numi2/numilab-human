"""Recover typed surface bindings and screen the opposing gluteus on each donor chord."""
from argparse import ArgumentParser
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from audit import BONE, digest
from audit_bed_layer_surfaces import owner
from extension import barycentric
from verify_containment import surface


def projected_hit(xy, triangle):
    triangle = triangle.astype(float)
    weights = barycentric(xy.reshape(1, 2), triangle[:, :2])[0]
    assert np.all(weights >= -1e-10) and abs(weights.sum() - 1.) < 1e-12
    point = weights @ triangle
    assert np.linalg.norm(point[:2] - xy) <= 1e-12
    return weights, point


def exact_point(x):
    return tuple(Fraction.from_float(float(a)) for a in x)


def opposing_witnesses(origin, end, triangle_records, model):
    p, q = exact_point(origin), exact_point(end)
    start = owner.point_location_prepared_indexed(p, model[0])["location"]
    lo, hi = np.minimum(origin, end), np.maximum(origin, end)
    use = np.flatnonzero(np.all((triangle_records.min(axis=1) <= hi)
                               & (triangle_records.max(axis=1) >= lo), axis=1))
    witnesses = []
    for i in use:
        triangle = tuple(exact_point(x) for x in triangle_records[i])
        normal = owner._cross(owner._sub(triangle[1], triangle[0]),
                              owner._sub(triangle[2], triangle[0]))
        if owner._segment_triangle(p, q, triangle, normal):
            witnesses.append(int(i))
    return start, witnesses


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("donors", "geometry", "bone", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.donors.read_text())
    rows = evidence["repaired"]["plus_sacrum"]["banks"]
    bones = np.memmap(args.bone, mode="r", dtype=BONE)
    muscles, surfaces, boundary_nodes, hashes = {}, {}, {}, {}
    for bank in rows:
        muscle_id = bank["muscle"]
        path = args.geometry / f"gluteal-{muscle_id}.npz"
        assert digest(path) == bank["source_hash"]
        with np.load(path) as asset:
            nodes, points, faces = (asset[key] for key in ("nodes", "source_nodes", "source_faces"))
        muscles[muscle_id] = points[faces].astype(float)
        surfaces[muscle_id] = surface(points, faces)
        own = {tuple(p): i for i, p in enumerate(nodes)}
        assert len(own) == len(nodes)
        boundary_nodes[muscle_id] = np.array([own[tuple(p)] for p in points])[faces]
        hashes[str(muscle_id)] = bank["source_hash"]
    bound = []
    for bank in rows:
        own = bank["muscle"]
        other = 30 if own == 29 else 29
        for row in bank["rows"]:
            indices = np.flatnonzero((bones["body"] == row["bone_body"])
                                     & (bones["bone"] == row["bone"])
                                     & (bones["triangle"] == row["bone_triangle"]))
            assert len(indices) == 1
            start = np.array(row["quadrature_point_m"])
            fat_weights = [1. / 3.] * 3
            entry_weights, entry = projected_hit(start[:2], muscles[own][row["entry_face"]])
            rigid_weights, rigid_point = projected_hit(start[:2], bones[indices[0]]["vertices"])
            # The original screen performs some triangle subtractions in
            # Float32. Check its coordinates within its existing chord margin,
            # retaining the independent double-precision reconstruction.
            initial_gap = float(entry[2] - start[2])
            donor_depth = float(rigid_point[2] - entry[2])
            assert min(initial_gap, donor_depth) > 1e-7
            assert abs(initial_gap - row["initial_gap_m"]) <= 1e-7
            assert abs(donor_depth - row["gluteal_donor_depth_m"]) <= 1e-7
            opposing_start, opposing_faces = opposing_witnesses(
                start, rigid_point, muscles[other], surfaces[other])
            bound.append(dict(row, fat_barycentric=fat_weights,
                              gluteal_entry_nodes=boundary_nodes[own][row["entry_face"]].tolist(),
                              gluteal_entry_barycentric=entry_weights.tolist(),
                              skeletal_barycentric=rigid_weights.tolist(),
                              reconstructed_gap_m=initial_gap,
                              reconstructed_donor_depth_m=donor_depth,
                              opposing_gluteal_start=opposing_start,
                              opposing_gluteal_faces=opposing_faces))
    kept = [r for r in bound if r["opposing_gluteal_start"] == "outside"
            and not r["opposing_gluteal_faces"]]
    summary = dict(rows=len(bound), without_opposing_gluteal_obstruction=len(kept),
                   sacral_rows_without_opposing_gluteal_obstruction=sum(r["bone_body"] == 1 for r in kept),
                   upper_sum_projected_area_cm2=1e4 * sum(r["area_m2"] for r in kept))
    result = dict(donor_screen_sha256=digest(args.donors), geometry_sha256=hashes,
                  bone_sha256=digest(args.bone), summary=summary, rows=bound)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
