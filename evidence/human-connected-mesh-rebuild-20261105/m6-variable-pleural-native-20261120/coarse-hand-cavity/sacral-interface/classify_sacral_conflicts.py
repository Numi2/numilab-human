"""Split gluteal intersection segments at the exact registered sacral boundary."""
from argparse import ArgumentParser
from collections import Counter
from fractions import Fraction
from pathlib import Path
import json

import numpy as np

from audit import BONE, digest
from audit_bed_layer_surfaces import owner
from verify_containment import surface


def exact_xyz(point):
    return tuple(Fraction.from_float(float(x)) for x in point)


def classify_segment(a, b, triangles, lo, hi, prepared):
    # Widen the float bounds outward to contain the exact rational endpoints.
    endpoints = np.array([a, b], dtype=float)
    lower = np.nextafter(endpoints.min(axis=0), -np.inf)
    upper = np.nextafter(endpoints.max(axis=0), np.inf)
    selected = np.flatnonzero(np.all((hi >= lower) & (lo <= upper), axis=1))
    axis = max(range(3), key=lambda j: abs(b[j] - a[j]))
    delta = tuple(y - x for x, y in zip(a, b))
    cuts = {Fraction(0), Fraction(1)}
    for i in selected:
        tri, normal = triangles[i]
        for point in owner._segment_triangle(a, b, tri, normal):
            value = (point[axis] - a[axis]) / delta[axis]
            assert 0 <= value <= 1
            cuts.add(value)
    intervals = []
    for first, second in zip(sorted(cuts)[:-1], sorted(cuts)[1:]):
        middle = (first + second) / 2
        point = tuple(x + middle * step for x, step in zip(a, delta))
        item = owner.point_location_prepared_indexed(point, prepared)
        assert item["location"] != "indeterminate"
        intervals.append((item["location"], float(second - first)))
    length = float(np.linalg.norm(np.array(delta, dtype=float)))
    fractions = Counter()
    for label, fraction in intervals:
        fractions[label] += fraction
    assert abs(sum(fractions.values()) - 1) < 1e-12
    return dict(interval_fractions=dict(fractions), length_m=length,
                endpoint_m=[list(map(float, p)) for p in (a, b)],
                exact_boundary_cuts=len(cuts) - 2)


def audit(glutes, bone_path):
    bone = np.memmap(bone_path, dtype=BONE, mode="r")
    sacrum = bone[(bone["body"] == 1) & (bone["bone"] == 1)]
    packed = sacrum["vertices"]
    points, welding = np.unique(packed.reshape(-1, 3), axis=0, return_inverse=True)
    triangles = welding.reshape(-1, 3)
    solid, *_ = surface(points, triangles)
    rational_faces = []
    for a, b, c in packed:
        tri = tuple(exact_xyz(x) for x in (a, b, c))
        normal = owner._cross(owner._sub(tri[1], tri[0]), owner._sub(tri[2], tri[0]))
        rational_faces.append((tri, normal))
    lo, hi = packed.min(axis=1), packed.max(axis=1)

    muscles, face_records = {}, {}
    for bank in (29, 30):
        path = glutes / f"gluteal-{bank}.npz"
        with np.load(path) as data:
            pts, faces = data["source_nodes"], data["source_faces"]
        muscles[bank] = pts[faces]
        face_records[bank], _ = owner.float32_surface_records(pts, faces)
    witness_pairs = owner._audit_pair_prepared_first(
        owner._prepare_surface_aabb(face_records[29]), face_records[30])["triangle_pairs"]
    assert len(witness_pairs) == 199

    reports = []
    for i, j in witness_pairs:
        shapes = (tuple(exact_xyz(x) for x in muscles[29][i]),
                  tuple(exact_xyz(x) for x in muscles[30][j]))
        points = sorted(set(owner.triangle_intersection_points(*shapes)))
        assert points
        if len(points) == 1:
            disposition = owner.point_location_prepared_indexed(points[0], solid)["location"]
            report = dict(kind="point", location=disposition)
        elif len(points) == 2:
            report = dict(kind="segment", **classify_segment(*points, rational_faces, lo, hi, solid))
        else:
            # No length or volume inference for a coplanar overlap polygon.
            report = dict(kind="coplanar", vertices_m=[list(map(float, p)) for p in points])
        reports.append(dict(faces=[i, j], **report))

    summary = dict(pairs=len(witness_pairs), sacral_faces=len(triangles),
                   kinds=dict(Counter(r["kind"] for r in reports)))
    segments = [r for r in reports if r["kind"] == "segment"]
    summary.update(
        strictly_inside_segments=sum(set(r["interval_fractions"]) == {"inside"} for r in segments),
        strictly_outside_segments=sum(set(r["interval_fractions"]) == {"outside"} for r in segments),
        segments_with_interior_length=sum("inside" in r["interval_fractions"] for r in segments),
        split_segments=sum(len(r["interval_fractions"]) > 1 for r in segments),
        total_segment_length_m=sum(r["length_m"] for r in segments),
        summed_segment_length_inside_m=sum(r["length_m"] * r["interval_fractions"].get("inside", 0.)
                                            for r in segments),
    )
    return dict(summary=summary, bone_sha256=digest(bone_path),
                gluteal_sha256={str(i): digest(glutes / f"gluteal-{i}.npz") for i in muscles},
                witnesses=reports)


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("glutes", "bone", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    results = audit(args.glutes, args.bone)
    with args.output.open("x") as f:
        json.dump(results, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(results["summary"]))


if __name__ == "__main__":
    main()
