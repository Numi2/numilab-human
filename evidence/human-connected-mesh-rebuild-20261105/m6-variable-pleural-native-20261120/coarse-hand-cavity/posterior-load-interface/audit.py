"""Measure bed-normal paths from the native shell to represented internal surfaces."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys
from time import monotonic

import numpy as np
from scipy.spatial import cKDTree


BONE = np.dtype([
    ("body", "<u4"), ("bone", "<u4"), ("triangle", "<u4"),
    ("vertices", "<f4", (3, 3)),
])
SHELL = "BodyParts3D whole-body skin-fat bonded shell"


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def records(log):
    points, anchors, cells, evidence = {}, {}, defaultdict(list), {}
    with log.open() as f:
        for line in f:
            if line.startswith("resting_FEM_node,"):
                v = dict(s.split("=", 1) for s in line.rstrip().split(",")[1:])
                key = int(v["global"])
                assert key not in points
                points[key] = [float(v["initial_" + axis]) for axis in "xyz"]
                anchors[key] = int(v["anchor_body"])
            elif line.startswith("resting_FEM_tet,"):
                v = dict(s.split("=", 1) for s in line.rstrip().split(",")[1:])
                cells[v["object"]].append([int(v[f"n{i}"]) for i in range(4)])
            elif line.startswith(("resting_fem_skin_bed_source=",
                                  "resting_fem_skin_bed_compiled=",
                                  "whole_body_skin_bone_triangle_contact_inventory=",
                                  "resting_fascia_fem_terminal_audit=",
                                  "resting_initial_anatomical_registration=")):
                evidence[line.split("=", 1)[0]] = line.rstrip()
    assert sorted(points) == list(range(len(points))) and len(cells) == 22
    # The printed decimal precision round-trips the exact stored float words.
    return np.array([points[i] for i in range(len(points))], dtype=np.float32).astype(float), anchors, cells, evidence


def shell_faces(points, cells, layer_count):
    """Exterior faces directed away from the owning current tetrahedron."""
    seen = {}
    for tet in cells:
        for a, b, c in combinations(range(4), 3):
            ids = tuple(sorted((tet[a], tet[b], tet[c])))
            if all(i < layer_count for i in ids):
                opposite = tet[6 - a - b - c]
                if ids in seen:
                    seen[ids] = None
                else:
                    seen[ids] = opposite
    ids, opposite = zip(*((key, val) for key, val in seen.items() if val is not None))
    face, opposite = np.array(ids), np.array(opposite)
    vert = points[face]
    normal = np.cross(vert[:, 1] - vert[:, 0], vert[:, 2] - vert[:, 0])
    direction = np.einsum("ij,ij->i", normal, points[opposite] - vert[:, 0])
    assert np.all(np.isfinite(direction)) and np.all(direction != 0)
    return face, np.where(direction[:, None] > 0., -normal, normal)


def contacts(points, vertices):
    """Find z of every transverse intersection with each vertical query line.

    The centroid tree is used only with enclosing projected radii, in
    powers-of-two bins. The narrow phase uses barycentric coordinates on the
    original triangle. Coplanar/parallel triangles do not have a point hit.
    """
    xy = vertices[:, :, :2]
    c = (xy.min(axis=1) + xy.max(axis=1)) * .5
    radius = np.max(np.linalg.norm(xy - c[:, None, :], axis=2), axis=1)
    p, q, r = np.moveaxis(xy, 1, 0)
    a, b = q - p, r - p
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    out = [[] for _ in points]
    bins = np.zeros(len(radius), dtype=np.int32)
    use = np.flatnonzero(det != 0.)
    bins[use] = np.frexp(radius[use])[1]
    for level in np.unique(bins[use]):
        ids = use[bins[use] == level]
        tree = cKDTree(c[ids])
        for i, candidates in enumerate(tree.query_ball_point(points, np.ldexp(1., int(level)) + 1e-14)):
            if not len(candidates):
                continue
            index = ids[candidates]
            delta = points[i] - p[index]
            u = (delta[:, 0] * b[index, 1] - delta[:, 1] * b[index, 0]) / det[index]
            v = (a[index, 0] * delta[:, 1] - a[index, 1] * delta[:, 0]) / det[index]
            ok = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1. + 1e-10)
            index = index[ok]
            z = vertices[index, 0, 2] + u[ok] * (vertices[index, 1, 2] - vertices[index, 0, 2])
            z += v[ok] * (vertices[index, 2, 2] - vertices[index, 0, 2])
            out[i].extend(zip(index.tolist(), z.tolist()))
    return [sorted(hits, key=lambda hit: (hit[1], hit[0])) for hits in out]


def fem_boundaries(points, cells, anchors, *, fascia_only=False):
    triangles, metadata = [], []
    for name, nodes in sorted(cells.items()):
        if name == SHELL or (fascia_only and not name.startswith("numi_human_regional_myofascia_")):
            continue
        support = sorted({anchors[node] for tet in nodes for node in tet} - {2**32 - 1})
        count = defaultdict(int)
        for tet in nodes:
            for face in combinations(tet, 3):
                count[tuple(sorted(face))] += 1
        assert max(count.values()) <= 2
        if fascia_only:
            assert support
        for face, multiplicity in sorted(count.items()):
            if multiplicity == 1:
                triangles.append(face)
                metadata.append({"object": name, "face_nodes": face, "anchor_bodies": support})
    return points[triangles], metadata


def hit_summary(hits, lo, hi, identity):
    within = next(((i, z) for i, z in hits if lo <= z < hi), None)
    ahead = next(((i, z) for i, z in hits if z >= hi), None)
    return {
        "inside_column_interval": None if within is None else {
            **identity(within[0]), "distance_ahead_of_deep_m": within[1] - hi,
        },
        "first_above_deep": None if ahead is None else {
            **identity(ahead[0]), "distance_ahead_of_deep_m": ahead[1] - hi,
        },
    }


def aggregate(probes, sources):
    area = sum(v["projected_area_m2"] for v in probes)
    groups = defaultdict(lambda: {"faces": 0, "projected_area_m2": 0.})
    for v in probes:
        if any(v[source]["inside_column_interval"] for source in sources):
            label = "surface_in_skin_fat_interval"
        else:
            ahead = [h["distance_ahead_of_deep_m"]
                     for source in sources
                     if (h := v[source]["first_above_deep"]) is not None]
            if not ahead:
                label = "no_ray_hit_ahead"
            elif min(ahead) <= .001:
                label = "nearest_surface_within_1_mm"
            elif min(ahead) <= .005:
                label = "nearest_surface_1_to_5_mm"
            else:
                label = "nearest_surface_more_than_5_mm"
        groups[label]["faces"] += 1
        groups[label]["projected_area_m2"] += v["projected_area_m2"]
    return {"faces": len(probes), "projected_area_m2": area, "groups": dict(groups)}


def accepted_pose(path, reference):
    with path.open() as f:
        return read_first_pose(csv.DictReader(f), reference, path)


def read_first_pose(rows, reference, path):
    poses = {}
    for row in rows:
        if int(row["accepted_step"]) != 1:
            break
        body = int(row["body_index"])
        assert body not in poses
        poses[body] = {
            "position_m": np.array([float(row[f"root_{a}_m"]) +
                                    float(row[f"com_relative_root_{a}_m"]) for a in "xyz"]),
            "quaternion_xyzw": np.array([float(row[f"orientation_{a}"]) for a in "xyzw"]),
        }
    shifts, rotations = [], []
    for before in reference["body_poses"]:
        after = poses[before["body_index"]]
        p, q = after["position_m"], after["quaternion_xyzw"]
        r = np.array(before["quaternion_xyzw"])
        shifts.append(np.linalg.norm(p - before["position_m"]))
        rotations.append(min(np.linalg.norm(q - r), np.linalg.norm(q + r)))
    assert max(shifts) <= 1e-6 and max(rotations) <= 1e-6
    return poses, {
        "accepted_step": 1, "common_initial_body_count": len(shifts),
        "maximum_common_position_delta_m": float(max(shifts)),
        "maximum_common_quaternion_delta": float(max(rotations)),
        "accepted_pose_csv_sha256": digest(path),
    }


def closed_surface(points, faces):
    """Require distinct faces and an oppositely oriented two-use welded edge."""
    _, points_to_welded = np.unique(points, axis=0, return_inverse=True)
    faces = points_to_welded[faces]
    ordered = np.sort(faces, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]) or len(np.unique(ordered, axis=0)) != len(faces):
        return False
    e = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    forward = e[:, 0] < e[:, 1]
    _, owners, counts = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True,
                                 return_counts=True)
    return bool(np.all(counts == 2) and np.all(np.bincount(owners, forward) == 1))


def solid_hits(hits, lo, hi, identity, closed):
    """Detect tissue already behind the entry surface at the query location."""
    banks = defaultdict(list)
    for i, z in hits:
        key = identity(i)["stable_id"]
        if closed[key]:
            banks[key].append(z)
    inside = []
    ambiguous = []
    for key, positions in sorted(banks.items()):
        # Coincident hits can be a common edge or a tangent. Keep them
        # unclassified, including unresolved pairs closer than the margin.
        if (len(positions) % 2 or np.any(np.diff(positions) <= 1e-8) or
                min(abs(hi - z) for z in positions) <= 1e-8):
            ambiguous.append(key)
            continue
        if sum(z < hi for z in positions) % 2:
            entry = max(z for z in positions if z < hi)
            exit_z = min(z for z in positions if z > hi)
            inside.append({
                "stable_id": key, "distance_since_entry_m": hi - entry,
                "distance_to_exit_m": exit_z - hi, "entry_below_shell_m": max(0., lo - entry),
            })
    return {"deep_inside_welded_source_ids": inside, "ambiguous_welded_source_ids": ambiguous}


def muscle_geometry(launch, poses, reader):
    """Existing NHTISS4 source pose, independently from the FEM objects."""
    sys.path.insert(0, str(reader))
    from register_tissue_bellies import source, blend

    argv = launch["argv"]
    payload = Path(argv[argv.index("--soft-tissue-payload") + 1])
    payload_hash = digest(payload)
    manifest = json.loads(payload.with_suffix(".manifest.json").read_text())
    assert payload_hash == manifest["payload"]["sha256"]
    labels = {r["stable_id"]: r.get("label") or r["source_name"] for r in manifest["source"]["surfaces"]}
    triangles, banks, closed = [], [], {}
    for stable_id, bank in sorted(source(payload).items()):
        world, _ = blend(bank, poses)
        faces = world[bank["source_faces"]].astype(np.float32).astype(float)
        assert len(faces) and np.isfinite(faces).all()
        triangles.append(faces)
        banks.append((stable_id, len(faces)))
        closed[stable_id] = closed_surface(world.astype(np.float32), bank["source_faces"])

    starts = np.cumsum([0] + [count for _, count in banks])
    def identity(i):
        j = int(np.searchsorted(starts, i, side="right") - 1)
        stable_id, _ = banks[j]
        return {"stable_id": stable_id, "label": labels[stable_id],
                "source_bank_face": int(i - starts[j])}
    starts_by_bank = {bank: int(starts[i]) for i, (bank, _) in enumerate(banks)}
    return np.concatenate(triangles), identity, payload_hash, closed, starts_by_bank


def measure(run, bone_path, gap_limit, muscle_reader=None):
    start = monotonic()
    p, anchors, cells, evidence = records(run / "stdout.log")
    data = rows(run / "initial-skin-fem.csv")
    indices = [int(v["global_node"]) for v in data]
    x = np.array([[v[f"{axis}_m"] for axis in "xyz"] for v in data], dtype=np.float32).astype(float)
    assert np.array_equal(x, p[indices])
    n, excess = divmod(len(x), 3)
    assert n and not excess
    assert all(int(v["layer"]) == i // n and int(v["row"]) == i % n
               for i, v in enumerate(data))
    global_to_local = {node: i for i, node in enumerate(indices)}
    shell = [[global_to_local[i] for i in tet] for tet in cells[SHELL]]
    faces, normals = shell_faces(x, shell, n)
    tet_csv = rows(run / "shell-mass.csv.tets.csv")
    assert len(shell) == len(tet_csv)
    assert all(tuple(tet) == tuple(int(v[f"parent_node{i}"]) for i in range(4))
               for tet, v in zip(shell, tet_csv))
    bed = evidence["resting_fem_skin_bed_source"]
    assert "normal=0,0,1 plane_offset_m=" in bed
    offset = float(np.float32(float(re.search(r"plane_offset_m=(\S+)", bed)[1])))
    all_faces = faces
    sample = x[faces].mean(axis=1)
    ids = np.flatnonzero((normals[:, 2] < 0.) & (sample[:, 2] - offset <= gap_limit))
    assert len(ids)
    faces, sample, normals = faces[ids], sample[ids], normals[ids]
    skin_inner = x[faces + n].mean(axis=1)
    deep = x[faces + 2 * n].mean(axis=1)
    thickness = np.linalg.norm(deep - sample, axis=1)
    good = deep[:, 2] > sample[:, 2]
    bone = np.memmap(bone_path, dtype=BONE, mode="r")
    inventory = evidence["whole_body_skin_bone_triangle_contact_inventory"]
    assert len(bone) == int(re.search(r"registered_bone_triangles=(\d+)", inventory)[1])
    fascia, meta = fem_boundaries(p, cells, anchors, fascia_only=True)
    bone_hits = contacts(deep[:, :2], bone["vertices"].astype(float))
    fascia_hits = contacts(deep[:, :2], fascia)
    # Use the actual vertical path from the deep sample through the exterior;
    # taking the matched outer centroid's z can misclassify sloped columns.
    foot = contacts(deep[:, :2], x[all_faces])
    if muscle_reader is not None:
        launch = json.loads((run / "launch.json").read_text())
        registration = json.loads(evidence["resting_initial_anatomical_registration"].partition("=")[2])
        poses, pose_check = accepted_pose(run / "native/resting-body-motion-partition.csv", registration)
        muscles, muscle_identity, muscle_hash, muscle_closed, _ = muscle_geometry(launch, poses, muscle_reader)
        muscle_hits = contacts(deep[:, :2], muscles)
    def bone_identity(i):
        return {k: int(bone[k][i]) for k in ("body", "bone", "triangle")}

    samples, skipped = [], defaultdict(int)
    for i, (outer, target, normal, face) in enumerate(zip(sample, deep, normals, faces)):
        area = float(-normal[2] * .5)
        if not good[i]:
            skipped["no_inward_vertical_component"] += 1
            continue
        lower_hits = [height for _, height in foot[i] if target[2] - 2 * thickness[i] <= height < target[2]]
        if not lower_hits:
            skipped["no_bounded_vertical_exit"] += 1
            continue
        bottom = max(lower_hits)
        samples.append({
            "outer_face_nodes": face.tolist(), "outer_centroid_m": outer.tolist(),
            "deep_centroid_m": target.tolist(),
            "current_fat_column_length_m": float(np.linalg.norm(target - skin_inner[i])),
            "bed_gap_m": float(outer[2] - offset),
            "projected_area_m2": area, "vertical_shell_interval_m": [bottom, float(target[2])],
            "bone": hit_summary(bone_hits[i], bottom, target[2], bone_identity),
            "fascia": hit_summary(fascia_hits[i], bottom, target[2], meta.__getitem__),
        })
        if muscle_reader is not None:
            samples[-1]["posed_muscle"] = hit_summary(muscle_hits[i], bottom, target[2], muscle_identity)
            samples[-1]["posed_muscle"].update(solid_hits(
                muscle_hits[i], bottom, target[2], muscle_identity, muscle_closed))
            # Register each contacted source triangle in its typed bank. Keep
            # exact local coordinates so the witness can seed a connected fit.
            forward = samples[-1]["posed_muscle"]["first_above_deep"]
            if forward is not None:
                index, z = next((j, z) for j, z in muscle_hits[i] if z >= target[2])
                t = muscles[index]
                uv = np.linalg.solve((t[1:, :2] - t[0, :2]).T, target[:2] - t[0, :2])
                barycentric = np.r_[1. - uv.sum(), uv]
                witness = np.r_[target[:2], z]
                assert np.max(np.abs(barycentric @ t - witness)) <= 1e-9
                forward["source_face_barycentric"] = barycentric.tolist()
                forward["initial_world_point_m"] = witness.tolist()
                forward["fat_column_extension_to_hit_m"] = float(
                    np.linalg.norm(witness - skin_inner[i]))
    solid_summary = {}
    if muscle_reader is not None:
        solid_summary["topologically_closed_source_banks"] = sum(muscle_closed.values())
        solid_summary["total_source_banks"] = len(muscle_closed)
        counts = Counter()
        for sample in samples:
            for row in sample["posed_muscle"]["deep_inside_welded_source_ids"]:
                counts[row["stable_id"]] += 1
        solid_summary["deep_samples_inside_by_source_id"] = dict(counts)
    source_lists = {"mechanical_surfaces": ("bone", "fascia")}
    if muscle_reader is not None:
        source_lists["with_posed_muscle_sources"] = ("bone", "fascia", "posed_muscle")
    return {
        "scope": "Initial emitted shell, sampled along bed-normal lines. Bone and fascia come from the mechanical surface inventory. Optional muscle surfaces are reconstructed at the first accepted 15.625-microsecond step from NHTISS4 and do not imply volumetric or contact ownership. These are surface hits, independent of collision and force certification.",
        "inputs_sha256": {f.name: digest(f) for f in [
            run / "stdout.log", run / "initial-skin-fem.csv",
            run / "shell-mass.csv.tets.csv", run / "launch.json", bone_path]},
        "bed_plane_offset_m": offset, "maximum_sample_bed_gap_m": gap_limit,
        "bone_triangles": len(bone), "fascia_triangles": len(fascia),
        "sample_candidates": len(ids), "skipped_by_reason": dict(skipped),
        "posed_muscle_source_sha256": muscle_hash if muscle_reader else None,
        "posed_muscle_pose_check": pose_check if muscle_reader else None,
        "posed_muscle_closed_inventory": solid_summary,
        "summary_closest_gluteal": gluteal_summary(samples),
        "posed_muscle_triangles": len(muscles) if muscle_reader else 0,
        "summary": {
            key: {
                str(limit): aggregate([v for v in samples if v["bed_gap_m"] <= limit], sources)
                for limit in sorted({v for v in (.005, .010, .020, gap_limit) if v <= gap_limit})
            } for key, sources in source_lists.items()
        },
        "samples": samples, "audit_wall_seconds": monotonic() - start,
    }


def gluteal_summary(samples):
    rows = []
    for s in samples:
        if s["bed_gap_m"] > .005:
            continue
        hit = s.get("posed_muscle", {}).get("first_above_deep")
        if hit is None or "gluteus maximus" not in hit["label"]:
            continue
        all_distances = [r["distance_ahead_of_deep_m"]
                         for source in ("bone", "fascia", "posed_muscle")
                         if (r := s[source]["first_above_deep"]) is not None]
        if hit["distance_ahead_of_deep_m"] != min(all_distances):
            continue
        rows.append((s, hit))
    if not rows:
        return None
    def quantiles(values):
        return np.quantile(values, (0., .5, .9, 1.)).tolist()
    return {
        "samples_with_outer_gap_at_most_5_mm": len(rows),
        "by_muscle": dict(Counter(hit["label"] for _, hit in rows)),
        "quantile_probabilities": [0., .5, .9, 1.],
        "current_fat_column_length_m": quantiles([s["current_fat_column_length_m"] for s, _ in rows]),
        "additional_bed_normal_distance_m": quantiles([v["distance_ahead_of_deep_m"] for _, v in rows]),
        "full_column_length_to_source_m": quantiles([v["fat_column_extension_to_hit_m"] for _, v in rows]),
        "note": "Bed-normal correspondence to the nearest source hit. The full column length is a proposed correspondence distance, not a measured anatomical thickness.",
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("bone", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gap-limit", type=float, default=.025)
    parser.add_argument("--muscle-reader", type=Path)
    a = parser.parse_args()
    if not np.isfinite(a.gap_limit) or a.gap_limit <= 0.:
        parser.error("--gap-limit must be finite and positive")
    result = measure(a.run, a.bone, a.gap_limit, a.muscle_reader)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
