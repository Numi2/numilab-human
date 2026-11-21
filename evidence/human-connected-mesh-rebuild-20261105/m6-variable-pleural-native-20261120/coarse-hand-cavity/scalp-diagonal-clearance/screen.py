"""Screen coherent three-layer surface diagonal changes on exact native states."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import json
from time import monotonic

import numpy as np

from audit import BONE, digest, rows
from audit_boundary_bones import bone_endpoints
from audit_bed_layer_surfaces import install_for_float32, owner, role_surfaces
from audit_bed_motion_shell import terminal_positions

ROLES = ("outer", "interface", "deep")


def diagonal_pair(edge, a, b):
    u, v = edge
    return ((u, a, v), (u, v, b)), ((u, a, b), (a, v, b))


def normal_guard(x, before, after):
    def normals(t):
        vertices = x[np.asarray(t)].astype(float)
        return np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    old, new = normals(before), normals(after)
    areas = np.linalg.norm(new, axis=1)
    normal = old.sum(axis=0)
    if np.any(areas <= 1.e-14) or np.linalg.norm(normal) <= 1.e-14:
        return False
    return bool(np.all(new @ normal > .5 * areas * np.linalg.norm(normal))
                and min(areas) > .15 * min(np.linalg.norm(old, axis=1)))


def options(layers, points, hit_faces):
    n = len(points[0]) // 3
    lookup = [{tuple(sorted(v - layer*n for v in f)) for f in layers[role]} for layer, role in enumerate(ROLES)]
    boundary = set.intersection(*lookup)
    by_edges = defaultdict(list)
    for face in sorted(boundary):
        for edge in combinations(face, 2):
            by_edges[edge].append(face)
    selected = []
    stats = Counter()
    for edge, neighbors in sorted(by_edges.items()):
        if len(neighbors) != 2 or not any(f in hit_faces for f in neighbors):
            continue
        stats["incident_to_skull_crossing"] += 1
        a,b = [next(v for v in face if v not in edge) for face in neighbors]
        if tuple(sorted((a,b))) in by_edges:
            stats["alternative_edge_already_exists"] += 1
            continue
        columns = sorted((*edge,a,b))
        if np.min(points[0][columns, 2]) < 1.e-4:
            stats["bed_neighbor"] += 1
            continue
        old, new = diagonal_pair(edge, a, b)
        before = np.concatenate([np.asarray(old) + layer*n for layer in range(3)])
        after = np.concatenate([np.asarray(new) + layer*n for layer in range(3)])
        if not all(normal_guard(x, before[j:j+2], after[j:j+2]) for x in points for j in (0,2,4)):
            stats["orientation_or_area"] += 1
            continue
        selected.append((edge, tuple(columns), before, after))
    return selected, stats


def bank_hits(points, candidates, bone, registration, motion, step):
    all_faces = np.concatenate([record[3] for record in candidates])
    frames = [defaultdict(Counter) for _ in points]
    prepared = [owner._prepare_surface_aabb(owner.float32_surface_records(x, all_faces)[0]) for x in points]
    gathered_nodes = np.unique(all_faces)
    for bank, endpoint in bone_endpoints(bone, registration, motion, step):
        for phase, (x, target) in enumerate(zip(points, (bank["vertices"], endpoint), strict=True)):
            low, high = x[gathered_nodes].min(axis=0), x[gathered_nodes].max(axis=0)
            candidates_box = np.all(target.max(axis=1) >= low, axis=1) & np.all(target.min(axis=1) <= high, axis=1)
            subset = np.flatnonzero(candidates_box)
            if not len(subset):
                continue
            p = target[subset].reshape(-1, 3)
            other = owner.float32_surface_records(p, np.arange(len(p)).reshape(-1,3))[0]
            pairs = owner._audit_pair_prepared_first(prepared[phase], other)["triangle_pairs"]
            for face, tri in pairs:
                row = bank[subset[tri]]
                frames[phase][face // 6][(ROLES[face % 6 // 2], int(row["body"]), int(row["bone"]))] += 1
    return frames


def by_face(record):
    result = defaultdict(Counter)
    for role, _, nodes, body, bone, _ in record["witnesses"]:
        result[tuple(sorted(nodes))][(role, body, bone)] += 1
    return result


def allowed(before, after):
    return (not (after-before) and
            sum(v for k,v in before.items() if k[1] == 23) >
            sum(v for k,v in after.items() if k[1] == 23))


def main():
    ap = ArgumentParser(description=__doc__)
    for key in ("run", "bones", "output"):
        ap.add_argument(key, type=Path)
    args = ap.parse_args()
    install_for_float32()
    start = monotonic()
    witness_file = args.run / "bone-endpoints.json"
    witness = json.loads(witness_file.read_text())
    nodedata = args.run / "shell-mass.csv.nodes.csv"
    tetdata = args.run / "shell-mass.csv.tets.csv"
    log = args.run / "stdout.log"
    motion = args.run / "native/resting-body-motion-partition.csv"
    for path in (nodedata, tetdata, log, motion, args.bones):
        assert witness["input_hashes"][path.name] == digest(path), path
    initial = np.array([[row[f"current_{a}_m"] for a in "xyz"] for row in rows(nodedata)], np.float32)
    step = int(witness["accepted_endpoint_step"])
    points = (initial, terminal_positions(log, initial))
    with log.open() as stream:
        registration = json.loads(next(line for line in stream
              if line.startswith("resting_initial_anatomical_registration=")).partition("=")[2])["body_poses"]
    surfaces = role_surfaces(rows(tetdata))
    states = (witness["initial"], witness["terminal"])
    n = len(initial)//3
    hit_faces = {tuple(sorted(v % n for v in r[2])) for s in states for r in s["witnesses"] if r[3] == 23}
    edits, stats = options(surfaces, points, hit_faces)
    pose_crossings = bank_hits(points, edits, np.memmap(args.bones, mode="r", dtype=BONE),
                               registration, motion, step)
    retained = []
    index = tuple(map(by_face, states))
    for i, (edge, columns, old, new) in enumerate(edits):
        before = [sum((entry[tuple(sorted(face))] for face in old), Counter()) for entry in index]
        after = [frame[i] for frame in pose_crossings]
        if not all(allowed(a, b) for a, b in zip(before, after, strict=True)):
            stats["crossing_or_no_two_pose_gain"] += 1
            continue
        pair = [[{"/".join(map(str,k)):v for k,v in counter.items()} for counter in phase]
                for phase in zip(before, after, strict=True)]
        retained.append({"edge": list(edge), "columns": list(columns), "before": old.tolist(),
                         "after": new.tolist(), "pairs_by_role_and_bank": pair,
                         "gains": [sum(a.values()) - sum(b.values()) for a,b in zip(before,after,strict=True)]})
    preferred = sorted(retained, key=lambda r: (-min(r["gains"]), -sum(r["gains"]), r["edge"]))
    picks, used = [], set()
    for row in preferred:
        if used.isdisjoint(row["columns"]):
            used.update(row["columns"])
            picks.append(row)
    out = {"scope": "Surface authoring screen, fixed vertices, initial and terminal accepted pose; no volume changes inferred.",
           "inputs_sha256": {p.name:digest(p) for p in (witness_file, nodedata, tetdata, args.bones)}, "stats": dict(stats),
           "all_surface_candidates": preferred, "disjoint_surface_candidates": picks,
           "disjoint_estimated_gain": np.asarray([r["gains"] for r in picks], dtype=int).sum(axis=0).tolist(),
           "elapsed_s": monotonic()-start}
    with args.output.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k:v for k,v in out.items() if not k.endswith("candidates")}))
    print("geometry candidates:",len(preferred), "disjoint:",len(picks))


if __name__ == "__main__":
    main()
