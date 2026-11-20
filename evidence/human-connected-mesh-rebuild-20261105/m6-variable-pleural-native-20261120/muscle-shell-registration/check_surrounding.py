"""Reject new exact self and adjacent-surface intersections in fitted muscle banks."""
import argparse
from itertools import combinations
import hashlib
import json
import mmap
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [
    str(ROOT / "resting-clearance-and-atrium-006"),
    str(ROOT / "resting-surface-clearance-001/src"),
]
from register_tissue_bellies import CASES, READER, SKIN, clear, exact, exact_skin_pair, install_for_float32

MUSCLE = 51005

def triangle_pairs(first, second):
    records = clear._exact_surface_records(*first)
    return exact_skin_pair(
        exact._prepare_surface_aabb(records), clear._float32_skin_aabb_index(records), *second
    )

def self_pairs(points, faces):
    unique, mapping = np.unique(points, axis=0, return_inverse=True)
    records = clear._exact_surface_records(unique, mapping.ravel()[faces])
    report = exact._audit_pair_prepared_first(
        exact._prepare_surface_aabb(records), records, same_surface=True
    )
    return {tuple(p) for p in report["triangle_pairs"]}

def audit(surfaces, updates, results):
    barrier_triangles = {}
    for key, trial in updates.items():
        initial, faces = surfaces[key]
        moving = np.any(np.any(initial[faces] != trial[0][faces], axis=2), axis=1)
        before_self = self_pairs(initial, faces)
        after_self = self_pairs(*trial)
        additions = sorted(after_self - before_self)
        left, right = [], []
        row = results[key]
        row["self"] = dict(before=len(before_self), after=len(after_self), new=len(additions),
                           first_new=additions[:10])
        boxes = np.concatenate((initial[faces[moving]].reshape(-1,3),
                                trial[0][faces[moving]].reshape(-1,3)))
        low, high = boxes.min(0), boxes.max(0)
        for other, neighbor in surfaces.items():
            if other in (key, SKIN):
                continue
            if np.any(neighbor[0].max(0) < low) or np.any(neighbor[0].min(0) > high):
                continue
            before = triangle_pairs(neighbor, (initial, faces[moving]))
            after = triangle_pairs(neighbor, (trial[0], faces[moving]))
            new = sorted(after - before)
            if new:
                neighbor_ids, moving_ids = np.asarray(new, dtype=int).T
                left.append(faces[moving][moving_ids])
                right.append(neighbor[0][neighbor[1][neighbor_ids]])
            if before or after:
                subset = np.flatnonzero(moving)
                row["surrounding"].append(dict(
                    target=other, before=len(before), after=len(after), new=len(new),
                    first_new=[(a, int(subset[b])) for a, b in new[:10]]
                ))
        if left:
            barrier_triangles[key] = (np.concatenate(left), np.concatenate(right))
        print(json.dumps(dict(key=key, self_new=len(additions),
            surrounding_new=sum(r["new"] for r in row["surrounding"]))), flush=True)
    return barrier_triangles

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--emit-neighbor-witnesses", action="store_true")
    args = parser.parse_args()
    install_for_float32()
    inputs = np.load(HERE / "paired-geometry.npz")
    deltas = {int(k.removeprefix("bank")): v for k,v in
              np.load(HERE / f"{args.experiment}.npz").items()}
    newest = ROOT / "resting-clearance-and-atrium-017/floor-native-live-parent-026"
    captures = ((newest / "native/accepted-geometry/step-16.mrvpack",
                 newest / "exact-skin-surface.json"), CASES[-1])
    report = []
    source_checks = {}
    witness_triangles = {}
    for sid, delta in deltas.items():
        old = inputs[f"b{sid}_v"]
        faces = inputs[f"b{sid}_f"]
        before = self_pairs(old, faces)
        after = self_pairs((old + delta).astype(np.float32), faces)
        new = sorted(after - before)
        source_checks[sid] = dict(before=len(before), after=len(after), new=len(new), first_new=new[:10])
    for index, (path, expected) in enumerate(captures):
        receipt = json.loads(path.with_suffix(".receipt.json").read_text())
        accepted = json.loads(expected.read_text())
        bodies = {row["body_index"]: row for row in receipt["accepted_body_poses"]}
        with tempfile.TemporaryFile(dir=HERE) as raw:
            subprocess.run(["/opt/homebrew/bin/zstd", "-qdc", path.with_suffix(".mrvpack.zst")],
                           stdout=raw, check=True)
            raw.seek(0)
            digest = hashlib.file_digest(raw, "sha256").hexdigest()
            assert digest == receipt["pack_file_sha256"] == accepted["accepted_pack_sha256"]
            with mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                surfaces = {}
                for key, _, points, faces in READER["primitives"](mapped, bodies):
                    assert key not in surfaces, key
                    surfaces[key] = (points, faces)
        records = {}
        updates = {}
        for sid, delta in deltas.items():
            key = MUSCLE, sid
            old, faces = surfaces[key]
            assert np.array_equal(old, inputs[f"pose{index}_b{sid}_v"])
            assert np.array_equal(faces, inputs[f"b{sid}_f"])
            posed = old + np.einsum("vij,vj->vi", inputs[f"pose{index}_b{sid}_j"], delta)
            updates[key] = posed.astype(np.float32), faces
            records[key] = dict(bank=sid, surrounding=[])
        new_neighbors = audit(surfaces, updates, records)
        for (_, sid), (left, right) in new_neighbors.items():
            witness_triangles[f"pose{index}_b{sid}_left"] = left
            witness_triangles[f"pose{index}_b{sid}_right"] = right
        joint = []
        for a, b in combinations(updates, 2):
            before = triangle_pairs(surfaces[a], surfaces[b])
            after = triangle_pairs(updates[a], updates[b])
            new = sorted(after - before)
            joint.append(dict(banks=(a[1], b[1]), before=len(before), after=len(after), new=len(new),
                              first_new=new[:10]))
        report.append(dict(pack_sha256=digest, step=receipt["accepted_step"], time_s=receipt["accepted_time_s"],
                           banks=list(records.values()), joint=joint))
    viable = set(deltas)
    viable.difference_update(sid for sid,row in source_checks.items() if row["new"])
    for pose in report:
        for bank in pose["banks"]:
            if bank["self"]["new"] or any(r["new"] for r in bank["surrounding"]):
                viable.discard(bank["bank"])
        for pair in pose["joint"]:
            if pair["new"]:
                viable.difference_update(pair["banks"])
    output = dict(baseline_source_checks=source_checks, poses=report, passes_surrounding=sorted(viable))
    if args.emit_neighbor_witnesses and witness_triangles:
        np.savez_compressed(HERE / f"{args.experiment}-neighbor-witnesses.npz", **witness_triangles)
    (HERE / f"{args.experiment}-surrounding.json").write_text(json.dumps(output, indent=2) + "\n")

if __name__ == "__main__":
    main()
