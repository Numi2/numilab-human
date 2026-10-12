#!/usr/bin/env python3
"""Replay reviewed cap flips without changing anatomy coordinates or parent bindings."""
import argparse, hashlib, json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def boundary(faces):
    result = Counter()
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1]):
            result[min(a, b), max(a, b)] += 1 if a < b else -1
    return {edge: value for edge, value in result.items() if value}

def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))

def replay(source, reports, expected):
    document = json.loads(source.read_text())
    geometry = document.get("geometry", document)
    original_points = geometry["points_exact"]
    original_parents = geometry["parent_maps"]
    previous_hash = digest(source)
    operations = []
    for path in reports:
        report = json.loads(path.read_text())
        require(report["base_candidate_sha256"] == previous_hash, "report chain mismatch")
        op = report["selected_flip"]
        sid = str(op["row"])
        indices, old, new = op["faces"], op["old_faces"], op["new_faces"]
        require(len(indices) == len(old) == len(new) == 2, "flip must replace two faces")
        require([geometry["faces"][sid][i] for i in indices] == old, "old faces differ")
        require(boundary(old) == boundary(new), "patch boundary changed")
        vertices = sorted({v for face in old+new for v in face})
        points = {i: tuple(F(int(a), int(b)) for a, b in original_points[sid][i]) for i in vertices}
        normal = cross(sub(points[old[0][1]], points[old[0][0]]), sub(points[old[0][2]], points[old[0][0]]))
        require(any(normal), "zero old triangle")
        require(all(sum(x*y for x, y in zip(normal, sub(points[i], points[old[0][0]]))) == 0 for i in vertices), "nonplanar source cap")
        for i, face in zip(indices, new):
            ancestry = geometry["face_ancestry"][sid][i]
            require(ancestry["origin_kind"] == 1 and ancestry["parent_face_index"] == 4294967295, "flip touches measured source face")
            area = cross(sub(points[face[1]], points[face[0]]), sub(points[face[2]], points[face[0]]))
            require(sum(x*y for x, y in zip(normal, area)) > 0, "new triangle reverses or collapses")
            geometry["faces"][sid][i] = face
        operations.append({"report_path": str(path.resolve()), "report_sha256": digest(path), "selected_flip": op})
        previous_hash = report["candidate_sha256"]
    target = json.loads(expected.read_text())
    target = target.get("geometry", target)
    require(digest(expected) == previous_hash, "expected candidate hash differs")
    for key in ("points_exact", "parent_maps", "faces", "face_ancestry"):
        require(geometry[key] == target[key], "replayed geometry differs: "+key)
    require(geometry["points_exact"] == original_points and geometry["parent_maps"] == original_parents, "coordinates or parents changed")
    geometry["replayed_cap_flips"] = {
        "source_path": str(source.resolve()), "source_sha256": digest(source),
        "verified_geometry_candidate_sha256": digest(expected), "operations": operations,
        "scope": "Inferred cap topology correction only; residual contacts remain; not full anatomical admission."
    }
    return document

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    result = replay(args.source, args.report, args.expected)
    with args.output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output), "geometry_matches_expected": True}))

if __name__ == "__main__":
    main()

