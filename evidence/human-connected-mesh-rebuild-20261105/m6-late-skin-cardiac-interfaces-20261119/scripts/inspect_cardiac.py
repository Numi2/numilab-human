"""Compare the rejected pleural load against the nearest authorized cardiac state."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from resting_anatomy_interface_patch import parse_payload

ROOT = Path(__file__).resolve().parent.parent
TRIAL = ROOT / "coupled-thumb-reference-001/combined-12s-no-export"
CAPTURE = ROOT / "coupled-thumb-reference-001/thumb-4s-capture/native/accepted-geometry/step-2000.common-coordinates.bin"

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def surface(vertices, faces):
    triangle = vertices[faces]
    return np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])

def main(output, exact, use_calibration):
    launch = json.loads((TRIAL / "launch.json").read_text())
    args = launch["argv"]
    receipt = Path(args[args.index("--resting-anatomy-receipt") + 1])
    anatomy = Path(args[args.index("--torso-anatomy-payload") + 1])
    field = json.loads(receipt.read_text())["provenance"]["cardiac_geometry_binding"]["common_field"]
    assert sha(anatomy) == field["anatomy_payload_sha256"]
    _, rows = parse_payload(anatomy)
    failure = json.loads((TRIAL / "common-failure.json").read_text())["first_failure_latch"]
    candidate = np.array(failure["common_first"] + failure["common_second"][:3], dtype="<f4")
    if use_calibration:
        calibration = json.loads((output.parent/"cardiac-range.json").read_text())
        candidate = np.asarray(calibration["worst_case_q"], dtype="<f4")
    boxes_file = receipt.parent / field["domain_boxes"]["path"]
    assert sha(boxes_file) == field["domain_boxes"]["sha256"]
    bounds = np.fromfile(boxes_file, "<f4").reshape(-1, 2, 8)[:, :, :7]
    projections = np.clip(candidate, bounds[:, 0], bounds[:, 1])
    nearest = np.linalg.norm(projections-candidate, axis=1).argmin()
    boundary = projections[nearest]
    accepted_raw = np.fromfile(CAPTURE, "<f4")
    accepted = np.r_[accepted_raw[:4], accepted_raw[4:7]]
    assert np.any(np.all((accepted >= bounds[:, 0]) & (accepted <= bounds[:, 1]), axis=1))
    map_path = receipt.parent / field["map"]["path"]
    assert sha(map_path) == field["map"]["sha256"]
    coefficients = np.memmap(map_path, "<f4", mode="r").reshape(-1, 7, 4)[:, :, :3]
    result = {"candidate_q": candidate.tolist(), "nearest_box": int(nearest),
              "authorized_boundary_q": boundary.tolist(), "captured_accepted_q": accepted.tolist(),
              "anatomy_sha256": sha(anatomy), "source_field_sha256": field["map"]["sha256"],
              "surfaces": []}
    for entry in field["vertex_ranges"]:
        key, first, count = (entry[k] for k in ("stable_id", "first_vertex", "vertex_count"))
        xyz = rows[key]["vertices6"][:, :3]
        faces = rows[key]["faces"]
        basis = coefficients[first:first + count]
        assert len(xyz) == count
        def posed(q):
            # One rounded coordinate per vertex. Native compiler FMA evaluation
            # can differ by ULPs, so this remains an authoring diagnostic.
            return (xyz + np.einsum("vqc,q->vc", basis.astype(float), q.astype(float))).astype("<f4")
        start, end = posed(boundary), posed(candidate)
        direction0, direction1 = surface(start.astype(float), faces), surface(end.astype(float), faces)
        areas0, areas1 = (np.linalg.norm(d, axis=1) for d in (direction0, direction1))
        assert areas0.min() > 0
        dot = (direction0*direction1).sum(axis=1)
        report = dict(stable_id=key, vertices=count, faces=len(faces),
                      maximum_change_mm=float(1000*np.linalg.norm(end-start, axis=1).max()),
                      min_area_ratio=float((areas1/areas0).min()), nonpositive_normal_dot=int((dot<=0).sum()))
        if exact and key == 318:
            from numilab_human import cardiac_cavity_intersections as i
            from numilab_human import common_atlas_skin_clearance as c
            for name, positions in (("boundary", start), ("candidate", end)):
                t = time.monotonic()
                records = c._exact_surface_records(positions, faces)
                hits = i._audit_pair_prepared_first(i._prepare_surface_aabb(records), records, same_surface=True)
                report[name] = hits
                print(name, hits["count"], time.monotonic()-t, flush=True)
        result["surfaces"].append(report)
        print(key, report["maximum_change_mm"], report["min_area_ratio"], report["nonpositive_normal_dot"], flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exact-right-atrium", action="store_true")
    parser.add_argument("--calibration-worst",action="store_true")
    args = parser.parse_args()
    main(args.output, args.exact_right_atrium, args.calibration_worst)
