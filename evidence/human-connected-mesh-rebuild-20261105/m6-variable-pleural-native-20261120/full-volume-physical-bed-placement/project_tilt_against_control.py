"""Measure tilted-plane clearance along the unmodified flat-bed control."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np

control = Path(sys.argv[1]).resolve()
plane = json.loads(Path(sys.argv[2]).read_text())
native = control / "native"
work = control.parent.parent
authoring = work / "resting-surface-clearance-001"
sys.path.insert(0, str(authoring / "src"))
parse = runpy.run_path(str(authoring / "audit.py"))["primitives"]
reference = native / "accepted-geometry/step-16.receipt.json"
receipt = json.loads(reference.read_text())
poses = {p["body_index"]: p for p in receipt["initial_anatomical_registration"]["body_poses"]}
normal = np.array(plane["normal"], dtype=np.float32).astype(np.float64)
unit = normal / np.linalg.norm(normal)
offset = float(np.float32(plane["normal"][2] * np.float32(plane["offset_m"] / plane["normal"][2])))


def measure(path, expected, poses):
    raw = subprocess.check_output(["/opt/homebrew/bin/zstd", "-dc", str(path.resolve(strict=True))])
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Geometry differs from accepted receipt: {path}")
    (_, _, points, faces), = parse(raw, poses, skin_only=True)
    if len(points) != 76153:
        raise ValueError("Unexpected authoritative outer-skin point count")
    gap = points.astype(np.float64) @ unit - offset
    return gap, faces


source, topology = measure(native / "resting-human.mrvpack.zst", receipt["base_pack_file_sha256"], poses)
accepted, accepted_topology = measure(
    native / "accepted-geometry/step-16.mrvpack.zst", receipt["pack_file_sha256"], {})
if not np.array_equal(topology, accepted_topology):
    raise ValueError("Control geometry changed its skin topology")
seeds = np.argsort(source)[:3]
if not np.allclose(source[seeds], plane["compiled_gaps_m"], rtol=1e-6, atol=1e-10):
    raise ValueError("Three-point footprint disagrees with native starting gaps")
result = {
    "scope": "Counterfactual geometry only. Flat-bed accepted motion was computed with the original plane.",
    "control_endpoint_s": receipt["accepted_time_s"],
    "start_minimum_gap_m": float(source.min()),
    "control_endpoint_minimum_gap_to_tilted_plane_m": float(accepted.min()),
    "control_endpoint_rows_below_tilted_plane": int(np.count_nonzero(accepted < 0.0)),
    "three_points": [{"typed_outer_vertex": int(j), "initial_gap_m": float(source[j]),
                      "control_endpoint_gap_m": float(accepted[j]),
                      "gap_change_m": float(accepted[j] - source[j])} for j in seeds],
}
print(json.dumps(result, indent=2, allow_nan=False))
