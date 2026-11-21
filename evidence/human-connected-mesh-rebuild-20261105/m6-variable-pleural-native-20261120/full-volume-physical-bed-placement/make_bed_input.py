"""Derive the native plane from the exact registered three-point footprint."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


fit_path, mass_path = map(Path, sys.argv[1:])
fit = json.loads(fit_path.read_text())
mass = json.loads(mass_path.read_text())
q0 = fit["original_root_pose"]
q1 = fit["trial_root_pose"]
rotation = Rotation.from_quat(q1[3:]) * Rotation.from_quat(q0[3:]).inv()
normal = rotation.inv().apply([0., 0., 1.])
points = np.array(fit["lower_hull_triangle"])
assert np.ptp(points @ normal) < 1e-13
# This is the existing native release window around the 10 um contact
# activation width. Clear by 0.1 um from the lower edge of its 1 um guard.
gap = 9.1e-6
offset = float(np.min(points @ normal) - gap)
normal_f32 = normal.astype(np.float32).astype(float)
normal_f32 /= np.linalg.norm(normal_f32)
offset_f32 = float(np.float32(normal[2] * float(np.float32(offset / normal[2]))))
emitted_gaps = points @ normal_f32 - offset_f32
assert np.all((emitted_gaps > 9e-6) & (emitted_gaps < 1e-5))
centre = np.array(mass["centre_of_accounted_mass_xy_m"])
weights = np.linalg.solve(np.vstack((points[:, :2].T, np.ones(3))), np.r_[centre, 1.])
assert min(weights) > 0

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


print(json.dumps(dict(
    argv_root_pose=q0, normal=normal.tolist(), offset_m=offset,
    compiled_gaps_m=emitted_gaps.tolist(), nominal_vertical_weights=weights.tolist(),
    mass_moment_coverage=mass["moment_accounted_mass_fraction"],
    weight_N=mass["total_body_weight_at_9_81_N_kg_N"],
    input_sha256={fit_path.name: sha(fit_path), mass_path.name: sha(mass_path)},
    scope="Fixed real flat bed geometry and identical registered body pose. Geometric force-split feasibility only."),
    indent=2))
