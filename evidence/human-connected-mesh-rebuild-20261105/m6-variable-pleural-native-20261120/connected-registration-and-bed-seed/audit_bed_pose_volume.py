"""Screen pose-dependent Float32 geometry before rebuilding registered bindings."""
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


mesh, fit_path = map(Path, sys.argv[1:3])
emit_metal_fixture = sys.argv[3:] == ["--metal-fixture"]
if sys.argv[3:] and not emit_metal_fixture:
    raise ValueError("Unexpected options")
pins = json.loads((mesh / "receipt.json").read_text())
for name, key in (("mesh.node", "mesh_node_sha256"), ("mesh.ele", "mesh_ele_sha256")):
    if sha256(mesh / name) != pins[key]:
        raise ValueError(f"Source mesh bytes changed: {name}")

points = np.loadtxt(mesh / "mesh.node", skiprows=1, usecols=(1, 2, 3))
cells = np.loadtxt(mesh / "mesh.ele", skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=np.int32)
tets, materials = cells[:, :4] - 1, cells[:, 4]
assert np.array_equal(points, points.astype(np.float32))
assert (len(points), len(tets)) == (pins["nodes"], pins["cells"])
assert tets.min() >= 0 and tets.max() < len(points) and set(materials) == {15, 16}

fit = json.loads(fit_path.read_text())
original = np.array(fit["original_root_pose"])
trial = np.array(fit["trial_root_pose"])
rotation = Rotation.from_quat(trial[3:]) * Rotation.from_quat(original[3:]).inv()
posed = rotation.apply(points - original[:3]) + trial[:3]
rounded = posed.astype(np.float32).astype(float)
edge_pairs = tuple(combinations(range(4), 2))


def measures(vertices, indices):
    rows = vertices[indices]
    edges = rows[:, 1:] - rows[:, :1]
    determinant = np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))
    length = np.maximum.reduce([np.linalg.norm(rows[:, a] - rows[:, b], axis=1)
                                for a, b in edge_pairs])
    return determinant / 6, determinant / length**3


def screen(volume, quality, selected):
    invalid = np.flatnonzero(selected & (volume <= 0))
    volume_floor = np.flatnonzero(selected & (volume <= 1e-18))
    quality_floor = np.flatnonzero(selected & (quality < 1e-5))
    return dict(minimum_volume_m3=float(volume[selected].min()),
                minimum_normalized_determinant=float(quality[selected].min()),
                inverted_or_zero=len(invalid), volume_floor_failures=len(volume_floor),
                quality_floor_failures=len(quality_floor),
                sample_inverted_or_zero_rows=invalid[:12].tolist(),
                sample_volume_floor_rows=volume_floor[:12].tolist(),
                sample_quality_floor_rows=quality_floor[:12].tolist())


reference_v, reference_q = measures(points, tets)
posed_v, posed_q = measures(posed, tets)
rounded_v, rounded_q = measures(rounded, tets)
assert min(reference_v.min(), posed_v.min()) > 1e-18
assert min(reference_q.min(), posed_q.min()) >= 1e-5
# Reconstruct the existing Matter compiler's high/low displacement storage
# from the same authored Float32 reference. This leaves reference mass and
# inverse-rest geometry tied to the original pose.
displacement = posed - points
high = displacement.astype(np.float32).astype(float)
low = (displacement - high).astype(np.float32).astype(float)
reconstructed = points + (high + low)
if emit_metal_fixture:
    stream = sys.stdout.buffer
    stream.write(np.array([len(points), len(tets)], dtype="<u4").tobytes())
    stream.write(rotation.as_matrix().astype("<f8").tobytes())
    stream.write(np.concatenate((high, low), axis=1).astype("<f4").tobytes())
else:
    relative_v, relative_q = measures(reconstructed, tets)
cooked_determinant = np.empty(len(tets))
cooked_error = np.empty(len(tets))
for first in range(0, len(tets), 65536):
    span = slice(first, first + 65536)
    rows = tets[span]
    reference = points[rows]
    dx = np.swapaxes(reference[:, 1:] - reference[:, :1], 1, 2)
    inverse = np.linalg.inv(dx).astype(np.float32).astype(float)
    if emit_metal_fixture:
        stream.write(rows.astype("<u4").tobytes())
        stream.write(inverse.astype("<f4").tobytes())
        continue
    u = (high + low)[rows]
    du = np.swapaxes(u[:, 1:] - u[:, :1], 1, 2)
    deformation = np.eye(3) + du @ inverse
    cooked_determinant[span] = np.linalg.det(deformation)
    cooked_error[span] = np.max(np.abs(deformation - rotation.as_matrix()), axis=(1, 2))

if emit_metal_fixture:
    sys.exit(0)

assert min(relative_v.min(), reference_v.min()) > 1e-18
assert min(relative_q.min(), reference_q.min()) >= 1e-5
assert np.all(np.isfinite(cooked_determinant)) and cooked_determinant.min() > 0.0
report = dict(
    scope="Offline rigid root transform followed by exact Float32 vertex emission. Native per-donor rebinding and force balance are separate.",
    fit_sha256=sha256(fit_path),
    mesh_node_sha256=pins["mesh_node_sha256"], mesh_ele_sha256=pins["mesh_ele_sha256"],
    nodes=len(points), cells=len(tets),
    max_position_rounding_m=float(np.linalg.norm(rounded - posed, axis=1).max()),
    relative_displacement_scope="Offline reconstruction of Matter compiler storage and initial F=I+grad(u); GPU contact and attachments remain unqualified.",
    max_position_reconstruction_error_m=float(np.linalg.norm(reconstructed - posed, axis=1).max()),
    minimum_cooked_J=float(cooked_determinant.min()),
    maximum_cooked_J=float(cooked_determinant.max()),
    maximum_cooked_rotation_error=float(cooked_error.max()),
    roles={})
for name, material in (("skin", 15), ("fat", 16)):
    selected = materials == material
    ratio = rounded_v[selected] / reference_v[selected]
    report["roles"][name] = dict(reference=screen(reference_v, reference_q, selected),
                                rigid_float64=screen(posed_v, posed_q, selected),
                                rounded_float32=screen(rounded_v, rounded_q, selected),
                                split_displacement=screen(relative_v, relative_q, selected),
                                volume_ratio_min=float(ratio.min()), volume_ratio_max=float(ratio.max()),
                                volume_ratio_outside_one_percent=int(np.count_nonzero(abs(ratio-1) > .01)),
                                source_signed_volume_m3=float(reference_v[selected].sum()),
                                rounded_signed_volume_m3=float(rounded_v[selected].sum()))
print(json.dumps(report, indent=2))
