"""Preserve measured source joint spacing when the native humerus and scapula have different scales."""

import json
import sys
from pathlib import Path

import numpy as np
from bone_overlap import intersections
from compare_registration import prepare
from fast_skin_intersection_audit import install_for_float32
from numilab_human.shoulder_registration import preserve_socket_clearance
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from shoulder_inputs import digest

from certify_material_branch import Bone


def closest(triangles, point):
    a, b, c = triangles.transpose(1, 0, 2)
    normals = np.cross(b - a, c - a)
    square = np.einsum("ij,ij->i", normals, normals)
    if np.any(square <= 0):
        raise ValueError("Degenerate humeral surface")
    h = np.einsum("ij,ij->i", point - a, normals) / square
    proj = point - h[:, None] * normals
    inside = np.ones(len(a), bool)
    candidates = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        inside &= np.einsum("ij,ij->i", np.cross(edge, proj - start), normals) >= 0
        t = np.einsum("ij,ij->i", point - start, edge) / np.einsum(
            "ij,ij->i", edge, edge
        )
        candidates.append(start + np.clip(t, 0, 1)[:, None] * edge)
    candidates = np.vstack([*candidates, proj[inside]])
    return candidates[np.argmin(np.linalg.norm(candidates - point, axis=1))]


def articular_patch(scapula, center, radius):
    face_centers = scapula.tri.mean(1)
    r = center - face_centers
    lengths = np.linalg.norm(r, axis=1)
    weights = np.linalg.norm(
        np.cross(
            scapula.tri[:, 1] - scapula.tri[:, 0], scapula.tri[:, 2] - scapula.tri[:, 0]
        ),
        axis=1,
    )
    eligible = set(
        np.flatnonzero(
            (lengths >= 0.76 * radius)
            & (lengths <= 1.26 * radius)
            & (np.einsum("ij,ij->i", r, scapula.normals) > 0.72 * lengths)
        )
    )
    components = []
    while eligible:
        front = [eligible.pop()]
        group = []
        while front:
            item = front.pop()
            group.append(item)
            adj = set(scapula.ef[scapula.fe[item]].flat) & eligible
            front.extend(adj)
            eligible -= adj
        components.append(np.array(sorted(group)))
    if not components:
        raise ValueError("No glenoid-facing surface")
    chosen = max(components, key=lambda g: weights[g].sum())
    if len(chosen) < 20:
        raise ValueError("Insufficient contiguous glenoid surface")
    return chosen, face_centers[chosen], scapula.normals[chosen]


def preserve_spacing(scapula, hu, head, s0, ss, sh):
    center = np.array(head["atlas_center_m"])
    faces, p, n = articular_patch(scapula, center, head["radius_m"])
    tri = hu.tri
    cent = tri.mean(1)
    tree = cKDTree(cent)
    radius = np.linalg.norm(tri - cent[:, None, :], axis=2).max()
    hp = []
    for point in p:
        d, _ = tree.query(point)
        indices = tree.query_ball_point(point, d + radius)
        hp.append(closest(tri[indices], point))
    hp = np.asarray(hp)
    translation, reference_gap, gain = preserve_socket_clearance(
        p, hp, n, center, s0, ss, sh
    )
    return translation, {
        "glenoid_faces": faces.tolist(),
        "translation_atlas_axes_m": translation.tolist(),
        "min_projected_gap_change_m": float(gain.min()),
        "median_reference_material_gap_m": float(np.median(reference_gap)),
        "maximum_normal_gap_increase_m": float(gain.max()),
    }


def run(launch, registration, motion, archive, heads, torso, out):
    *_, atlas = prepare(launch, registration, motion, archive)
    source = json.loads(heads.read_text())
    scale = json.loads(torso.read_text())["scale"]
    ident = np.array([0, 0, 0, 0, 0, 0, 1.0])
    install_for_float32()
    rows = []
    for parent in source["rows"]:
        side = parent["side"]
        sc, hu = (11, 13) if side == "l" else (10, 12)
        sv, sf = atlas[sc][2:4]
        hv, hf = atlas[hu][2:4]
        ss, sh = atlas[sc][7], atlas[hu][7]
        head = parent["variants"][0]
        center = np.array(head["atlas_center_m"])
        offset, result = preserve_spacing(
            Bone((sv, sf), ident, sc), Bone((hv, hf), ident, hu), head, scale, ss, sh
        )
        tests = []
        for axis, angle in [(0, 0)] + [
            (a, d) for a in range(3) for d in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
        ]:
            rot = Rotation.from_rotvec(np.eye(3)[axis] * np.radians(angle))
            x = rot.apply((hv - center) * sh)
            old = intersections((sv - center) * ss, sf, x, hf)
            new = intersections((sv - center) * ss, sf, x + offset, hf)
            tests.append(
                {
                    "axis": axis,
                    "angle_deg": angle,
                    "centered": len(old),
                    "registered": len(new),
                    "first": new[:1],
                }
            )
        rows.append(
            dict(side=side, scapula_scale=ss, humerus_scale=sh, **result, samples=tests)
        )
    out.write_text(
        json.dumps(
            {
                "atlas_sha256": digest(archive),
                "heads_sha256": digest(heads),
                "torso_sha256": digest(torso),
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
