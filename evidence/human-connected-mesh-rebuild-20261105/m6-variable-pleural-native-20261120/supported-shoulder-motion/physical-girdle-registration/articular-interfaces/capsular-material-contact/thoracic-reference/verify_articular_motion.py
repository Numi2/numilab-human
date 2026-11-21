"""Replay saved contact features against the closed, registered atlas surfaces."""

import itertools
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from audit_solids import winding
from compare_atlas import vertices_and_faces
from compare_registration import prepare
from scipy.spatial.transform import Rotation
from shoulder_inputs import digest, old

from certify_material_branch import Bone, certify, coords, posed, unit


def adjacent_surfaces(archive, anchors):
    bank_ids = [8, 9, 27, *range(36, 60)]
    neighbors = {}
    with zipfile.ZipFile(archive) as source:
        members = {
            Path(name).stem: name for name in source.namelist() if name.endswith(".obj")
        }
        import hashlib

        for bank in bank_ids:
            anchor = anchors[bank - 1]
            raw = source.read(members[anchor["member_id"]])
            if hashlib.sha256(raw).hexdigest() != anchor["member_sha256"]:
                raise ValueError(f"Atlas neighbor {bank} changed")
            v, f = vertices_and_faces(raw)
            neighbors[bank] = v, f
    return neighbors


def adjacent_hits(points, neighbors):
    hits = []
    lo, hi = points.min(0), points.max(0)
    for bank, (verts, faces, lower, upper) in neighbors.items():
        if (lower > hi).any() or (upper < lo).any():
            continue
        tri = verts[faces]
        front, back = tri.min(1), tri.max(1)
        for i, (a, b) in enumerate(itertools.pairwise(points)):
            relevant = np.flatnonzero(
                np.all((front <= np.maximum(a, b)) & (back >= np.minimum(a, b)), axis=1)
            )
            for fraction, j in old.audit.lp.vector_hits(a, b, (verts, faces[relevant])):
                hits.append([bank, i, "cross", int(relevant[j]), fraction])
            if abs(winding(tri, (a + b) * 0.5)) > 0.5:
                hits.append([bank, i, "inside"])
    return hits


def run(
    launch,
    registration,
    motion,
    archive,
    sites,
    heads,
    torso,
    spacing,
    features,
    output,
):
    config, _, _, _, _, _, atlas = prepare(launch, registration, motion, archive)
    body = json.loads(Path(config["argv"][3]).with_suffix(".manifest.json").read_text())
    nearby = adjacent_surfaces(archive, body["source"]["anchors"])
    topology = json.loads(features.read_text())
    if topology["atlas_sha256"] != digest(archive) or topology[
        "sites_sha256"
    ] != digest(sites):
        raise ValueError("Shoulder feature input changed")
    centers = {
        r["side"]: np.array(r["variants"][0]["atlas_center_m"])
        for r in json.loads(heads.read_text())["rows"]
    }
    rows = json.loads(sites.read_text())["rows"]
    scale = json.loads(torso.read_text())["scale"]
    angles = (1, 2, 3, 4, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0)
    schedule = [(0, 0)] + [(k, angle) for k in range(3) for angle in angles]
    identity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    shifts = {
        r["side"]: np.array(r["translation_atlas_axes_m"])
        for r in json.loads(spacing.read_text())["rows"]
    }
    results = []
    for part in topology["rows"]:
        side = part["side"]
        sites_side = next(
            r["attachments"] for r in rows if r["side"] == int(side == "r")
        )
        center = centers[side]
        adjacent = {}
        for bank, (v, f) in nearby.items():
            world = (v - center) * scale - shifts[side]
            adjacent[bank] = world, f, world.min(0), world.max(0)
        bones = {}
        anchors = []
        for site in sites_side:
            sid = site["stable"]
            _, _, v, f, _, _, _, physical_scale, _ = atlas[sid]
            local = (v - center) * physical_scale - (
                shifts[side] if not anchors else np.zeros(3)
            )
            bones[sid] = Bone((local, f), identity, sid)
            anchors.append(np.asarray(site["barycentric"]) @ local[f[site["triangle"]]])
        humerus = sites_side[1]["stable"]
        prev = None
        samples = []
        for idx, (axis, angle, features_row) in enumerate(part["samples"]):
            if idx >= len(schedule) or (axis, angle) != schedule[idx]:
                raise ValueError(
                    "Motion sequence differs from its prescribed coordinates"
                )
            if not all(
                0 <= t <= 1 and 0 <= e < len(bones[humerus].edges)
                for e, t in features_row
            ):
                raise ValueError("Invalid material edge or coordinate")
            edges = [(humerus, e, t) for e, t in features_row]
            rot = Rotation.from_rotvec(np.eye(3)[axis] * np.radians(angle))
            body_pose = np.r_[np.zeros(3), rot.as_quat()]
            current = {**bones, humerus: posed(bones[humerus], body_pose)}
            points = coords(current, [anchors[0], rot.apply(anchors[1])], edges)
            check = certify(
                list(current.values()),
                points,
                edges,
                [t for _, _, t in edges],
                sites_side,
            )
            others = adjacent_hits(points, adjacent)
            tangent = unit(np.diff(points, axis=0))
            length_gradient = np.vstack(
                (-tangent[0], tangent[:-1] - tangent[1:], tangent[-1])
            )
            # Every knot after the scapular anchor is supported on the humerus.
            moment = np.cross(points[1:], length_gradient[1:]).sum(0)
            length = check["length_m"]
            good = (
                not (check["crossings"] or check["inside"] or others)
                and check["normal_cone_residual_per_unit_tension"] <= 1e-6
            )
            record = {
                "axis": axis,
                "angle_deg": angle,
                "accepted": good,
                "length_m": length,
                "cone_residual": check["normal_cone_residual_per_unit_tension"],
                "force_residual": check["force_residual_per_unit_tension"],
                "moment_residual_m": check["moment_residual_m_per_unit_tension"],
                "owner_pair_hits": check["crossings"],
                "owner_inside": check["inside"],
                "adjacent_hits": others,
            }
            if good and prev:
                old_angle, old_moment, old_length = prev
                delta = np.radians(angle - old_angle)
                record["integrated_work_error_m_per_unit_tension"] = abs(
                    (length - old_length)
                    - 0.5 * delta * (moment[axis] + old_moment[axis])
                )
            if idx and not samples[-1]["accepted"]:
                raise ValueError("A rejected pose cannot seed further continuation")
            prev = (angle, moment, length)
            samples.append(record)
        passed = [s for s in samples if s["accepted"]]
        neutral = samples[0]["length_m"]
        results.append(
            {
                "side": side,
                "prescribed": len(schedule),
                "attempted": len(samples),
                "accepted": len(passed),
                "neutral_length_m": neutral,
                "max_neutral_return_error_m": max(
                    (
                        abs(neutral - s["length_m"])
                        for s in passed
                        if s["angle_deg"] == 0
                    ),
                    default=None,
                ),
                "max_trapezoidal_work_error_m_per_unit_tension": max(
                    (
                        s.get("integrated_work_error_m_per_unit_tension", 0)
                        for s in passed
                    ),
                    default=None,
                ),
                "max_accepted_cone_residual": max(
                    (s["cone_residual"] for s in passed), default=None
                ),
                "max_accepted_force_residual": max(
                    (s["force_residual"] for s in passed), default=None
                ),
                "max_accepted_moment_residual_m": max(
                    (s["moment_residual_m"] for s in passed), default=None
                ),
                "failures": [s for s in samples if not s["accepted"]],
            }
        )
    output.write_text(
        json.dumps(
            {
                "atlas_sha256": digest(archive),
                "features_sha256": digest(features),
                "torso_sha256": digest(torso),
                "spacing_sha256": digest(spacing),
                "centers_sha256": digest(heads),
                "neighbor_banks": sorted(nearby),
                "rows": results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
