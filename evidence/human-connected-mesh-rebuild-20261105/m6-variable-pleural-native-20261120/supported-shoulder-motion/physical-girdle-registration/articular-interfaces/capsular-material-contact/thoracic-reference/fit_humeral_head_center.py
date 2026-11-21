"""Estimate a shared glenohumeral center from the registered source's articular head."""

import json
import sys
from pathlib import Path

import numpy as np
from compare_registration import prepare
from numilab_human.shoulder_registration import fit_articular_sphere
from scipy.spatial.transform import Rotation as R
from shoulder_inputs import digest


def run(
    launch,
    refs,
    motion,
    archive,
    registrations,
    sites,
    center_input,
    torso,
    capture,
    out,
):
    _, cat, _meshes, _, _poses, rest, atlas = prepare(launch, refs, motion, archive)
    raw = json.loads(torso.read_text())
    fr = R.from_quat(raw["rot_xyzw"])
    ft = np.asarray(raw["translation_m"])
    scale = raw["scale"]
    cap = {
        r["body_index"]: np.array(r["position_m"] + r["quaternion_xyzw"])
        for r in json.loads(capture.read_text())["accepted_body_poses"]
    }
    qt = cap[cat[27]["body"]]
    results = []
    cent = {k["side"]: k for k in json.loads(center_input.read_text())["rows"]}
    routes = {
        k["side"]: k
        for k in json.loads(registrations.read_text())["routes"]
        if k["name"].startswith("s_glen")
    }
    for row in json.loads(sites.read_text())["rows"]:
        sc, hu = row["attachments"]
        side = "r" if row["side"] else "l"
        item = atlas[hu["stable"]]

        def to_raw(e, local):
            aa = atlas[e["stable"]]
            q = rest[e["body"]]
            return (
                aa[6].inv().apply(q[:3] + R.from_quat(q[3:]).apply(local) - aa[8])
                / aa[7]
            )

        guess = to_raw(hu, cent[side]["centers"][1]["atlas_center_body_m"])
        ref_sc = atlas[sc["stable"]]
        sup = np.asarray(sc["barycentric"]) @ ref_sc[2][ref_sc[3][sc["triangle"]]]
        r0 = routes[side]["source_wrap"]["radius_m"] / item[7]
        v, f = item[2:4]
        tri = v[f]
        points = tri.mean(1)
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        areas = np.linalg.norm(cr, axis=1)
        orientation = np.sign(np.einsum("ij,ij->i", tri[:, 0] - v.mean(0), cr).sum())
        normal = orientation * cr / areas[:, None]
        direction = sup - guess
        direction /= np.linalg.norm(direction)
        d = points - guess
        rad = np.linalg.norm(d, axis=1)
        outward = d / np.maximum(rad[:, None], 1e-15)
        az = np.asarray([1.0, 0.0, 0.0])
        az -= az @ direction * direction
        az /= np.linalg.norm(az)
        theta = np.arctan2(outward @ np.cross(direction, az), outward @ az)
        wedge = np.floor((theta + np.pi) * 8 / (2 * np.pi)).astype(int) % 2
        results_side = []
        for lower, upper, co, look in [
            (0.70, 1.40, 0.80, 0),
            (0.75, 1.30, 0.85, 0.1),
            (0.65, 1.45, 0.75, -0.1),
        ]:
            eligible = (
                (rad >= lower * r0)
                & (rad <= upper * r0)
                & (np.sum(normal * outward, axis=1) > co)
                & ((outward @ direction) > look)
            )
            if eligible.sum() < 30:
                raise ValueError("Insufficient humeral articular coverage")
            p, cond = fit_articular_sphere(points[eligible], areas[eligible], guess, r0)
            checks = []
            for k in (0, 1):
                train = eligible & (wedge == k)
                hold = eligible & ~train
                pp, _cm = fit_articular_sphere(points[train], areas[train], guess, r0)
                err = np.linalg.norm(points[hold] - pp[:3], axis=1) - pp[3]
                checks.append(
                    {
                        "train": int(train.sum()),
                        "held": int(hold.sum()),
                        "center": pp[:3].tolist(),
                        "center_shift_m": np.linalg.norm(pp[:3] - p[:3]),
                        "hold_rms_m": np.sqrt(
                            np.average(err * err, weights=areas[hold])
                        ),
                        "hold_p95_abs_m": np.quantile(abs(err), 0.95),
                    }
                )
            native_world = qt[:3] + R.from_quat(qt[3:]).apply(
                fr.apply(p[:3]) * scale + ft
            )
            results_side.append(
                {
                    "face_count": int(eligible.sum()),
                    "atlas_center_m": p[:3].tolist(),
                    "radius_m": p[3],
                    "native_entrance_target_m": native_world.tolist(),
                    "normal_matrix_design_condition": cond,
                    "split": checks,
                }
            )
        results.append(
            {
                "side": side,
                "variants": results_side,
                "max_center_sensitivity_m": max(
                    np.linalg.norm(
                        np.asarray(a["atlas_center_m"])
                        - results_side[0]["atlas_center_m"]
                    )
                    for a in results_side
                ),
            }
        )
    out.write_text(
        json.dumps(
            {
                "registration_sha256": digest(torso),
                "inputs_anchors_sha256": digest(sites),
                "atlas_sha256": digest(archive),
                "rows": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
