"""Close the native elbow hinge to the retained wrist about an atlas-derived shoulder center."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from myo_sim.build.compose import build_model
from numilab_human.shoulder_registration import solve_elbow_reach
from scipy.spatial.transform import Rotation as R
from shoulder_inputs import digest
from source_pair_audit import inputs


def unit(x):
    return x / np.linalg.norm(x)


def align(a, b):
    a, b = unit(a), unit(b)
    c = np.cross(a, b)
    l = np.linalg.norm(c)
    if l < 1.0e-12:
        if a @ b > 0:
            return R.identity()
        return R.from_rotvec(np.pi * unit(np.cross(a, np.eye(3)[np.argmin(abs(a))])))
    return R.from_rotvec(c * (np.arctan2(l, a @ b) / l))


def run(launch, refs, motion, heads, torso, capture, spacing, out):
    _, poses, rest, owners = inputs(launch, refs, motion)
    poses[0] = rest
    init = {
        r["body_index"]: np.array(r["position_m"] + r["quaternion_xyzw"])
        for r in json.loads(capture.read_text())["accepted_body_poses"]
    }
    poses[-4] = init
    fit = json.loads(torso.read_text())
    T = R.from_quat(fit["rot_xyzw"])
    model = build_model("myofullbody")
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    offsets = {
        r["side"]: np.asarray(r["translation_atlas_axes_m"])
        for r in json.loads(spacing.read_text())["rows"]
    }
    ret = []
    for item in json.loads(heads.read_text())["rows"]:
        side = item["side"]
        hum = owners["humerus_" + side][1]
        rad = owners["radius_" + side][1]
        GH = state.xanchor[model.joint("shoulder_rot_" + side).id].copy()
        el = model.joint("elbow_flexion_" + side)
        E = state.xanchor[el.id].copy()
        n = unit(state.xaxis[el.id].copy())
        W = state.xanchor[model.joint("deviation_" + side).id].copy()
        A = E - GH
        F = W - E
        initial = state.qpos[el.qposadr[0]]
        bounds = np.asarray(el.range) - initial
        # Donor default xanchor and NHRIGID2 use the same verified body frames.
        er = R.from_quat(rest[hum][3:]).inv().apply(E - rest[hum][:3])
        wr = R.from_quat(rest[rad][3:]).inv().apply(W - rest[rad][:3])
        tg = (
            T.apply(item["variants"][0]["atlas_center_m"]) * fit["scale"]
            + fit["translation_m"]
            + T.apply(offsets[side])
        )
        # Source sternum and rib records bind to one torso core body.
        rawbone = Path(json.loads(launch.read_text())["argv"][3])
        manifest = json.loads(rawbone.with_suffix(".manifest.json").read_text())
        tb = next(
            a["core_body_index"]
            for a in manifest["source"]["anchors"]
            if a["member_id"] == "FJ3290"
        )
        rows = []
        for st, p in poses.items():
            torso_q = p[tb]
            target = torso_q[:3] + R.from_quat(torso_q[3:]).apply(tg)
            exact = p[rad][:3] + R.from_quat(p[rad][3:]).apply(wr)
            elbow = p[hum][:3] + R.from_quat(p[hum][3:]).apply(er)
            d = np.linalg.norm(exact - target)
            sol, extent, ok = solve_elbow_reach(A, F, n, bounds, d)
            prior = R.from_quat(p[hum][3:]) * R.from_quat(rest[hum][3:]).inv()
            candidates = []
            for angle in sol:
                V = A + R.from_rotvec(n * angle).apply(F)
                axis = unit(exact - target)
                Q = align(prior.apply(V), axis) * prior
                a = Q.apply(A)
                a -= axis * (axis @ a)
                b = elbow - target
                b -= axis * (axis @ b)
                twist = np.arctan2(axis @ np.cross(a, b), a @ b)
                Q = R.from_rotvec(axis * twist) * Q
                pred_el = target + Q.apply(A)
                pred_w = target + Q.apply(V)
                candidates.append(
                    {
                        "step": int(st),
                        "reachable": ok,
                        "elbow_q_rad": angle + initial,
                        "elbow_displacement_m": np.linalg.norm(pred_el - elbow),
                        "shoulder_relative_rotation_deg": np.degrees(
                            (Q * prior.inv()).magnitude()
                        ),
                        "wrist_error_m": np.linalg.norm(pred_w - exact),
                        "shoulder_target_m": target.tolist(),
                        "elbow_target_m": pred_el.tolist(),
                        "wrist_target_m": pred_w.tolist(),
                        "humeral_rotation_xyzw": Q.as_quat().tolist(),
                        "source_elbow_hinge_axis_world": Q.apply(n).tolist(),
                    }
                )
            winner = min(candidates, key=lambda x: x["elbow_displacement_m"])
            if ok and winner["wrist_error_m"] > 1e-9:
                raise ValueError("Hinge closure failed")
            rows.append(winner)
        ret.append(
            {
                "side": side,
                "forearm_pronation": "donor default",
                "source_lengths_m": [np.linalg.norm(A), np.linalg.norm(F)],
                "native_elbow_q_range_rad": el.range.tolist(),
                "reach_range_m": extent,
                "poses": len(rows),
                "reachable": sum(r["reachable"] for r in rows),
                "max_wrist_error_m": max(r["wrist_error_m"] for r in rows),
                "max_elbow_motion_m": max(r["elbow_displacement_m"] for r in rows),
                "samples": [
                    v for v in rows if v["step"] in (-4, -3, -2, -1, 0, 32, 256, 8000)
                ],
            }
        )
    out.write_text(
        json.dumps(
            {"heads_sha256": digest(heads), "rows": ret},
            indent=2,
            default=lambda o: o.item(),
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
