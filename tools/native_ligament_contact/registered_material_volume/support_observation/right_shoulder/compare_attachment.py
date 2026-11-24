"""Verify the anatomical source edit against its matched native rigid, material and physiological states."""

import json
import re
import sys
from pathlib import Path

import numpy as np
from compile_connected_joint import digest

from measure_applied_route import get, rows, trace


def compare(ctl, run, reg, src, ctr_summary, run_summary, out):
    cfg = get(run / "launch.json")
    r = get(reg / "registration.json")
    old = get(ctl / "launch.json")
    if cfg["enthesis_control"] != digest(ctl / "launch.json") or cfg[
        "registered_shoulder"
    ] != digest(reg / "registration.json"):
        raise ValueError("Wrong motion or registration")
    if (
        cfg["argv"][0] != old["argv"][0]
        or cfg["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] != "1"
    ):
        raise ValueError("Changed physics owner")
    a, b = get(ctr_summary), get(run_summary)
    if a["launch"] != digest(ctl / "launch.json") or b["launch"] != digest(
        run / "launch.json"
    ):
        raise ValueError("Changed motion measurements")

    def read_geometry(p):
        g = get(next((p / "native-traces/accepted-geometry").glob("*.receipt.json")))
        return (
            g,
            np.array([v["position_m"] for v in g["accepted_body_poses"]]),
            np.array([v["quaternion_xyzw"] for v in g["accepted_body_poses"]]),
        )

    def get_report(p):
        txt = (p / "stdout.log").read_text()
        l = next(
            l for l in txt.splitlines() if l.startswith("tendon_payload=NHTENDON3 ")
        )
        q = dict(re.findall(r"(\w+)=(\S+)", l))
        return q

    q0, q1 = get_report(ctl), get_report(run)
    if (
        int(q1["tendon_envelope_bindings"]) != r["new_envelopes"]
        or int(q0["tendon_point_bindings"]) != int(q1["tendon_point_bindings"]) + 1
        or q1["tendon_geometry_identity_verified"] != "true"
        or q1["tendon_muscle_payload_sha256"] != r["muscle"]
    ):
        raise ValueError("Native tendon owner did not consume the repair")
    if any(
        q1[k] != q0[k]
        for k in (
            "tendon_max_reference_path_delta_m",
            "tendon_max_architecture_scale_change",
            "tendon_endpoints",
            "tendon_bone_payload_sha256",
        )
    ):
        raise ValueError("Tendon architecture bounds changed")
    g, x, u = read_geometry(ctl)
    h, y, v = read_geometry(run)
    if (
        g["accepted_step"] != h["accepted_step"]
        or g["accepted_time_s"] != h["accepted_time_s"]
        or [z["body_index"] for z in g["accepted_body_poses"]]
        != [z["body_index"] for z in h["accepted_body_poses"]]
    ):
        raise ValueError("Unmatched physical state")
    dist = np.linalg.norm(x - y, axis=1)
    u = u / np.linalg.norm(u, axis=1)[:, None]
    v = v / np.linalg.norm(v, axis=1)[:, None]
    angle = 4 * np.arcsin(
        np.clip(
            0.5
            * np.minimum(np.linalg.norm(u - v, axis=1), np.linalg.norm(u + v, axis=1)),
            0,
            1,
        )
    )
    c0, c1 = trace(ctl), trace(run)
    if len(c0) != len(c1) or c0[0].keys() != c1[0].keys():
        raise ValueError("Unmatched coupled channels")
    trace_delta = {
        k: max(abs(float(p[k]) - float(q[k])) for p, q in zip(c0, c1, strict=True))
        for k in c0[0]
    }
    exact = []
    for i in range(len(c0)):
        ii, z = rows(ctl, i)
        jj, w = rows(run, i)
        exact.append(
            dict(
                step=i,
                ids_identical=bool(np.array_equal(ii, jj)),
                values_identical=bool(np.array_equal(z, w)),
            )
        )
    files = [
        "physical-accepted.node",
        "native-traces/resting-body-motion-partition.csv",
        "native-traces/resting-com-q-integration.csv",
        "force-observer.f32",
    ]
    mat = get(src / "material-check.json")
    match = (
        a["emitted_end"] == b["emitted_end"] == mat["physical_frame"]
        and mat["accepted_s"] == h["accepted_time_s"]
    )
    summary = dict(
        registration=digest(reg / "registration.json"),
        native_muscle_sha256=q1["tendon_muscle_payload_sha256"],
        native_tendon_payload_sha256=r["tendon"],
        control=a["launch"],
        trial=b["launch"],
        accepted_s=h["accepted_time_s"],
        accepted_steps=h["accepted_step"],
        source_contact=exact,
        bone_count=len(x),
        max_bone_translation_m=float(dist.max()),
        max_bone_rotation_rad=float(angle.max()),
        file_comparisons={k: digest(ctl / k) == digest(run / k) for k in files},
        numerical_channels=len(trace_delta),
        nonzero_channel_deltas={k: v for k, v in trace_delta.items() if v},
        bed_N=[a["bed_N"], b["bed_N"]],
        accepted_material_identical_to_audited=match,
        shape_failures={k: z["below_shape"] for k, z in mat["roles"].items()}
        if match
        else None,
        inversions={k: z["inverted"] for k, z in mat["roles"].items()}
        if match
        else None,
        cold_wall_s=b["cold_wall_s"],
        native_body_clock=b["integrated"],
        scope="The source and distributed force geometry were changed. An individual infraspinatus tension was not sampled. The comparison covers two coupled physical steps.",
    )
    out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    compare(*(Path(v).resolve() for v in sys.argv[1:]))
