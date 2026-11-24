"""Measure registered attachment motion, bed support, and coupled physiology at matched accepted times."""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from compile_connected_joint import digest


def get(p):
    return json.loads(p.read_bytes())


def table(case, name):
    with (case / "native-traces" / name).open() as f:
        return list(csv.DictReader(f))


def write(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def compare(ctl, run, out):
    r = get(run / "launch.json")
    if (
        r["enthesis_control"] != digest(ctl / "launch.json")
        or r["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] != "1"
    ):
        raise ValueError("Unmatched trial")
    if get(run / "execution.json")["exit"] != 0:
        raise ValueError("Incomplete motion")
    series = [table(p, "resting-coupled.csv") for p in [ctl, run]]
    data = [
        {k: np.array([float(row[k]) for row in ss]) for k in ss[0]} for ss in series
    ]
    old, new = data
    if (
        list(old) != list(new)
        or not np.array_equal(old["step"], new["step"])
        or not np.array_equal(old["time_s"], new["time_s"])
        or int(new["step"][-1]) != 8000
    ):
        raise ValueError("Accepted clocks differ")
    if not all(np.isfinite(v).all() for t in data for v in t.values()):
        raise ValueError("Nonfinite coupled response")
    bounds = {k: [float(v.min()), float(v.max()), float(v[-1])] for k, v in new.items()}
    d = {k: float(abs(new[k] - v).max()) for k, v in old.items()}
    outputs = (
        "lung_volume_ml",
        "pleural_pa",
        "PaO2_mmhg",
        "PaCO2_mmhg",
        "lv_mmhg",
        "rv_mmhg",
        "aorta_mmhg",
        "lv_ml",
        "rv_ml",
        "aortic_ejected_ml",
        "blood_ml",
    )
    scaled = {k: d[k] / max(abs(old[k]).max(), 1e-15) for k in outputs}
    # Each sample contains only the final physical step of its segment.
    # Divide by that step duration; earlier impulses are not present in this series.
    loads = []
    for case in [ctl, run]:
        owner = "external_support_impulses=per_contact_pre_step_basis_and_external_world_sum_final_physical_step_per_segment"
        if owner not in (case / "stdout.log").read_text():
            raise ValueError("Unknown bed impulse duration")
        m = table(case, "resting-com-support-impulses.csv")
        bins = {}
        keys = set()
        for row in m:
            if row["row_class"] != "external_support_contact":
                continue
            i, j = int(row["sample_start_step"]), int(row["accepted_step"])
            k = (i, j, int(row["contact_index"]))
            if k in keys:
                raise ValueError("Repeated support contribution")
            keys.add(k)
            bins[i, j] = bins.get((i, j), 0.0) + float(row["normal_impulse_ns"])
        spans = sorted(bins)
        if (
            not spans
            or any(b != c for (a, b), (c, d0) in zip(spans, spans[1:]))
            or spans[0][0] != 0
            or spans[-1][1] != 8000
        ):
            raise ValueError("Nonconsecutive bed impulses")
        dt = old["time_s"][-1] / 8000
        near = [k for k in spans if k[0] >= 7000]
        loads.append(
            dict(
                sampled_mean_normal_N=sum(bins.values()) / (len(spans) * dt),
                last_two_seconds_sampled_mean_N=sum(bins[k] for k in near)
                / (dt * len(near)),
                last_sample_normal_N=bins[spans[-1]] / dt,
                sampled_physical_steps=len(spans),
                cadence_steps=sorted(set(j - i for i, j in spans)),
            )
        )
    logs = (run / "stdout.log").read_text()
    sig = re.search(
        r"resting_integrated_body=completed simulated_s=([^ ]+) wall_s=([^ ]+) real_time_factor=([^ ]+) physiology_body_clock=matched root_assistance=false",
        logs,
    )
    if (
        not sig
        or max(abs(new["root_assistance_n"])) != 0
        or max(abs(new["root_assistance_nm"])) != 0
    ):
        raise ValueError("Unmatched assisted run")
    control_poses = {}
    changes = []
    for case in [ctl, run]:
        caps = sorted((case / "native-traces/accepted-geometry").glob("*.receipt.json"))
        for cap in caps:
            g = get(cap)
            n = g["accepted_step"]
            bones = g["accepted_body_poses"]
            x = np.array([b["position_m"] for b in bones])
            u = np.array([b["quaternion_xyzw"] for b in bones])
            u /= np.linalg.norm(u, axis=1)[:, None]
            if case == ctl:
                control_poses[n] = (x, u)
                continue
            x0, u0 = control_poses[n]
            delta = np.linalg.norm(x - x0, axis=1)
            theta = 4 * np.arcsin(
                np.clip(
                    np.minimum(
                        np.linalg.norm(u - u0, axis=1), np.linalg.norm(u + u0, axis=1)
                    )
                    * 0.5,
                    0,
                    1,
                )
            )
            changes.append(
                dict(
                    step=n,
                    bones=len(bones),
                    max_bone_translation_m=float(delta.max()),
                    max_bone_rotation_rad=float(theta.max()),
                    receipt=digest(cap),
                )
            )
    if set(control_poses) != set(q["step"] for q in changes):
        raise ValueError("Different native capture phases")
    peaks = new["lv_mmhg"]
    beats = int(
        np.sum(
            (peaks[1:-1] > peaks[:-2]) & (peaks[1:-1] >= peaks[2:]) & (peaks[1:-1] > 70)
        )
    )
    sec, wall, rtf = map(float, sig.groups())
    cold = get(run / "execution.json")["wall_s"]
    res = dict(
        control=digest(ctl / "launch.json"),
        trial=digest(run / "launch.json"),
        accepted_steps=8000,
        simulated_s=sec,
        body_s=wall,
        body_rtf=rtf,
        cold_wall_s=cold,
        cold_rtf=sec / cold,
        trace_samples=len(series[1]),
        absolute_response_delta=d,
        relative_to_control_peak=scaled,
        changed_pose=changes,
        bed_reaction=loads,
        first_to_final_blood_ml=float(new["blood_ml"][-1] - new["blood_ml"][0]),
        sampled_lv_pressure_peaks_above_70_mmhg=beats,
        range_min_max_last=bounds,
        native_configuration=dict(
            variable_pleural=r["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"],
            regional_bed=r["environment"]["NUMI_HUMAN_LOCAL_BED_LAYERS"],
            passive_fascia=r["environment"]["NUMI_HUMAN_RESTING_PASSIVE_FASCIA"],
        ),
        measured_gpu_s=sum(
            map(
                float,
                re.findall(
                    r"resting_native_profile step=\d+ command_ms=\S+ gpu_ms=(\S+)", logs
                ),
            )
        )
        * 0.001,
        presentation_calls=len(re.findall(r"resting_present_profile step=", logs)),
        presentation_wall_s=sum(
            map(
                float,
                re.findall(
                    r"resting_present_profile step=\d+ finished=[01] total_present_wall_ms=(\S+)",
                    logs,
                ),
            )
        )
        * 0.001,
        scope="Registered regional skin/fat bed contact and native articulated body. Dense whole-body fascia and skin/fat certification are outside this execution mode. Wall time includes cold start. Native presentation throughput is unqualified.",
    )
    write(out, res)
    print(
        rtf,
        sec / cold,
        loads,
        beats,
        {
            k: bounds[k]
            for k in [
                "breaths",
                "tidal_ml",
                "aortic_ejected_ml",
                "PaO2_mmhg",
                "PaCO2_mmhg",
                "peak_penetration_m",
            ]
        },
        flush=True,
    )


if __name__ == "__main__":
    compare(*(Path(v).resolve() for v in sys.argv[1:]))
