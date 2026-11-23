"""Measure actual skeletal drift and unsupported release on a native settled-body trajectory."""

import argparse, json, re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from registered_forearms import read, sha
from compare_supported_forearm_runs import table


def measure(case):
    early, late = [
        read(case / name)
        for name in ("step-3999.receipt.json", "terminal.receipt.json")
    ]
    if (
        early["physical_endpoint"] != "accepted"
        or late["physical_endpoint"] != "accepted"
        or not (0 < early["accepted_time_s"] < late["accepted_time_s"])
    ):
        raise ValueError("Expected two accepted native endpoints")
    old, new = [
        {p["body_index"]: p for p in r["accepted_body_poses"]} for r in (early, late)
    ]
    if (
        not old
        or old.keys() != new.keys()
        or len(old) != early["accepted_body_pose_count"]
        or len(new) != late["accepted_body_pose_count"]
    ):
        raise ValueError("Mismatched bodies")
    diff = []
    for k, x in sorted(old.items()):
        y = new[k]
        rot = (
            R.from_quat(x["quaternion_xyzw"]).inv() * R.from_quat(y["quaternion_xyzw"])
        ).magnitude()
        diff.append(
            dict(
                body=k,
                translation_mm=float(
                    1000 * np.linalg.norm(np.subtract(x["position_m"], y["position_m"]))
                ),
                rotation_deg=float(np.degrees(rot)),
            )
        )
    trace = table(case / "resting-com-momentum-diagnostic.csv")
    final = [r for r in trace if float(r["time_s"]) > late["accepted_time_s"] - 1]
    coupled = table(case / "resting-coupled.csv")
    (initial,) = re.findall(
        r"^resting_initialization=dynamic_release .*?$",
        (case / "stdout.log").read_text(),
        re.M,
    )
    if not final or not coupled:
        raise ValueError("Missing physical trace")
    time = [float(r["time_s"]) for r in coupled]
    if abs(time[-1] - late["accepted_time_s"]) > 1e-7:
        raise ValueError("Stale physiology")

    def values(key):
        return np.array([float(r[key]) for r in coupled])

    def bounds(key):
        return [float(values(key).min()), float(values(key).max())]

    return dict(
        sources=[
            early["pack_file_sha256"],
            late["pack_file_sha256"],
            sha(case / "resting-coupled.csv"),
        ],
        start_s=early["accepted_time_s"],
        end_s=late["accepted_time_s"],
        body_count=len(diff),
        initialization=initial,
        peak_body_translation=max(diff, key=lambda k: k["translation_mm"]),
        peak_body_rotation=max(diff, key=lambda k: k["rotation_deg"]),
        median_body_translation_mm=float(
            np.median([r["translation_mm"] for r in diff])
        ),
        all_bodies=diff,
        gas_and_circulation_bounds={
            k: bounds(k)
            for k in (
                "PaO2_mmhg",
                "PaCO2_mmhg",
                "pleural_pa",
                "lv_ml",
                "right_atrium_ml",
                "blood_error_ml",
                "root_assistance_n",
                "root_assistance_nm",
            )
        },
        cycle_counts={
            k: float(coupled[-1][k])
            for k in ("breaths", "complete_filling_ejection_cycles")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    print(json.dumps(measure(**vars(parser.parse_args())), indent=2, allow_nan=False))
