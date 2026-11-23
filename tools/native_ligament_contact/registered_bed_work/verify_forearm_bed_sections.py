"""Verify each inherited bed column, its physical stiffness and the realized section history."""

import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from registered_forearms import read, sha


def table(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def verify(control, run):
    old, new = [read(p / "launch.json") for p in (control, run)]
    if (
        new["environment"].get("NUMI_HUMAN_FOREARM_BED_SECTIONS") != "1"
        or old["environment"].get("NUMI_HUMAN_FOREARM_BED_SECTIONS", "0") != "0"
    ):
        raise ValueError("Different contact experiment")
    for flag in (
        "--skin-payload",
        "--support-contact-payload",
        "--joint-equality-payload",
        "--tendon-payload",
        "--soft-tissue-payload",
    ):
        a, b = [Path(j["argv"][j["argv"].index(flag) + 1]) for j in (old, new)]
        if (
            sha(a) != sha(b)
            or old["pins"][str(a)] != new["pins"][str(b)]
            or sha(b) != new["pins"][str(b)]
        ):
            raise ValueError("Different material model")
    q0, q1 = [table(p / "resting-local-bed-footprint.csv") for p in (control, run)]
    base = {int(q["vertex"]): q for q in q0}
    if (
        len(base) != len(q0)
        or set(base) != {int(q["vertex"]) for q in q1}
        or len(q0) != len(q1)
    ):
        raise ValueError("Footprint topology changed")
    lineage = {i: i for i in range(32)} | {
        32: 12,
        33: 12,
        34: 12,
        35: 22,
        36: 22,
        37: 22,
    }
    for q in q1:
        p = base[int(q["vertex"])]
        if lineage[int(q["row"])] != int(p["row"]) or any(
            p[k] != q[k] for k in p if k != "row"
        ):
            raise ValueError("A physical column changed material, area or pose")
    rows0, rows1 = [table(p / "resting-local-bed-sections.csv") for p in (control, run)]
    if [int(q["contact_index"]) for q in rows0] != list(range(32)) or [
        int(q["contact_index"]) for q in rows1
    ] != list(range(38)):
        raise ValueError("Incomplete independent contacts")

    def stiffness(r):
        return 1 / (
            float(r["skin_compliance_m_per_n"]) + float(r["fat_compliance_m_per_n"])
        )

    result = {}
    for i, parent in enumerate(rows0):
        children = [r for r in rows1 if lineage[int(r["contact_index"])] == i]
        if i not in (12, 22) and children != [parent]:
            raise ValueError("Unrelated material changed")
        if any(
            r["source_geometry_index"] != parent["source_geometry_index"]
            for r in children
        ):
            raise ValueError("Contact lineage changed")
        a0 = float(parent["projected_area_m2"])
        a1 = sum(float(r["projected_area_m2"]) for r in children)
        k0 = stiffness(parent)
        k1 = sum(stiffness(r) for r in children)
        if abs(a1 / a0 - 1) > 1e-12 or abs(k1 / k0 - 1) > 2e-6:
            raise ValueError("Uniform displacement work or area changed")
        if len(children) > 1:
            result[i] = dict(
                area_m2=a1,
                control_stiffness_N_m=k0,
                sum_stiffness_N_m=k1,
                stiffness_relative_error=k1 / k0 - 1,
                children=[int(r["contact_index"]) for r in children],
            )
    t = table(run / "resting-com-support-impulses.csv")
    info = read(run / "terminal.receipt.json")
    if (
        len(t) % 38
        or int(t[-1]["accepted_step"]) != info["accepted_step"]
        or int(info["accepted_step"]) <= 0
    ):
        raise ValueError("Incomplete physical impulse trace")
    impulses = defaultdict(float)
    active = defaultdict(int)
    for row in t:
        k = int(row["contact_index"])
        J = float(row["normal_impulse_ns"])
        if not np.isfinite(J) or J < 0:
            raise ValueError("Nonphysical normal impulse")
        impulses[k] += J
        active[k] += int(J > 0)
    return dict(
        accepted_step=info["accepted_step"],
        seconds=info["accepted_time_s"],
        old_pack=read(control / "terminal.receipt.json")["pack_file_sha256"],
        new_pack=info["pack_file_sha256"],
        scope="Preservation of source column area and constitutive thickness, stiffness for uniform compression, and sampled normal impulses in native motion.",
        control_launch=sha(control / "launch.json"),
        launch=sha(run / "launch.json"),
        footprint_columns=len(q0),
        parents=result,
        sampled_normal_impulse_Ns=dict(impulses),
        samples_under_load=dict(active),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "run"):
        p.add_argument(name, type=Path)
    print(json.dumps(verify(**vars(p.parse_args())), indent=2, allow_nan=False))
