"""Bound support load within the installed dense tissue stretch limits on the real bed footprint."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from fit_bed_section_curve import column_force, stress
from material_context import sha


def read(p):
    return json.loads(p.read_text())


def csvrows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def bound(source, case):
    spec = read(source / "curves.json")
    footprint = case / "resting-local-bed-footprint.csv"
    if sha(footprint) != spec["footprint"]:
        raise ValueError("Different source bed areas")
    cols = csvrows(footprint)
    prop = [
        (r["mu_pa"], r["log_volume_coefficient_pa"])
        for r in spec["material"]["materials"]
    ]
    lower = 2 / 3
    # The confined stress in the two layers is equal. The softer layer
    # reaches the native lower stretch limit first under monotone compression.
    pmax = min(stress(lower, *p) for p in prop)
    end = [brentq(lambda s: stress(s, *p) - pmax, lower - 1e-9, 1.0) for p in prop]
    impulses = csvrows(case / "resting-com-support-impulses.csv")
    dt = float(
        read(case / "launch.json")["argv"][
            read(case / "launch.json")["argv"].index("--muscle-step-seconds") + 1
        ]
    )
    surf = csvrows(case / "resting-com-momentum-diagnostic.csv")
    weight = float(surf[-1]["total_body_mass_kg"]) * 9.81
    data = []
    for r in spec["sections"]:
        i = r["index"]
        section = [c for c in cols if int(c["row"]) == i]
        area, ts, tf = [
            np.array([float(c[k]) for c in section])
            for k in ("area_m2", "skin_m", "fat_m")
        ]
        d = (ts * (1 - end[0]) + tf * (1 - end[1])).min()
        cap = column_force(np.array([d]), area, ts, tf, prop)[0][0]
        loads = [
            float(j["normal_impulse_ns"]) / dt
            for j in impulses
            if int(j["contact_index"]) == i and int(j["accepted_step"]) * dt >= 15.0
        ]
        if len(loads) != 16:
            raise ValueError("Incomplete supported endpoint")
        data.append(
            dict(
                section=i,
                source=r["source"],
                allowed_common_shortening_m=float(d),
                bounded_common_shortening_N=float(cap),
                bounded_independent_columns_N=float(pmax * area.sum()),
                last_second_mean_sampled_force_N=float(np.mean(loads)),
            )
        )
    return dict(
        footprint=sha(footprint),
        material=sha(source / "curves.json"),
        minimum_admissible_principal_stretch=lower,
        soft_layer_limited_pressure_Pa=float(pmax),
        limiting_layer_stretches=end,
        weight_N=weight,
        all_38_common_shortening_capacity_N=sum(
            r["bounded_common_shortening_N"] for r in data
        ),
        all_columns_independent_capacity_N=sum(
            r["bounded_independent_columns_N"] for r in data
        ),
        sections_above_static_material_capacity=sum(
            r["last_second_mean_sampled_force_N"] > r["bounded_common_shortening_N"]
            for r in data
        ),
        scope="Static upper bounds allowing all sections to touch, exact bed-normal confined elastic model at reference state. No sustained damping support in static equilibrium. A force requirement below the sum does not establish moment balance or contact feasibility.",
        sections=data,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "case"):
        p.add_argument(k, type=Path)
    p.add_argument(
        "--require-capacity",
        action="store_true",
        help="Require static support capacity on the actual registered columns",
    )
    args = p.parse_args()
    data = bound(args.source, args.case)
    print(json.dumps(data, indent=2, allow_nan=False))
    if (
        args.require_capacity
        and data["all_38_common_shortening_capacity_N"] < data["weight_N"]
    ):
        raise SystemExit(
            "The projected contact sections cannot sustain body weight within the dense material envelope"
        )
