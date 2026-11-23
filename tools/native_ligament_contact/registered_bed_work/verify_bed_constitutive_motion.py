"""Measure the physical tissue strain implied by native bed contact at each retained sample."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from fit_bed_section_curve import column_force, stress
from material_context import sha


def read(p):
    return json.loads(p.read_text())


def table(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def verify(case, source, control):
    spec = read(source / "curves.json")
    cfg = read(case / "launch.json")
    frame = read(case / "terminal.receipt.json")
    execute = read(case / "execution.json")
    if (
        execute["exit_code"]
        or frame["accepted_step"] != 8000
        or frame["physical_endpoint"] != "accepted"
        or frame["surface_audit_endpoint"] != "passed"
    ):
        raise ValueError("Incomplete physical trajectory")
    if sha(Path(cfg["argv"][0]).parent / "metalrobo/MetalRobo.metallib") != spec["gpu"]:
        raise ValueError("Different force-law program")
    footprint = case / "resting-local-bed-footprint.csv"
    if (
        sha(footprint) != spec["footprint"]
        or sha(control / footprint.name) != spec["footprint"]
    ):
        raise ValueError("Different layer stack")
    columns = table(footprint)
    sections = table(case / "resting-local-bed-sections.csv")
    priors = table(control / "resting-local-bed-sections.csv")
    mat = [
        (r["mu_pa"], r["log_volume_coefficient_pa"])
        for r in spec["material"]["materials"]
    ]
    points = table(case / "resting-support-selected-skin-witness.csv")
    impulses = table(case / "resting-com-support-impulses.csv")
    dt = float(cfg["argv"][cfg["argv"].index("--muscle-step-seconds") + 1])
    rows = [[] for _ in sections]
    support = {}
    for w, q in zip(points, impulses, strict=True):
        i = int(q["contact_index"])
        step = int(q["accepted_step"])
        if (
            int(w["accepted_step"]) != step
            or int(w["region"]) != i
            or w["source_geometry_index"] != q["source_geometry_index"]
            or w["point_query_index"] != str(spec["sections"][i]["point"])
            or abs(float(q["normal_z"]) - 1) > 1e-6
        ):
            raise ValueError("Unmatched physical node impulse or bed normal")
        d = max(0.0, -float(w["selected_point_world_z_m"]))
        normal = float(q["normal_impulse_ns"]) / dt
        rows[i].append((step, d, normal))
        support[step] = support.get(step, 0.0) + normal
    report = []
    for i, (s, reference, samples) in enumerate(
        zip(sections, priors, rows, strict=True)
    ):
        if (
            len(samples) != 250
            or s["source_geometry_index"] != reference["source_geometry_index"]
        ):
            raise ValueError("Incomplete 32-step material history")
        depths = np.asarray([a[1] for a in samples])
        r = [v for v in columns if int(v["row"]) == i]
        area, ts, tf = [
            np.asarray([float(v[n]) for v in r]) for n in ("area_m2", "skin_m", "fat_m")
        ]
        c = sum(
            float(s[n]) for n in ("skin_compliance_m_per_n", "fat_compliance_m_per_n")
        )
        if (
            abs(c / spec["sections"][i]["linear_compliance_m_per_N"] - 1) > 1e-5
            or max(depths) > spec["sections"][i]["maximum_m"]
        ):
            raise ValueError("Sampled motion outside the installed force law")
        # The most compressed sample attains the greatest strain in every
        # column of this section because the two constitutive laws are monotone.
        worst = int(depths.argmax())
        d = depths[worst]
        lo = np.maximum(0.0, d - tf)
        hi = np.minimum(d, ts)
        for _ in range(48):
            u = (lo + hi) * 0.5
            high = stress(1 - u / ts, *mat[0]) > stress(1 - (d - u) / tf, *mat[1])
            hi = np.where(high, u, hi)
            lo = np.where(high, lo, u)
        u = (lo + hi) * 0.5
        ss = 1 - u / ts
        sf = 1 - (d - u) / tf
        f, _ = column_force(depths, area, ts, tf, mat)
        report.append(
            dict(
                section=i,
                source=int(s["source_geometry_index"]),
                columns=len(ts),
                step_at_peak=samples[worst][0],
                max_compression_m=float(d),
                minimum_skin_stretch=float(min(ss)),
                minimum_fat_stretch=float(min(sf)),
                columns_below_dense_stretch_floor_at_peak=int(
                    ((ss < 2 / 3) | (sf < 2 / 3)).sum()
                ),
                peak_elastic_force_N=float(max(f)),
                max_stock_fraction=float(max(d / (ts + tf))),
                uncompressed_stiffness_N_per_m=1 / c,
            )
        )
    surf = table(case / "resting-surface-audit.csv")
    coupled = table(case / "resting-coupled.csv")
    if not coupled or not surf:
        raise ValueError("Missing surface or physiology evidence")
    weight = float(surf[-1]["represented_body_mass_kg"]) * 9.81
    return dict(
        old_pack=read(control / "terminal.receipt.json")["pack_file_sha256"],
        new_pack=frame["pack_file_sha256"],
        launch=sha(case / "launch.json"),
        geometry=frame["pack_file_sha256"],
        seconds=frame["accepted_time_s"],
        steps=frame["accepted_step"],
        cold_wall_s=execute["cold_wall_s"],
        cold_factor=frame["accepted_time_s"] / execute["cold_wall_s"],
        weight_N=weight,
        mean_last_second_sampled_normal_N=float(
            np.mean([f for n, f in support.items() if n * dt >= 15.0])
        ),
        unique_columns_outside_dense_bounds_at_samples=sum(
            r["columns_below_dense_stretch_floor_at_peak"] for r in report
        ),
        minimum_fat_stretch=min(r["minimum_fat_stretch"] for r in report),
        minimum_skin_stretch=min(r["minimum_skin_stretch"] for r in report),
        max_geometry_vertices_below_tolerance=max(
            int(r["vertices_below_1mm"]) for r in surf
        ),
        minimum_sampled_skin_bed_gap_m=min(
            float(r["min_skin_bed_gap_m"]) for r in surf
        ),
        sections=report,
        scope="Released-body gap at 32-step cadence and series equilibrium within each parallel column. The count is the union of per-column stretch failures over samples; between-sample peaks are not measured.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("case", "source", "control"):
        p.add_argument(k, type=Path)
    p.add_argument(
        "--require-dense-stretch",
        action="store_true",
        help="Require each sampled column to meet the reference skin/fat material bounds",
    )
    args = p.parse_args()
    result = verify(args.case, args.source, args.control)
    print(json.dumps(result, indent=2, allow_nan=False))
    if (
        args.require_dense_stretch
        and result["unique_columns_outside_dense_bounds_at_samples"]
    ):
        raise SystemExit(
            "Dense tissue stretch validation failed in "
            + str(result["unique_columns_outside_dense_bounds_at_samples"])
            + " registered columns"
        )
