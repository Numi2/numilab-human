"""Measure physical load footprints on each scene's own clock and tissue representation."""

import csv, json, sys
from pathlib import Path
import numpy as np
from bind_posterior_equilibrium import coordinates
from fit_native_bed_barrier_load import force
from material_context import sha


def footprint(points, loads):
    loads = np.asarray(loads, dtype=float)
    if not np.isfinite(loads).all() or np.any(loads < 0) or loads.sum() <= 0:
        raise ValueError("Invalid normal traction")
    center = np.average(points, axis=0, weights=loads)
    rel = points - center
    cov = (rel * loads[:, None]).T @ rel / loads.sum()
    return dict(
        normal_N=float(loads.sum()),
        weighted_center_m=center.tolist(),
        horizontal_rms_radius_m=float(np.sqrt(np.trace(cov[:2, :2]))),
        effective_points=float(loads.sum() ** 2 / (loads @ loads)),
        peak_fraction=float(loads.max() / loads.sum()),
        active=int(np.sum(loads > 0)),
    )


def measure(compact, fine, case, split, bed, out):
    trace = {}
    for name in (
        "resting-com-support-impulses.csv",
        "resting-support-point-motion.csv",
        "resting-com-momentum-diagnostic.csv",
        "resting-local-bed-sections.csv",
    ):
        p = compact / name
        with p.open() as f:
            trace[name] = list(csv.DictReader(f))
    steps = {
        int(r["accepted_step"]): r
        for r in trace["resting-com-momentum-diagnostic.csv"]
        if float(r["time_s"]) >= 58
    }
    if max(steps) != 30000:
        raise ValueError("Expected the verified 60 s control")
    positions = {
        (int(r["accepted_step"]), int(r["contact_index"])): r
        for r in trace["resting-support-point-motion.csv"]
        if int(r["accepted_step"]) in steps
    }
    flux = {s: [] for s in steps}
    for r in trace["resting-com-support-impulses.csv"]:
        s = int(r["accepted_step"])
        if s not in steps or r["row_class"] != "external_support_contact":
            continue
        n = int(r["contact_index"])
        p = positions[s, n]
        if (r["body_index"], r["source_geometry_index"]) != (
            p["body_index"],
            p["source_geometry_index"],
        ):
            raise ValueError("Native point and contact identities differ")
        point = [float(p["point_world_" + dim + "_m"]) for dim in "xyz"]
        impulse = float(r["normal_impulse_ns"])
        flux[s].append((n, point, impulse / 0.002))
    compact_results = []
    means = {}
    sections = {
        int(r["contact_index"]): r for r in trace["resting-local-bed-sections.csv"]
    }
    for s, records in flux.items():
        if sorted(r[0] for r in records) != sorted(sections):
            raise ValueError("Omitted bed section")
        points = np.array([r[1] for r in records])
        loads = np.array([r[2] for r in records])
        r = footprint(points, loads)
        if (
            abs(r["normal_N"] - float(steps[s]["normal_force_last_physical_step_n"]))
            > 0.01
        ):
            raise ValueError(
                (
                    "Native support normalization differs",
                    s,
                    r["normal_N"],
                    steps[s]["normal_force_last_physical_step_n"],
                )
            )
        compact_results.append(r)
        for n, pt, f in records:
            means.setdefault(n, []).append(f)
    support = json.loads((bed / "load.json").read_text())
    massfile = case / "fine-mass-routed/current-density-closed-flow.npz"
    if (
        sha(massfile)
        != json.loads((case / "supported-registration.json").read_text())["source_mass"]
    ):
        raise ValueError("Source mass mismatch")
    outer = np.load(split / "physical.npz")["Orows"]
    mass = np.load(massfile)["nodal_mass_kg"]
    dense = {}
    for name in ["reference", "accepted"]:
        pos = coordinates(fine / ("physical-" + name + ".node"))
        gap = pos[outer, 2]
        used = outer[gap < support["band_m"]]
        loads = force(pos[used, 2], mass[used], support["dt_s"], support["band_m"])
        dense[name] = footprint(pos[used], loads)
    live = json.loads((fine / "physical-response.json").read_text())
    sig = {n: sha(compact / n) for n in trace}
    res = dict(
        compact_scene=sha(compact / "launch.json"),
        compact_samples=len(steps),
        compact_interval_s=[
            min(float(r["time_s"]) for r in steps.values()),
            max(float(r["time_s"]) for r in steps.values()),
        ],
        compact_mean={
            n: np.mean([v[n] for v in compact_results], axis=0).tolist()
            for n in compact_results[0]
        },
        compact_section_loads=[
            dict(
                index=n,
                mean_N=float(np.mean(vals)),
                area_m2=float(sections[n]["projected_area_m2"]),
            )
            for n, vals in sorted(means.items())
        ],
        inputs=sig,
        fine_scene=sha(fine / "launch.json"),
        fine_states=dense,
        fine_native_first_step_N=live["reported_dynamic_bed_N"],
        scope="Late 32-section native bed reactions and the fine-mesh stationary barrier at its registered start and first accepted step. The sampled clocks and discretizations differ; no pointwise force parity is inferred.",
    )
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(res["compact_mean"], dense)


if __name__ == "__main__":
    measure(*(Path(p).resolve() for p in sys.argv[1:]))
