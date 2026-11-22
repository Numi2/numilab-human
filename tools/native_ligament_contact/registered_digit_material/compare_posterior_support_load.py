"""Compare the registered refinements under the same normal load and retain the free pitch/roll residual."""

import argparse
import json
from pathlib import Path

import numpy as np
from material_bed_release import stationary_bed
from material_context import sha
from scipy.optimize import brentq
from support_patch import prepare

from refine_posterior_contact import pullback


def solve_load(points, mass, dt, width, weight):
    # This translates the entire body rigidly along the authored bed normal.
    # All normal gaps and tangential moment arms are evaluated in world metres.
    z = points[:, 2].astype(float)
    if (
        points.shape != (len(mass), 3)
        or not len(mass)
        or not np.isfinite(mass).all()
        or not np.all(mass > 0)
        or not np.isfinite(points).all()
        or not np.isfinite([dt, width, weight]).all()
        or min(dt, width, weight) <= 0
    ):
        raise ValueError("Invalid material or applied load")

    def rows(h):
        active = (z + h) < width
        result = np.zeros((3, len(mass)))
        result[:, active] = stationary_bed(z[active] + h, mass[active], dt, width)
        return result

    lower = width * 2e-4 - float(z.min())
    if rows(lower)[1].sum() <= weight:
        raise ValueError("No regular strictly separated equilibrium")
    h = brentq(
        lambda a: rows(a)[1].sum() - weight,
        lower,
        width - float(z.min()),
        xtol=5e-16,
        rtol=8e-14,
    )
    energy, normal, curvature = rows(h)
    tangent = np.column_stack((np.ones(len(mass)), points[:, :2]))
    K = tangent.T @ (curvature[:, None] * tangent)
    return h, energy, normal, K


def measure(prep, variants, run, material, transfer, owner, peaks):
    original = prepare(run, material, transfer, owner)
    previous = json.loads(peaks.read_text())
    if previous["receipt"] != sha(run / "terminal.receipt.json"):
        raise ValueError("Wrong weight reference")
    weight = previous["weight_N"]
    width = original["width"]
    dt = original["dt"]
    manifest = json.loads((prep / "source.json").read_text())
    if (
        sha(prep / "patch.npz") != manifest["topology_patch_sha256"]
        or manifest["registered"] != original["source"]
    ):
        raise ValueError("Unmatched material patch")
    with np.load(prep / "patch.npz") as z:
        data = dict(z)
    classes = data["layer"] == 1
    other = np.flatnonzero(
        (original["layer"] == 1)
        & ~np.isin(np.arange(len(original["layer"])), data["nodes"])
    )
    other_mass = original["mass"][other]
    reports = []
    previous_by_pose = {}
    for path in sorted(
        variants.glob("level-*.npz"), key=lambda path: int(path.stem.split("-")[-1])
    ):
        with np.load(path) as gs:
            s = dict(gs)
        child = s["tetrahedra"]
        pairs = s["midpoint_parents"]
        nc = len(pairs)
        nv = len(data["nodes"])
        m = np.r_[data["mass"], np.zeros(nc)]
        for i in range(4):
            np.add.at(m, data["t"][:, i], -data["cell_mass"] / 4)
            np.add.at(m, child[:, i], s["child_mass_kg"] / 4)
        active = np.r_[classes, np.ones(nc, bool)]
        metrics = []
        for label, key in (("loaded", "current"), ("accepted", "end")):
            local = np.empty((nv + nc, 3))
            local[:nv] = original[key][data["nodes"]]
            for i, (a, b) in enumerate(pairs):
                local[nv + i] = ((local[a] + local[b]) * 0.5).astype("f4")
            if key == "current" and not np.array_equal(local, s["loaded"]):
                raise ValueError("Midpoint replay differs")
            points = np.vstack((local[active], original[key][other]))
            m_outer = np.r_[m[active], other_mass]
            h, e, n, K = solve_load(points, m_outer, dt, width, weight)
            row = np.zeros(len(local))
            row[active] = n[: np.sum(active)]
            source = np.r_[pullback(row, pairs), n[np.sum(active) :]]
            center = (n @ points[:, :2]) / n.sum()
            if abs(n.sum() - weight) > 1e-5 or np.min(points[:, 2] + h) <= 0:
                raise ValueError("Static normal force or gap failed")
            last = previous_by_pose.get(key)
            metrics.append(
                {
                    "pose": label,
                    "rigid_height_change_m": h,
                    "minimum_gap_m": float((points[:, 2] + h).min()),
                    "equilibrium_force_N": float(n.sum()),
                    "active_nodes": int(np.sum(n > 0)),
                    "elastic_barrier_energy_J": float(e.sum()),
                    "vertical_tangent_N_per_m": float(K[0, 0]),
                    "center_of_pressure_xy_m": center.tolist(),
                    "peak_native_node_N": float(n.max()),
                    "peak_projected_source_node_N": float(source.max()),
                    "relative_source_force_change_from_previous": None
                    if last is None
                    else float(
                        np.linalg.norm(source - last[0]) / np.linalg.norm(last[0])
                    ),
                    "center_of_pressure_change_mm": None
                    if last is None
                    else float(1e3 * np.linalg.norm(center - last[1])),
                    "force_closed_on_newly_active_outside_patch_N": float(
                        n[np.sum(active) :].sum()
                    ),
                }
            )
            previous_by_pose[key] = (source, center)
        reports.append({"variant": path.name, "sha256": sha(path), "results": metrics})
    return {
        "registered": manifest["registered"],
        "source_receipt": previous["receipt"],
        "weight_N": weight,
        "levels": reports,
        "scope": "All registered outer nodes participate. At each resolution the same body weight is applied and the vertical root displacement is solved. Orientation, material shape and articulated pose are held independently at the two recorded configurations. CoP and projected source reactions are out-of-fit observables; this is not a released dynamic step.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prep", "variants", "run", "material", "transfer", "owner", "peaks"):
        parser.add_argument(key, type=Path)
    print(json.dumps(measure(**vars(parser.parse_args())), indent=2, allow_nan=False))
