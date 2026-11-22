"""Measure the force and consistent tangent of the fixed-bed rule on retained human refinements."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from compare_bed_sampling import integrate
from fit_native_bed_barrier_load import force
from material_context import sha
from scipy.optimize import brentq

from refine_posterior_contact import pullback


def area(points, triangles):
    p = points[triangles]
    return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)


def extend(start, edges, quantize=False):
    out = np.empty((len(start) + len(edges), *start.shape[1:]))
    out[: len(start)] = start
    for i, (a, b) in enumerate(edges, len(start)):
        if not (0 <= a < i and 0 <= b < i):
            raise ValueError("Unresolved midpoint material lineage")
        v = (out[a] + out[b]) * 0.5
        out[i] = v.astype("f4") if quantize else v
    return out


def vertical_shift(points, dz):
    pose = points.copy()
    pose[:, 2] += dz
    return pose


def vertical_imbalance(dz, points, faces, area, density, dt, width, weight):
    result = integrate(vertical_shift(points, dz), faces, area, density, dt, width, 4)
    return result[0].sum() - weight


def relative(actual, target):
    return float(np.linalg.norm(actual - target) / np.linalg.norm(target))


def measure(root, control, expected, gpu_dir=None):
    if gpu_dir:
        from check_face_operator import NativeFace

        gpu = NativeFace(gpu_dir)
    prepared = root / "prepared"
    source = json.loads((prepared / "source.json").read_text())
    previous = json.loads(expected.read_text())
    if (
        source["registered"] != previous["pins"]
        or source["matched_receipt"] != previous["receipt"]
        or sha(prepared / "patch.npz") != source["topology_patch_sha256"]
    ):
        raise ValueError("Different mesh or native frame")
    with np.load(prepared / "patch.npz") as z:
        d = dict(z)
    run = json.loads((control / "launch.json").read_text())
    env = run["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"

    def pins(path, key):
        p = Path(path)
        if sha(p) != key:
            raise ValueError("Material source hash mismatch")
        return p

    ref_path = pins(
        env[prefix + "TETGEN_NODE"], source["registered"]["reference_sha256"]
    )
    x = np.loadtxt(ref_path, skiprows=1, usecols=(1, 2, 3)).astype("f4").astype(float)
    mapping = pins(
        env[prefix + "COMBINED_NODE_MAP_CSV"], env[prefix + "COMBINED_NODE_MAP_SHA256"]
    )
    with mapping.open() as f:
        rows = sorted(
            (int(r["registered_surface_vertex_row"]), int(r["joint_node_id"]) - 1)
            for r in csv.DictReader(f)
            if r["registered_surface_layer"] == "O"
        )
    indices, outer = np.asarray(rows).T
    if not np.array_equal(indices, np.arange(len(outer))) or len(
        np.unique(outer)
    ) != len(outer):
        raise ValueError("Ambiguous registered outer material")
    fpath = pins(source["registered_surface"], source["surface_sha256"])
    faces = outer[
        np.loadtxt(fpath, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype=int)
    ]
    if not np.array_equal(faces[d["face_ids"]], d["nodes"][d["f"]]):
        raise ValueError("Different material surface")
    dual = np.bincount(
        faces.ravel(), weights=np.repeat(area(x, faces) / 3, 3), minlength=len(x)
    )
    local_dual = dual[d["nodes"]]
    o = d["layer"] == 1
    dt = float(d["dt"])
    w = float(d["width"])
    density = np.zeros(len(o))
    density[o] = (
        force(0.85 * w, d["mass"][o], dt, w)
        / force(0.85 * w, 1.0, dt, w)
        / local_dual[o]
    )
    # Smooth, nonuniform probe in physical space, independent of the triangulation.
    direction = np.cos(13 * d["x"][:, 0]) * np.sin(3 * d["x"][:, 1]) + 0.43
    # Bound every excluded outer face over the entire permitted rigid translation.
    current = Path(env["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    endpoint = control / "physical-accepted.node"
    frame = json.loads((control / "verified-endpoint.json").read_text())
    pins(current, source["registered"]["installed_pose_sha256"])
    pins(endpoint, frame["physical_nodes_sha256"])
    excluded = np.setdiff1d(np.arange(len(faces)), d["face_ids"])
    outside = {}
    for label, path in [("initial", current), ("endpoint", endpoint)]:
        z = np.loadtxt(path, skiprows=1, usecols=(3,))
        if label == "initial":
            z = z.astype("f4").astype(float)
        outside[label] = float(z[faces[excluded]].min())
    target_doc = json.loads((root / "matched-weight.json").read_text())
    if target_doc["source_receipt"] != previous["receipt"]:
        raise ValueError("Different weight experiment")
    target = target_doc["weight_N"]
    base_at_load = {}
    base = {}
    result = []
    recovery = json.loads((root / "recovered-release.json").read_text())[
        "verified_member_sha256"
    ]
    paths = sorted((root / "guarded").glob("level-*.npz"))
    if len(paths) != 4:
        raise ValueError("Incomplete retained refinement family")
    for path in paths:
        if sha(path) != recovery[str(path.relative_to(root))]:
            raise ValueError("Changed registered refinement")
        with np.load(path) as s:
            edges = s["midpoint_parents"]
            faces = s["outer_triangles"]
            ref = s["reference"].astype(float)
            loaded = s["loaded"].astype(float)
            if not np.array_equal(s["local_source_nodes"], d["nodes"]):
                raise ValueError("Changed material coordinates")
        rho = extend(density, edges)
        v = extend(direction, edges)
        A = area(ref, faces)
        phases = {}
        for name, key in [("initial", "y"), ("endpoint", "end")]:
            y = extend(d[key].astype(float), edges, True)
            if name == "initial" and not np.array_equal(loaded, y):
                raise ValueError("Different released pose")
            f, m, k = integrate(y, faces, A, rho, dt, w, 4, normal_direction=v)
            coarse_f = pullback(f, edges)
            coarse_k = pullback(k, edges)
            if name not in base:
                base[name] = (coarse_f, coarse_k)
                old = previous[name]["rules"][-1]["force_N"]
                if abs(f.sum() - old) > 1e-4:
                    raise ValueError(f"Base quadrature mismatch: {f.sum()} vs {old}")
            h = 1e-5 * w
            yp = y.copy()
            ym = y.copy()
            yp[:, 2] += h * v
            ym[:, 2] -= h * v
            plus = integrate(yp, faces, A, rho, dt, w, 4)[0]
            minus = integrate(ym, faces, A, rho, dt, w, 4)[0]
            fd = -(plus - minus) / (2 * h)
            f0, k0 = base[name]
            parity = relative(k, fd)
            if parity > 2e-5 or not v @ k > 0:
                raise ValueError(f"Invalid support tangent {parity}")
            phases[name] = {
                "normal_N": float(f.sum()),
                "cop_xy_m": (m[:2] / f.sum()).tolist(),
                "directional_stiffness_N_m": float(v @ k),
                "tangent_vs_force_difference_relative": parity,
                "source_reaction_change_relative": relative(coarse_f, f0),
                "source_stiffness_action_change_relative": relative(coarse_k, k0),
            }
            if gpu_dir:
                phases[name]["M6"] = gpu.compare(y, faces, A, rho, dt, w, v, f, k)
            # Balance only the normal resultant. Tangential coordinates, CoP and
            # local nodal reactions are independent measurements.
            gap_min = float(y[np.unique(faces), 2].min())
            lo = -0.95 * gap_min
            if outside[name] + lo <= w:
                raise ValueError(
                    "Load probe could activate a face outside its inventory"
                )
            hi = w

            height = brentq(
                vertical_imbalance,
                lo,
                hi,
                args=(y, faces, A, rho, dt, w, target),
                xtol=5e-18,
            )
            force_at_load, moment_at_load, vertical = integrate(
                vertical_shift(y, height),
                faces,
                A,
                rho,
                dt,
                w,
                4,
                normal_direction=np.ones(len(y)),
            )
            transfer = pullback(force_at_load, edges)
            if abs(force_at_load.sum() - target) > 1e-5 or vertical.sum() <= 0:
                raise ValueError("No stable normal load match")
            center = moment_at_load[:2] / force_at_load.sum()
            if name not in base_at_load:
                base_at_load[name] = (center, transfer)
            c0, t0 = base_at_load[name]
            phases[name]["matched_weight"] = {
                "rigid_vertical_shift_m": height,
                "minimum_gap_m": gap_min + height,
                "minimum_excluded_gap_m": outside[name] + height,
                "normal_N": float(force_at_load.sum()),
                "vertical_tangent_N_m": float(vertical.sum()),
                "cop_xy_m": center.tolist(),
                "cop_change_mm": 1e3 * float(np.linalg.norm(center - c0)),
                "source_reaction_change_relative": relative(transfer, t0),
            }
            if gpu_dir:
                phases[name]["matched_weight"]["M6"] = gpu.compare(
                    vertical_shift(y, height),
                    faces,
                    A,
                    rho,
                    dt,
                    w,
                    np.ones(len(y)),
                    force_at_load,
                    vertical,
                )
        result.append({"mesh": sha(path), "added_nodes": len(edges), "phases": phases})
    return {
        "source": source["topology_patch_sha256"],
        "prior_face_rule": sha(expected),
        "levels": result,
    }


if __name__ == "__main__":
    print(
        json.dumps(measure(*(Path(s) for s in sys.argv[1:])), indent=2, allow_nan=False)
    )
