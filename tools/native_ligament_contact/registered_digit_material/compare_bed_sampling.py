"""Compare the installed vertex rule to face integration on the same fixed native geometry."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from fit_native_bed_barrier_load import force
from material_context import sha
from support_patch import prepare


def clipped_simplices(z, width):
    pieces, ids = [], []
    eye = np.eye(3)
    for face in np.flatnonzero(z.min(axis=1) < width):
        height = z[face]
        vertices = []
        for i in range(3):
            j = (i + 1) % 3
            if height[i] < width:
                vertices.append(eye[i])
            if (height[i] < width) != (height[j] < width):
                t = (width - height[i]) / (height[j] - height[i])
                vertices.append((1 - t) * eye[i] + t * eye[j])
        for i in range(1, len(vertices) - 1):
            pieces.append([vertices[0], vertices[i], vertices[i + 1]])
            ids.append(face)
    return np.asarray(ids, dtype=int), np.asarray(pieces).reshape(-1, 3, 3)


def integrate(x, faces, area, density, dt, width, order, *, normal_direction=None):
    # The optional third return is the positive energy Hessian applied to a
    # material normal displacement. The bed force derivative has opposite sign.
    if normal_direction is not None:
        normal_direction = np.asarray(normal_direction)
        if (
            normal_direction.shape != (len(x),)
            or not np.isfinite(normal_direction).all()
        ):
            raise ValueError("Invalid material direction")
        action = np.zeros(len(x))

    ids, bary = clipped_simplices(x[faces, 2], width)
    result = np.zeros(len(x))
    if not len(ids):
        return (
            (result, np.zeros(3), action)
            if normal_direction is not None
            else (result, np.zeros(3))
        )
    sub_area = np.abs(np.linalg.det(bary)) * area[ids]
    point, weight = np.polynomial.legendre.leggauss(order)
    a, b = np.meshgrid((point + 1) / 2, (point + 1) / 2, indexing="ij")
    u, v = a.ravel(), b.ravel()
    weights = (np.outer(weight, weight) / 2).ravel() * (1 - u)
    simplex = np.stack([1 - u - (1 - u) * v, u, (1 - u) * v], axis=1)
    basis = simplex @ bary
    gaps = np.einsum("fqi,fi->fq", basis, x[faces[ids], 2])
    coeff = np.einsum("fqi,fi->fq", basis, density[faces[ids]])
    traction = (
        force(gaps, 1.0, dt, width) * coeff * weights[None, :] * sub_area[:, None]
    )
    nodal = np.einsum("fq,fqi->fi", traction, basis)
    np.add.at(result, faces[ids].ravel(), nodal.ravel())
    points = basis @ x[faces[ids]]
    centroid = np.einsum("fq,fqk->k", traction, points)
    if not np.isclose(
        result.sum(), traction.sum(), rtol=3e-14, atol=1e-13
    ) or not np.allclose(result @ x, centroid, rtol=2e-12, atol=1e-11):
        raise ValueError("Integrated plane load failed resultant or moment balance")
    if normal_direction is None:
        return result, centroid
    # Both the force and its first derivative vanish at the clipping boundary.
    # The stationary radial IPC law uses constant force below its gap floor.
    distance = np.maximum(gaps, 1e-4 * width)
    squared = distance**2
    width_squared = width**2
    ratio = (squared - width_squared) / squared
    log_gap = np.log(squared / width_squared)
    unit_k = 10.0 / max(dt**2 * width_squared, 1e-20)
    first = -unit_k * squared * (2 * ratio * log_gap + ratio**2)
    second = -unit_k * (2 * log_gap + 4 * ratio - ratio**2)
    curvature = np.where(
        gaps < 1e-4 * width, 0.0, np.maximum(4 * squared * second + 2 * first, 0.0)
    )
    dq = np.einsum("fqi,fi->fq", basis, normal_direction[faces[ids]])
    variation = coeff * weights[None, :] * sub_area[:, None] * curvature * dq
    local_action = np.einsum("fq,fqi->fi", variation, basis)
    np.add.at(action, faces[ids].ravel(), local_action.ravel())
    return result, centroid, action


def compare(run, material, transfer, owner, bed_path):
    data = prepare(run, material, transfer, owner)
    bed = json.loads(bed_path.read_text())
    if sha(run / "terminal.receipt.json") != bed["receipt"]:
        raise ValueError("Different force frame")
    env = json.loads((run / "launch.json").read_text())["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    keys = ("COMBINED_NODE_MAP", "SURFACE_O_FACES")
    mapping, facet = [Path(env[prefix + k + "_CSV"]) for k in keys]
    if any(
        sha(p) != env[prefix + k + "_SHA256"] for k, p in zip(keys, (mapping, facet))
    ):
        raise ValueError("Unverified geometry")
    with mapping.open() as f:
        pairs = sorted(
            (int(r["registered_surface_vertex_row"]), int(r["joint_node_id"]) - 1)
            for r in csv.DictReader(f)
            if r["registered_surface_layer"] == "O"
        )
    ids, outer = np.asarray(pairs).T
    if not np.array_equal(ids, np.arange(len(ids))) or not np.array_equal(
        np.sort(outer), np.flatnonzero(data["layer"] == 1)
    ):
        raise ValueError("Ambiguous boundary")
    local = np.loadtxt(facet, skiprows=1, delimiter=",", usecols=(1, 2, 3), dtype=int)
    rest = data["ref"][outer][local]
    area = 0.5 * np.linalg.norm(
        np.cross(rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]), axis=1
    )
    dual = np.bincount(
        local.ravel(), weights=np.repeat(area / 3, 3), minlength=len(outer)
    )
    if np.min(dual) <= 0 or np.min(area) <= 0:
        raise ValueError("Unresolved boundary material measure")
    w, dt = data["width"], data["dt"]
    scale = force(0.85 * w, data["mass"][outer], dt, w) / force(0.85 * w, 1.0, dt, w)
    density = scale / dual
    out = {
        "pins": data["source"],
        "receipt": bed["receipt"],
        "mass_flow": json.loads(transfer.read_text())["mass"],
        "rule": "The authored nodal barrier coefficients are distributed over the reference material surface using their barycentric dual areas. Gravity, tissue deformation and the floor are identical for both measurements. No simulation steps or force changes are applied.",
    }
    for key, source in [("initial", "current"), ("endpoint", "end")]:
        x = data[source][outer].astype(float)
        f = np.zeros(len(x))
        active = x[:, 2] < w
        f[active] = force(x[active, 2], data["mass"][outer][active], dt, w)
        target = bed[key + "_barrier_N"]
        if abs(f.sum() - target) > 2e-5:
            raise ValueError(
                f"Stationary nodal barrier differs from the installed replay {f.sum()} {target}"
            )
        row = {
            "native_vertex_replay_N": float(f.sum()),
            "witness_local": int(np.argmax(f)),
            "rules": [],
        }
        last = None
        for order in (2, 4, 8):
            integrated, center = integrate(x, local, area, density, dt, w, order)
            row["rules"].append(
                {
                    "order": order,
                    "force_N": float(integrated.sum()),
                    "max_node_N": float(integrated.max()),
                    "max_node": int(outer[np.argmax(integrated)]),
                    "force_at_native_peak_N": float(integrated[np.argmax(f)]),
                    "max_change_from_previous_N": None
                    if last is None
                    else float(np.max(abs(last - integrated))),
                    "center_of_pressure_xy_m": (center[:2] / integrated.sum()).tolist(),
                }
            )
            last = integrated
        out[key] = row
    return out


if __name__ == "__main__":
    print(
        json.dumps(
            compare(*(Path(s).resolve() for s in sys.argv[1:])),
            indent=2,
            allow_nan=False,
        )
    )
