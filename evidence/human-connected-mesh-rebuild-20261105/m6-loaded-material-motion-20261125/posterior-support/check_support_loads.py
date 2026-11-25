"""Find the normal load that each connected bed contact must carry to balance gravity."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def clipped_triangle(points, gaps, width, normal, offset):
    polygon = []
    for i in range(3):
        j = (i + 1) % 3
        if gaps[i] < width:
            polygon.append(points[i])
        if (gaps[i] < width) != (gaps[j] < width):
            t = (width - gaps[i]) / (gaps[j] - gaps[i])
            polygon.append(points[i] + t * (points[j] - points[i]))
    polygon = np.array(polygon)
    return polygon - np.outer(polygon @ normal - offset, normal)


def contact_loads(data, positions, intercept, body_weight):
    normal = data["normal"]
    normal_load = body_weight * normal[2]
    gaps = positions @ normal - float(data["offset"])
    width = float(data["width"])
    faces = data["face"][gaps[data["face"]].min(axis=1) < width]
    # Connect faces through the positive-area contact patch. An inactive vertex
    # shared far from the plane must not connect distant footprints.
    inside = gaps < width
    edge = []
    for face in faces:
        active = face[inside[face]]
        edge.extend((active[0], i) for i in active)
    i, j = np.array(edge).T
    graph = coo_matrix(
        (np.ones(len(i)), (i, j)), shape=(len(positions), len(positions))
    )
    _, labels = connected_components(graph, directed=False)
    components = {}
    for face in faces:
        label = labels[face[inside[face]][0]]
        polygon = clipped_triangle(
            positions[face], gaps[face], width, normal, float(data["offset"])
        )
        area = sum(
            np.linalg.norm(np.cross(p - polygon[0], q - polygon[0])) * 0.5
            for p, q in zip(polygon[1:-1], polygon[2:])
        )
        if area <= 0:
            continue
        group = components.setdefault(label, {"points": [], "area": 0.0})
        group["points"].extend(polygon)
        group["area"] += area
    vertices, owners = [], []
    groups = sorted(components)
    for group in groups:
        points = np.unique(components[group]["points"], axis=0)
        vertices.extend(points)
        owners.extend([group] * len(points))
    vertices = np.array(vertices)
    owners = np.array(owners)
    # Finite planar support. The normal center of pressure must coincide with
    # the gravity line's intersection with the bed. Tangential forces in this
    # plane cannot generate a tipping moment about either in-plane axis.
    xy = (vertices - intercept)[:, :2]
    scale = np.max(np.abs(xy), axis=0)
    if np.any(scale == 0) or abs(normal[2]) < 1e-6:
        raise ValueError("Singular bed projection")
    matrix = np.vstack((np.ones(len(vertices)), (xy / scale).T))
    b = np.array([1.0, 0.0, 0.0])
    rows = []
    for group in groups:
        mask = (owners == group).astype(float)
        values = []
        for sign in (1.0, -1.0):
            sol = linprog(
                sign * mask, A_eq=matrix, b_eq=b, bounds=(0, None), method="highs"
            )
            if not sol.success or np.max(abs(matrix @ sol.x - b)) > 1e-7:
                raise ValueError("No balanced support for this contact footprint")
            dual = sol.eqlin.marginals
            if (
                np.max(matrix.T @ dual - sign * mask) > 1e-7
                or abs(dual @ b - sol.fun) > 1e-7
            ):
                raise ValueError("Load bound lacks a matching dual certificate")
            values.append(float((mask @ sol.x) * normal_load))
        ids = data["node"][(labels == group) & inside]
        area = components[group]["area"]
        rows.append(
            dict(
                active_nodes=ids.tolist(),
                projected_patch_area_mm2=area * 1e6,
                normal_load_interval_N=values,
                minimum_mean_pressure_Pa=values[0] / area,
            )
        )
    return rows


if __name__ == "__main__":
    work, support = map(Path, sys.argv[1:])
    with np.load(work / "surface.npz") as bundle:
        data = dict(bundle)
    ref = json.loads(support.read_bytes())
    positions = np.loadtxt(work / "current.node", skiprows=1, usecols=(1, 2, 3))[
        data["node"]
    ]
    rows = {
        name: contact_loads(
            data, x, np.array(ref["gravity_intercept_m"]), ref["weight_N"]
        )
        for name, x in [("original", data["x"]), ("pressure-fit", positions)]
    }
    rows["scope"] = (
        "Static feasibility on the full clipped opening bed patches at the registered geometry. "
        "Force in a particular internal tissue is not determined by bed pressure. "
        "The area is the sum of projected active triangle pieces."
    )
    (work / "support-load-intervals.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(rows["pressure-fit"])
