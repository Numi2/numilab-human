"""Necessary peak pressure for gravity balance on the existing exterior within a bed-distance band."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from check_support_loads import clipped_triangle


def minimum_peak(faces, points, normal, offset, depth, gravity_point, weight):
    gaps = points @ normal - offset
    chosen = faces[gaps[faces].min(axis=1) < depth]
    areas, coordinates, ids = [], [], []
    for face in chosen:
        poly = clipped_triangle(points[face], gaps[face], depth, normal, offset)
        area = sum(
            abs(np.dot(normal, np.cross(x - poly[0], y - poly[0]))) * 0.5
            for x, y in zip(poly[1:-1], poly[2:])
        )
        if area > 1e-18:
            ids.extend([len(areas)] * len(poly))
            areas.append(area)
            coordinates.extend(poly)
    area = np.array(areas)
    center = (np.array(coordinates) - gravity_point)[:, :2]
    variables = len(center)
    total = area.sum()
    # Unit normal force. Minimize peak pressure divided by the uniform average.
    eq = sparse.csr_matrix(
        np.column_stack((np.vstack((np.ones(variables), center.T)), np.zeros(3)))
    )
    upper = sparse.hstack(
        (
            sparse.csr_matrix(
                (np.ones(variables), (ids, np.arange(variables))),
                shape=(len(area), variables),
            ),
            (-area / total)[:, None],
        )
    )
    obj = np.r_[np.zeros(variables), 1.0]
    res = linprog(
        obj,
        A_ub=upper,
        b_ub=np.zeros(len(area)),
        A_eq=eq,
        b_eq=[1.0, 0.0, 0.0],
        bounds=(0, None),
        method="highs",
    )
    if (
        not res.success
        or np.max(abs(eq @ res.x - [1, 0, 0])) > 1e-7
        or np.max(upper @ res.x) > 1e-7
    ):
        raise ValueError("Unbalanced support footprint")
    return dict(
        band_m=depth,
        projected_area_m2=float(total),
        active_triangles=len(area),
        mean_pressure_lower_bound_Pa=float(weight / total),
        peak_pressure_lower_bound_Pa=float(res.fun * weight / total),
    )


if __name__ == "__main__":
    work = Path(sys.argv[1])
    config = json.loads((work / "expanded-motion/launch.json").read_bytes())
    env = config["environment"]
    path = Path(env["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    topology = (
        Path(
            env["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE"]
        ).parent
        / "material-walls.npz"
    )
    with np.load(topology) as w:
        faces = w["O"]
    xyz = np.loadtxt(path, skiprows=1, usecols=(1, 2, 3))
    with np.load(work / "surface.npz") as b:
        n = b["normal"]
        h = float(b["offset"])
        width = float(b["width"])
    support = json.loads(Path(sys.argv[2]).read_bytes())
    load = support["weight_N"] * n[2]
    samples = [
        minimum_peak(
            faces,
            xyz,
            n,
            h,
            width + extra,
            np.array(support["gravity_intercept_m"]),
            load,
        )
        for extra in (0, 0.00025, 0.001, 0.003, 0.005, 0.010, 0.020, 0.040)
    ]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = dict(
        loaded_node_sha256=digest(path),
        wall_topology_sha256=digest(topology),
        gravity_normal_N=load,
        samples=samples,
        scope="Opening skin only. Extra depth admits potential posterior surface without advancing, translating or constraining the body. "
        "Projected area is summed even where surfaces could overlap. Each triangle may freely place its force anywhere on its clipped polygon. "
        "These relaxations give a necessary lower bound on peak normal pressure; actual contact can require more.",
    )
    (work / "contact-area-screen.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        [
            (
                round(r["band_m"] * 1000, 3),
                round(r["projected_area_m2"] * 1e4, 2),
                round(r["peak_pressure_lower_bound_Pa"] / 1e3, 2),
            )
            for r in samples
        ]
    )
