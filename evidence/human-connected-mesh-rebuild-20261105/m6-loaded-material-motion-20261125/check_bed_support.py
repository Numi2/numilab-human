"""Measure the loaded opening contact footprint and required friction from a native run."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def record(lines, key):
    line, = (line for line in lines if line.startswith(key + "="))
    return dict(word.split("=", 1) for word in line.split() if "=" in word)


def polygon(points, indices):
    try:
        hull = ConvexHull(points[indices])
    except QhullError:
        return False, None, []
    vertices = points[indices[hull.vertices]]
    edges = np.roll(vertices, -1, axis=0) - vertices
    fractions = -np.einsum("ij,ij->i", vertices, edges) / np.einsum("ij,ij->i", edges, edges)
    closest = vertices + np.clip(fractions, 0, 1)[:, None] * edges
    inside = np.max(hull.equations[:, -1]) <= 0
    return bool(inside), float(np.min(np.linalg.norm(closest, axis=1))), indices[hull.vertices]


def measure(run):
    launch = json.loads((run / "launch.json").read_bytes())
    lines = (run / "stdout.log").read_text().splitlines()
    bed = record(lines, "resting_fem_skin_bed_source")
    registration = record(lines, "registered_loaded_material_install")
    gravity_line, = (line for line in lines if line.startswith("initial_gravity_total="))
    body = json.loads(gravity_line.split("=", 1)[1])
    pack = Path(launch["argv"][1])
    data = pack.read_bytes()
    identity = hashlib.sha256(data).hexdigest()
    if identity != launch["pins"][str(pack)] or struct.unpack_from("<8sII", data) != (b"NHRIGID2", 1, 5):
        raise ValueError("Rigid pack identity or ABI mismatch")
    world_offset = struct.calcsize("<8s10I32s")
    gravity = np.array(struct.unpack_from("<3f", data, world_offset + struct.calcsize("<16I")))
    normal = np.fromstring(bed["normal"], sep=",")
    plane = float(registration["fixed_bed_offset_m"]) * normal[2]
    if abs(plane - float(bed["plane_offset_m"])) > 1e-10 or abs(normal @ normal - 1) > 1e-9:
        raise ValueError("Inconsistent bed plane")
    env = launch["environment"]
    positions = np.loadtxt(env["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"], skiprows=1, usecols=(1, 2, 3))
    wall_directory = Path(env["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE"]).parent
    with np.load(wall_directory / "material-walls.npz") as walls:
        outer = np.unique(walls["O"])
    gap = positions[outer] @ normal - plane
    width = float(registration["ipc_width_m"])
    eligible = np.flatnonzero(gap < width)
    if gap.min() < 0 or len(eligible) != int(registration["outer_contact_nodes"]):
        raise ValueError("Opening footprint differs from the installed bed contact owner")
    center = np.array(body["center_m"])
    gravity_normal = gravity @ normal
    intercept = center + (plane - center @ normal) / gravity_normal * gravity
    tangent = np.cross(normal, [0.0, 1.0, 0.0])
    tangent /= np.linalg.norm(tangent)
    basis = np.column_stack((tangent, np.cross(normal, tangent)))
    points = (positions[outer] - gap[:, None] * normal - intercept) @ basis
    inside, margin, hull = polygon(points, eligible)

    # Adding contacts can only enlarge the convex hull.
    order = np.argsort(gap, kind="stable")
    low, high = 3, len(order)
    if not polygon(points, order)[0]:
        raise ValueError("Gravity line lies outside the complete outer-node footprint")
    while low < high:
        middle = (low + high) // 2
        if polygon(points, order[:middle])[0]:
            high = middle
        else:
            low = middle + 1
    return {
        "scope": "Opening geometry only. The gravity intercept and friction demand test static support feasibility on the actual bed plane.",
        "source_rigid_sha256": identity,
        "mass_kg": body["mass_kg"],
        "gravity_m_s2": gravity.tolist(),
        "weight_N": float(body["mass_kg"] * np.linalg.norm(gravity)),
        "minimum_static_friction_coefficient": float(np.linalg.norm(gravity - gravity_normal * normal) / abs(gravity_normal)),
        "rigid_material_static_friction": float(bed["rigid_material_static_friction"]),
        "com_world_m": center.tolist(),
        "gravity_intercept_m": intercept.tolist(),
        "registered_outer_vertices": len(outer),
        "contact_width_m": width,
        "eligible_outer_nodes": outer[eligible].tolist(),
        "eligible_hull_nodes": outer[hull].tolist(),
        "gravity_inside_eligible_hull": inside,
        "distance_gravity_to_hull_edge_m": margin,
        "first_covering_gap_m": float(gap[order[low - 1]]),
        "nodes_at_first_covering_gap": low,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(args.run), indent=2) + "\n")
