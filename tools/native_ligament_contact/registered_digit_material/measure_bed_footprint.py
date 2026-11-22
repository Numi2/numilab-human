"""Measure projected skin area inside the installed plane's active band on captured triangles."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from material_context import sha


def area_fraction(z, width):
    """Area fraction with affine triangle height strictly below width."""
    if z.ndim != 2 or z.shape[1] != 3 or not np.isfinite(z).all():
        raise ValueError("Need three finite heights per face")
    a, b, c = np.sort(z, axis=1).T
    result = np.zeros(len(a))
    result[(c <= width) & (a < width)] = 1
    one = (a < width) & (b >= width) & (c > width)
    result[one] = (width - a[one]) ** 2 / ((b[one] - a[one]) * (c[one] - a[one]))
    two = (b < width) & (c > width)
    result[two] = 1 - (c[two] - width) ** 2 / ((c[two] - a[two]) * (c[two] - b[two]))
    return result


def measure(case, bed_path):
    bed = json.loads(bed_path.read_text())
    layers = json.loads((case / "accepted-layers.json").read_text())["inputs"]
    env = json.loads((case / "launch.json").read_text())["environment"]
    if bed["source_launch"] != sha(case / "launch.json") or bed["receipt"] != sha(
        case / "terminal.receipt.json"
    ):
        raise ValueError("Different captured reaction")
    source = {}
    for key in ("SURFACE_O_FACES", "COMBINED_NODE_MAP"):
        k = key + "_CSV"
        p = Path(env["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_" + k])
        if sha(p) != layers[k]:
            raise ValueError("Changed source " + k)
        source[key] = p
    with source["COMBINED_NODE_MAP"].open() as stream:
        pairs = sorted(
            (int(r["registered_surface_vertex_row"]), int(r["joint_node_id"]) - 1)
            for r in csv.DictReader(stream)
            if r["registered_surface_layer"] == "O"
        )
    rows, node = np.array(pairs).T
    if not np.array_equal(rows, np.arange(len(rows))) or len(np.unique(node)) != len(
        node
    ):
        raise ValueError("Outer material rows are not one to one")
    faces = np.loadtxt(
        source["SURFACE_O_FACES"],
        skiprows=1,
        delimiter=",",
        usecols=(1, 2, 3),
        dtype=int,
    )
    if np.min(faces) < 0 or np.max(faces) >= len(rows):
        raise ValueError("Invalid outer boundary")
    node_faces = node[faces]
    result = {
        "bed_receipt": bed["receipt"],
        "width_m": bed["width_m"],
        "source_faces": sha(source["SURFACE_O_FACES"]),
    }
    for phase, rows_key in (
        ("reference", "initial_rows"),
        ("accepted", "endpoint_rows"),
    ):
        p = case / f"physical-{phase}.node"
        if sha(p) != layers[f"NATIVE_{phase.upper()}_NODE_SHA256"]:
            raise ValueError("Different material coordinates")
        x = np.loadtxt(p, skiprows=1, usecols=(1, 2, 3))
        tri = x[node_faces]
        if (
            np.min(x[node, 2]) <= 0
            or np.sum(x[node, 2] < bed["width_m"]) != bed[rows_key]
        ):
            raise ValueError("Projected contact differs from native node accounting")
        a, b = (tri[:, 1] - tri[:, 0]).T, (tri[:, 2] - tri[:, 0]).T
        projected = 0.5 * np.abs(a[0] * b[1] - a[1] * b[0])
        clipped = projected * area_fraction(tri[:, :, 2], bed["width_m"])
        witness = max(bed["largest_force_changes"], key=lambda r: r["endpoint_N"])
        incident = np.any(node_faces == witness["outer_material_node"], axis=1)
        face_id = int(np.argmax(np.where(incident, clipped, -1)))
        ring_edges = np.linalg.norm(
            np.roll(tri[incident], 1, axis=1) - tri[incident], axis=2
        )
        result[phase] = {
            "witness_incident_faces": int(np.sum(incident)),
            "witness_one_ring_max_edge_mm": float(1000 * ring_edges.max()),
            "largest_incident_clipped_face_zero_based": face_id,
            "largest_incident_clipped_face_nodes_zero_based": node_faces[
                face_id
            ].tolist(),
            "largest_incident_clipped_face_area_cm2": float(1e4 * clipped[face_id]),
            "largest_incident_clipped_face_heights_um": (
                1e6 * tri[face_id, :, 2]
            ).tolist(),
            "phase_sha256": sha(p),
            "active_nodes": bed[rows_key],
            "clipped_faces": int(np.sum(clipped > 0)),
            "projected_triangle_area_in_band_cm2": float(np.sum(clipped) * 1e4),
            "projected_incident_area_in_band_cm2": float(clipped[incident].sum() * 1e4),
            "witness_node_zero_based": witness["outer_material_node"],
            "witness_force_N": witness[
                "initial_N" if phase == "reference" else "endpoint_N"
            ],
        }
    result["scope"] = (
        "Sum of projected triangle areas including projected overlaps. The active band is numerical. This area is neither a measured mattress footprint nor a pressure measurement."
    )
    return result


if __name__ == "__main__":
    print(
        json.dumps(measure(*(Path(v) for v in sys.argv[1:])), indent=2, allow_nan=False)
    )
