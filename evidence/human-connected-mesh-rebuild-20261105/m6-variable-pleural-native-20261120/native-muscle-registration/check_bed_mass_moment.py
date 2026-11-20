"""Close the native bed-footprint mass moment with the mapped skin/fat cells."""
import json
from pathlib import Path
import re

import numpy as np

from check_bed_footprint import PLAN, REFERENCE, HERE, ROLE_O, digest, hull_distance

TAG = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
ROLES = ("skin", "fat")


def input_path(key, digest_key=None):
    env = PLAN["environment"]
    p = Path(env[key])
    expected = env[digest_key or key+"_SHA256"]
    assert digest(p) == expected, p
    return p


def load_mesh():
    p = input_path(TAG+"TETGEN_NODE")
    e = input_path(TAG+"TETGEN_ELEMENTS")
    coords = np.loadtxt(p, skiprows=1)
    elements = np.loadtxt(e, skiprows=1, dtype=np.int32)
    assert np.array_equal(coords[:, 0], np.arange(1, len(coords)+1))
    assert np.array_equal(elements[:, 0], np.arange(1, len(elements)+1))
    indices = elements[:, 1:5] - 1
    assert np.all(indices >= 0) and np.all(indices < len(coords))
    coords = coords[:, 1:4].astype("<f4").astype(np.float64)
    return coords, indices, elements[:, 5]


def moments(points, tets, materials):
    mass_by_role = {}
    occupied = np.zeros(len(tets), dtype=bool)
    env = PLAN["environment"]
    inputs = {}
    mass_total = 0.
    first_moment = np.zeros(3)
    for material, role in enumerate(ROLES, start=15):
        key = "NUMI_H706_FRACTIONAL_P1_" + role.upper()
        ancestry = input_path(key+"_NPZ", key+"_SHA256")
        packet = ancestry.parent.parent / f"{role}-role-geometry.npz"
        packet_hash = digest(packet)
        assert packet_hash == env[f"NUMI_H706_FRACTIONAL_GEOMETRY_{role.upper()}_SHA256"]
        with np.load(packet, allow_pickle=False) as data:
            rows = data["combined_child_rows"].astype(np.intp)
            nodes = data["combined_node_rows"].astype(np.intp)
            assert np.array_equal(points[nodes], data["points_world_m_f32"])
            assert np.array_equal(nodes[data["tetrahedra"]], tets[rows])
        assert np.array_equal(rows, np.flatnonzero(materials == material))
        flow_key = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_" + role.upper()
        flow = input_path(flow_key+"_TRANSPORT_FLOW_CSV").parent
        receipt_path = flow / f"{role}-recorded-flow-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["complete_allocation"]
        flow_name = f"{role}-transport-flow.csv"
        assert receipt["flow_csv"][flow_name]["sha256"] == env[flow_key+"_TRANSPORT_FLOW_CSV_SHA256"]
        file = flow / f"{role}-child-mass.csv"
        expected = receipt["flow_csv"][file.name]["sha256"]
        assert digest(file) == expected
        masses = np.loadtxt(file, delimiter=",", skiprows=1, usecols=(0, 2))
        assert np.array_equal(masses[:, 0], np.arange(len(masses)))
        assert len(masses) == len(rows) and np.all(masses[:, 1] > 0.)
        assert np.all(rows >= 0) and np.all(rows < len(tets))
        assert len(np.unique(rows)) == len(rows) and not np.any(occupied[rows])
        occupied[rows] = True
        centres = points[tets[rows]].mean(axis=1)
        role_mass = float(masses[:, 1].sum())
        moment = masses[:, 1] @ centres
        mass_by_role[role] = {"cell_count": len(rows), "mapped_mass_kg": role_mass,
                              "mapped_density_centre_m": (moment / role_mass).tolist()}
        mass_total += role_mass
        first_moment += moment
        inputs[role] = {"child_mass_sha256": expected, "p1_sha256": digest(ancestry),
                        "geometry_sha256": packet_hash,
                        "mapped_mass_receipt_sha256": digest(receipt_path)}
    assert occupied.all()
    assert abs(mass_total-13.939678963460478) < 1.e-9
    return mass_total, first_moment, mass_by_role, inputs


def main():
    footprint = json.loads((HERE / "bed-footprint.json").read_text())
    coords, tets, materials = load_mesh()
    skin_fat_mass, skin_fat_moment, roles, inputs = moments(coords, tets, materials)
    rigid_mass = footprint["articulated_mass_kg"]
    fem_match = re.search(r"^resting_tissue_shared_runtime=bound.* FEM_mass_kg=(\S+)",
                          (REFERENCE / "stdout.log").read_text(), re.M)
    assert fem_match
    full_fem_mass = float(fem_match[1])
    remainder = full_fem_mass - skin_fat_mass
    assert 0. < remainder < 1.
    total = rigid_mass + full_fem_mass
    projected_moment = skin_fat_moment[:2] + rigid_mass * np.array(footprint["initial_articulated_com_xy_m"])
    known_center = projected_moment / (rigid_mass + skin_fat_mass)
    force = total * 9.81
    plane = np.array(footprint["nearest_position_m"][:2])
    necessary_unmodelled_xy = (total*plane - projected_moment) / remainder

    skin_path = input_path(ROLE_O, TAG+"SURFACE_O_NODES_SHA256")
    skin = np.loadtxt(skin_path, delimiter=",", skiprows=1, usecols=(2, 3, 4)).astype("<f4")
    assert len(skin) == footprint["outer_node_count"]
    near = skin[skin[:, 2] <= 1.e-5, :2]
    assert near.shape == (1, 2) and np.array_equal(near[0], plane)
    r = {
        "mapped_layer_inputs": inputs,
        "mapped_layers": roles,
        "native_total_body_mass_kg": total,
        "total_body_weight_at_9_81_N_kg_N": force,
        "unresolved_other_FEM_moment_mass_kg": remainder,
        "moment_accounted_mass_fraction": (total - remainder) / total,
        "centre_of_accounted_mass_xy_m": known_center.tolist(),
        "closest_registered_skin_point_xy_m": plane.tolist(),
        "closest_point_to_accounted_centre_m": hull_distance(near.astype(float), known_center)[0],
        "required_remainder_centre_xy_for_point_support_m": necessary_unmodelled_xy.tolist(),
        "required_remainder_centre_distance_from_contact_m": float(np.linalg.norm(necessary_unmodelled_xy-plane)),
        "test_scope": "A static balance using the only skin point within 10 micrometers. The mass moment uses the mapped skin and fat cell masses and native Float32 vertices plus the articulated COM. It is an initialization feasibility check."
    }
    print(json.dumps(r, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
