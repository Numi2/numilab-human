"""Query the existing exact P1 owner in the paired source frame, including distant corrections."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "resting-clearance-and-atrium-011"
sys.path.insert(0, str(PREVIOUS))
from candidate import COMPILER, load
from paired_reference_carriers import paired_reference_carriers


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_route_index(mass, parent):
    positive = np.flatnonzero(mass["mass_flow_kg"] > 0)
    group = mass["group"][positive]
    order = np.argsort(group, kind="stable")
    positive = positive[order]
    child_group = mass["group_for_cell"]
    group_mass = np.bincount(child_group, weights=mass["element_mass_kg"])
    offsets = np.r_[0, np.cumsum(np.bincount(group, minlength=len(group_mass)))]
    source_row = mass["source_parent"][mass["carrier_row"][positive]]
    return (offsets, group_mass, parent["materials"][source_row], parent["regions"][source_row],
            mass["mass_flow_kg"][positive], child_group, mass["element_mass_kg"])


def verify_sample_routes(selected, role, material_id, parent, transport,
                         owner, groups, p1):
    adjacency, _ = owner.build_region_adjacency(parent, role)
    one_hop = owner.expand_onehop(p1, groups, adjacency, role, len(selected))
    allowed = one_hop["cell_admissible_region_ids"]
    bounds = one_hop["cell_admissible_region_offsets"]
    offsets, group_mass, material, region, flow, cell_group, cell_mass = transport
    rejected_cells = 0
    rejected_mass = 0.0
    rejected_routes = 0
    checked_routes = 0
    witnesses = []
    for local, child in enumerate(selected):
        group = cell_group[child]
        route = slice(offsets[group], offsets[group + 1])
        regions = allowed[bounds[local]:bounds[local + 1]]
        assert np.all(material[route] == material_id)
        rejected = ~np.isin(region[route], regions)
        checked_routes += len(rejected)
        if np.any(rejected):
            count = int(np.count_nonzero(rejected))
            mass = float(flow[route][rejected].sum() * cell_mass[child] / group_mass[group])
            rejected_routes += count
            rejected_cells += 1
            rejected_mass += mass
            if len(witnesses) < 16:
                witnesses.append({
                    "child_tet_row": int(child),
                    "invalid_source_regions": np.unique(region[route][rejected]).tolist(),
                    "admissible_source_regions": regions.tolist(),
                    "inadmissible_sample_mass_kg": mass,
                })
    return {
        "tested_positive_routes": checked_routes,
        "outside_one_hop_cell_count": rejected_cells,
        "outside_one_hop_route_count": rejected_routes,
        "outside_one_hop_mass_kg": rejected_mass,
        "witnesses": witnesses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, required=True)
    args = parser.parse_args()
    assert args.sample_size >= 256
    start = time.monotonic()
    expected = json.loads((HERE / "repaired-tetgen/geometry-receipt.json").read_text())
    prior = json.loads((HERE.parent / "resting-clearance-and-atrium-013/fine-local-current-p1.json").read_text())
    parent_paths = {name: Path(pin["path"]) for name, pin in prior["parent_inputs"].items()}
    for name, path in parent_paths.items():
        assert sha(path) == prior["parent_inputs"][name]["sha256"], name
    entry = Path(json.loads((COMPILER / "p1-Skin.command.json").read_text())["command"][1])
    assert sha(entry) == prior["owner_sha256"]
    spec = importlib.util.spec_from_file_location("pinned_role_p1", entry)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    assert sha(api.OWNER230) == api.OWNER230_SHA

    # The prior budget's carrier positions change in the paired reference.
    # Querying the historical node cache would use a different source frame.
    owner = api._load_230(api.OWNER230)
    parent = owner.native.read_parent(parent_paths["source330_mass"])
    rest, tets, _, frame = paired_reference_carriers(COMPILER, parent_paths["source330_nodes"],
                                                    parent_paths["source330_tets"])
    assert np.array_equal(parent["tets"], tets)
    recorded = np.loadtxt(parent_paths["source330_nodes"], delimiter=",", skiprows=1)
    rebased = recorded[:, 2:5] + (rest - recorded[:, 5:8])
    changed_parent = np.any(rebased != parent["points"], axis=1)
    parent["points"] = rebased
    index = owner.native.build_index(parent)
    material_index = owner.build_material_indices(parent)

    old, points, cells, material, origin, _, _, _, _ = load()
    with np.load(PREVIOUS / "fine-mass-local-position-delta.npz", allow_pickle=False) as patch:
        points[patch["changed_rows"]] = patch["positions_f32"]
    assert hashlib.sha256(points.astype("<f4", copy=False).tobytes()).hexdigest() == expected["float32_array_sha256"]
    assert hashlib.sha256(cells.astype("<i4", copy=False).tobytes()).hexdigest() == expected["tetrahedron_array_sha256"]
    change_m = np.linalg.norm(points.astype(np.float64) - old, axis=1)
    report = {
        "source_frame": frame,
        "source_points_moved": int(np.count_nonzero(changed_parent)),
        "source_tetrahedra_affected": int(np.count_nonzero(np.any(changed_parent[tets], axis=1))),
        "source_parent_hashes": prior["parent_inputs"],
        "registration_owner_sha256": prior["owner_sha256"],
        "role_samples": {},
        "native_admission": False,
    }
    map_path = HERE.parent / "resting-clearance-and-atrium-014/fine-current-paired-semantic-cycle-mass.npz"
    pinned_mass = json.loads((map_path.parent / "fine-semantic-cycle-mass-verification.json").read_text())
    report["mass_flow_sha256"] = sha(map_path)
    assert report["mass_flow_sha256"] == pinned_mass["mass_sha256"]
    with np.load(map_path, allow_pickle=False) as mass:
        transport = build_route_index(mass, parent)
    for role, material_id in api.ROLE_MATERIAL.items():
        active = np.flatnonzero(material == material_id)
        changed = active[(origin[active] < 0) | np.any(change_m[cells[active]] > 0, axis=1)]
        ranking = np.argsort(-np.max(change_m[cells[changed]], axis=1), kind="stable")
        selected = np.unique(np.r_[changed[ranking[:args.sample_size // 8]],
            active[np.linspace(0, len(active) - 1, args.sample_size * 7 // 8, dtype=int)],
            active[origin[active] < 0]])
        nodes, local = np.unique(cells[selected], return_inverse=True)
        groups, lookup = api._source_groups(parent, role)
        begin = time.monotonic()
        support = api.build_node_support(points[nodes], role, parent, index, material_index,
                                         groups, lookup, owner)
        summed = api.integrate_p1_cells(local.reshape(-1, 4), support["maps"], groups, parent, role)
        numerator = summed["cell_group_p1_score_numerators"]
        denominator = summed["cell_p1_score_denominator"]
        partition = np.add.reduceat(numerator, summed["cell_group_offsets"][:-1])
        assert np.array_equal(partition, denominator)
        result = {
            "checked_cells": len(selected),
            "repaired_topology_cells_checked": int(np.count_nonzero(origin[selected] < 0)),
            "role_cells": len(active),
            "role_nodes": len(np.unique(cells[active])),
            "queried_nodes": len(nodes),
            "maximum_sample_transport_distance_m": support["query_distance_max_m"],
            "tied_cell_labels": len(summed["unresolved_child_tet_rows"]),
            "elapsed_seconds": time.monotonic() - begin,
        }
        result["mass_routes"] = verify_sample_routes(selected, role, material_id, parent,
                                                      transport, api, groups, summed)
        report["role_samples"][role] = result
        print(role, json.dumps(result), flush=True)
    report["elapsed_seconds"] = time.monotonic() - start
    (HERE / "paired-mass-route-screen.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
