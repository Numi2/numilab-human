"""Audit every positive mass route against the exact paired-frame one-hop domain."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from screen_paired_mass_routes import (
    HERE, PREVIOUS, COMPILER, load, sha, paired_reference_carriers, build_route_index,
)

WORKER = {}


def initialize_worker(entry, parent, region_ids, material_id, role):
    spec = importlib.util.spec_from_file_location("pinned_role_p1", entry)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    owner = api._load_230(api.OWNER230)
    groups, lookup = api._source_groups(parent, role)
    adjacency, _ = api.build_region_adjacency(parent, role)
    region_masks = {}
    for region, neighbors in adjacency.items():
        bits = np.zeros((len(region_ids) + 63) // 64, dtype=np.uint64)
        for neighbor in neighbors:
            digit = int(np.searchsorted(region_ids, neighbor))
            assert region_ids[digit] == neighbor
            bits[digit // 64] |= np.uint64(1) << np.uint64(digit % 64)
        region_masks[region] = bits
    assert api.ROLE_MATERIAL[role] == material_id
    WORKER.update(api=api, owner=owner, parent=parent, role=role, groups=groups,
                  lookup=lookup, region_masks=region_masks, index=owner.native.build_index(parent),
                  material_index=owner.build_material_indices(parent))


def query_batch(points):
    context = WORKER
    api, groups = context["api"], context["groups"]
    support = api.build_node_support(points, context["role"], context["parent"], context["index"],
                                     context["material_index"], groups, context["lookup"], context["owner"])
    region_masks = context["region_masks"]
    bits = np.zeros((len(points), len(next(iter(region_masks.values())))), dtype=np.uint64)
    for index, mapping in enumerate(support["maps"]):
        for group in mapping:
            bits[index] |= region_masks[int(groups[group, 2])]
    return bits, float(np.max(support["distance"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error("workers must be within 1 through 4")
    t0 = time.monotonic()
    base = HERE.parent / "resting-clearance-and-atrium-013"
    prior = json.loads((base / "fine-local-current-p1.json").read_text())
    verification = json.loads((HERE / "repaired-tetgen/geometry-receipt.json").read_text())
    paths = {name: Path(item["path"]) for name, item in prior["parent_inputs"].items()}
    for name, path in paths.items():
        assert sha(path) == prior["parent_inputs"][name]["sha256"]
    entry = Path(json.loads((COMPILER / "p1-Skin.command.json").read_text())["command"][1])
    assert sha(entry) == prior["owner_sha256"]
    spec = importlib.util.spec_from_file_location("pinned_role_p1", entry)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    assert sha(api.OWNER230) == api.OWNER230_SHA
    owner = api._load_230(api.OWNER230)
    parent = owner.native.read_parent(paths["source330_mass"])
    rest, source_tets, _, frame = paired_reference_carriers(COMPILER, paths["source330_nodes"],
                                                           paths["source330_tets"])
    assert np.array_equal(parent["tets"], source_tets)
    recorded = np.loadtxt(paths["source330_nodes"], delimiter=",", skiprows=1)
    parent["points"] = recorded[:, 2:5] + (rest - recorded[:, 5:8])

    _, points, cells, material, _, _, _, _, _ = load()
    with np.load(PREVIOUS / "fine-mass-local-position-delta.npz", allow_pickle=False) as patch:
        points[patch["changed_rows"]] = patch["positions_f32"]
    from export_repaired_tetgen import array_hash
    assert array_hash(points, "<f4") == verification["float32_array_sha256"]
    assert array_hash(cells, "<i4") == verification["tetrahedron_array_sha256"]
    mass_file = HERE.parent / "resting-clearance-and-atrium-014/fine-current-paired-semantic-cycle-mass.npz"
    prior_mass = json.loads((mass_file.parent / "fine-semantic-cycle-mass-verification.json").read_text())
    assert sha(mass_file) == prior_mass["mass_sha256"]
    with np.load(mass_file, allow_pickle=False) as data:
        transport = build_route_index(data, parent)
    offset, old_mass, donor_material, donor_region, carrier_flow, group_for_cell, cell_mass = transport
    region_ids = np.unique(parent["regions"])
    route_digit = np.searchsorted(region_ids, donor_region)
    route_word = route_digit // 64
    route_bit = np.uint64(1) << (route_digit % 64).astype(np.uint64)
    result = {
        "source_frame": frame,
        "source_mass_sha256": prior_mass["mass_sha256"],
        "geometry_sha256": verification["float32_array_sha256"],
        "tetrahedron_sha256": verification["tetrahedron_array_sha256"],
        "role_reports": {},
    }

    for role, material_id in api.ROLE_MATERIAL.items():
        rows = np.flatnonzero(material == material_id)
        node_rows, indices = np.unique(cells[rows], return_inverse=True)
        node_points = points[node_rows]
        shape = (len(node_rows), (len(region_ids) + 63) // 64)
        node_masks = np.empty(shape, dtype=np.uint64)
        batch_size = 4096
        batches = [node_points[start:start+batch_size] for start in range(0, len(node_rows), batch_size)]
        worker_arguments = (entry, parent, region_ids, material_id, role)
        max_distance = 0.0
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker, initargs=worker_arguments) as pool:
            for batch_index, (bits, distance) in enumerate(pool.map(query_batch, batches)):
                start = batch_index * batch_size
                node_masks[start:start+len(bits)] = bits
                max_distance = max(max_distance, distance)
                if (batch_index + 1) % 8 == 0 or batch_index + 1 == len(batches):
                    print(role, start + len(bits), "/", len(node_rows), "nodes", flush=True)

        cell_masks = np.bitwise_or.reduce(node_masks[indices.reshape(-1, 4)], axis=1)
        # Confirm the compact union against the exact original P1/one-hop path.
        sample = np.unique(np.r_[np.linspace(0, len(rows)-1, 256, dtype=int),
            np.flatnonzero(np.isin(rows, [495255, 551159, 579504, 623990]))])
        original = indices.reshape(-1, 4)[sample]
        sample_nodes, sample_tets = np.unique(original, return_inverse=True)
        groups, lookup = api._source_groups(parent, role)
        true_support = api.build_node_support(node_points[sample_nodes], role, parent,
                                              owner.native.build_index(parent),
                                              owner.build_material_indices(parent), groups, lookup, owner)
        p1 = api.integrate_p1_cells(sample_tets.reshape(-1, 4), true_support["maps"], groups, parent, role)
        adjacency, _ = api.build_region_adjacency(parent, role)
        exact = api.expand_onehop(p1, groups, adjacency, role, len(sample))
        for local, row in enumerate(sample):
            digits = np.arange(len(region_ids), dtype=np.uint64)
            actual = region_ids[((cell_masks[row, digits//64] >> (digits%64)) & np.uint64(1)) != 0]
            a, b = exact["cell_admissible_region_offsets"][local:local+2]
            assert np.array_equal(actual, exact["cell_admissible_region_ids"][a:b])

        # Split only for audit accounting. Preserve all original transport flows.
        domains = np.column_stack((group_for_cell[rows].astype(np.uint64), cell_masks))
        groups, inverse = np.unique(domains, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        counts = np.bincount(inverse)
        receiving_mass = np.bincount(inverse, weights=cell_mass[rows])
        bad = np.zeros(len(groups), dtype=bool)
        bad_pair_count = route_count = 0
        bad_mass = 0.0
        bad_edges = set()
        for index, (group, *mask) in enumerate(groups):
            group = int(group)
            route = slice(offset[group], offset[group+1])
            assert np.all(donor_material[route] == material_id)
            missing = (np.asarray(mask, dtype=np.uint64)[route_word[route]] & route_bit[route]) == 0
            route_count += int(counts[index]) * len(missing)
            if np.any(missing):
                bad[index] = True
                bad_pair_count += int(counts[index]) * int(np.count_nonzero(missing))
                bad_mass += float(carrier_flow[route][missing].sum() * receiving_mass[index] / old_mass[group])
                bad_edges.update((group, int(region)) for region in np.unique(donor_region[route][missing]))
        have_failure = bad[inverse]
        output = HERE / f"paired-route-region-bounds-{role.lower()}.npz"
        with output.open("xb") as stream:
            np.savez_compressed(stream, child_rows=rows.astype(np.uint32),
                child_mask_index=inverse.astype(np.uint32), group_and_regions=groups, region_ids=region_ids)
        result["role_reports"][role] = {
            "checked_cells": len(rows),
            "exact_original_p1_control_cells": len(sample),
            "distinct_transport_group_and_region_domains": len(groups),
            "positive_expanded_routes": route_count,
            "cells_outside_one_hop": int(np.count_nonzero(have_failure)),
            "positive_routes_outside_one_hop": bad_pair_count,
            "mass_outside_one_hop_kg": bad_mass,
            "inadmissible_group_region_pairs": [list(pair) for pair in sorted(bad_edges)],
            "maximum_node_query_distance_m": max_distance,
            "region_bounds_file": output.name,
            "region_bounds_sha256": sha(output),
        }
        (HERE / "full-paired-route-audit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(role, json.dumps(result["role_reports"][role]), flush=True)
    result["elapsed_seconds"] = time.monotonic() - t0
    (HERE / "full-paired-route-audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
