"""Rebuild full Skin/Fat P1 ancestry for the independently verified repaired volume."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
previous = root.parent / "resting-clearance-and-atrium-016"
sys.path.insert(0, str(previous))
from audit_full_paired_mass_routes import initialize_worker, WORKER
from paired_route_projection import PairedRouteProjection
from screen_paired_mass_routes import COMPILER, sha
from export_repaired_tetgen import array_hash


def query(points):
    owner = WORKER
    api = owner["api"]
    support = api.build_node_support(points, owner["role"], owner["parent"], owner["index"],
                                    owner["material_index"], owner["groups"], owner["lookup"], owner["owner"])
    return support["maps"], support["distance"]


def enforce_region_parity(data, rows, pinned, role, group_masks, mask_ids, region_ids):
    assert np.array_equal(rows, pinned["child_rows"])
    expected_mask = group_masks[mask_ids, 1:]
    starts = data["cell_admissible_region_offsets"]
    ids = data["cell_admissible_region_ids"]
    digits = np.searchsorted(region_ids, ids)
    assert np.array_equal(region_ids[digits], ids)
    actual = np.zeros_like(expected_mask)
    owners = np.repeat(np.arange(len(rows)), np.diff(starts))
    np.bitwise_or.at(actual, (owners, digits//64), np.uint64(1) << (digits%64).astype(np.uint64))
    difference = np.count_nonzero(np.any(actual != expected_mask, axis=1))
    if difference:
        raise ValueError(f"{role}: {difference} full P1 region masks differ from the independent route audit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    start = time.monotonic()
    audit = json.loads((previous/"full-paired-route-audit.json").read_text())
    receipt = json.loads((root/"archived-compensation-independent-check.json").read_text())
    assert receipt["typed_route_failure_count"] == 0
    current_mass = root/"paired-density-closed-flow-candidate.npz"
    assert sha(current_mass) == receipt["mass_sha256"]
    plan = json.loads((root.parent/"resting-clearance-and-atrium-013/fine-local-current-p1.json").read_text())
    entry = Path(json.loads((COMPILER/"p1-Skin.command.json").read_text())["command"][1])
    p1_hash = plan["owner_sha256"]
    assert sha(entry) == p1_hash
    with np.load(current_mass) as packet:
        mass = {name: packet[name] for name in ("source_parent", "group_for_cell")}
    p = PairedRouteProjection(mass, audit, mass["group_for_cell"])
    api, parent = p.p1, p.parent
    region_ids = np.unique(parent["regions"])
    output = root/"complete-fine-p1"
    output.mkdir()
    state_path = output/"current-paired-volume-state.json"
    state = {
        "current_reference": audit["source_frame"],
        "geometry_receipt": {"path": str(previous/"repaired-tetgen/geometry-receipt.json"),
                            "sha256": sha(previous/"repaired-tetgen/geometry-receipt.json")},
        "geometry_sha256": audit["geometry_sha256"],
        "tetrahedra_sha256": audit["tetrahedron_sha256"],
        "mass_sha256": receipt["mass_sha256"],
        "source330_identity": plan["parent_inputs"],
        "native_admitted": False,
    }
    state_path.write_text(json.dumps(state, indent=2)+"\n")
    state_sha = sha(state_path)
    for role, material in api.ROLE_MATERIAL.items():
        rows = np.flatnonzero(p.material == material)
        nodes, local = np.unique(p.cells[rows], return_inverse=True)
        local = local.reshape(-1,4)
        points = p.points[nodes]
        assert not len(points)-len(np.unique(api._point_key(points)))
        groups, _ = api._source_groups(parent,role)
        compare_path = previous/audit["role_reports"][role]["region_bounds_file"]
        assert sha(compare_path) == audit["role_reports"][role]["region_bounds_sha256"]
        context = multiprocessing.get_context("spawn")
        node_maps, distances = [], []
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=context,initializer=initialize_worker,
                                 initargs=(entry,parent,region_ids,material,role)) as pool:
            batches = (points[i:i+4096] for i in range(0, len(points),4096))
            for ix, (m,d) in enumerate(pool.map(query,batches)):
                node_maps.extend(m)
                distances.append(d)
                if ix%16 == 15 or len(node_maps)==len(nodes):
                    print(role,"same-frame nodes",len(node_maps),"/",len(nodes),flush=True)
        assert len(node_maps) == len(nodes)
        distance = np.concatenate(distances)
        info = {"maps":node_maps, "distance":distance, "origin":np.full(len(nodes),3,dtype=np.uint8)}
        node_arrays = api._serialize_node_maps(points,info,groups,parent)
        print(role,"integrating",len(rows),"cells",flush=True)
        cell_arrays = api.integrate_p1_cells(local, node_maps, groups, parent, role)
        adjacency, _ = api.build_region_adjacency(parent,role)
        neighbors = api.expand_onehop(cell_arrays,groups,adjacency,role,len(rows))
        with np.load(compare_path,allow_pickle=False) as masks:
            enforce_region_parity(neighbors,rows,masks,role,masks["group_and_regions"],
                                  masks["child_mask_index"],masks["region_ids"])
        geom = output/f"{role.lower()}-role-geometry.npz"
        np.savez_compressed(geom, points_world_m_f32=points, tetrahedra=local.astype("<u4"),
                            combined_node_rows=nodes, combined_child_rows=rows)
        volume_sha = sha(geom)
        base = {**node_arrays, **cell_arrays,
            "current_state_sha256":np.asarray(state_sha),
            "current_state_path":np.asarray(str(state_path)),
            "geometry_sha256":np.asarray(volume_sha), "volume_sha256":np.asarray(volume_sha),
            "source330_mass_sha256":np.asarray(plan["parent_inputs"]["source330_mass"]["sha256"])}
        primary, full, primary_sha = api._write_support_pair(base,neighbors,output/role,role)
        summary = {
            "role":role, "cells":len(rows), "nodes":len(nodes),
            "all_cell_regions_match_independent_audit":True,
            "child_indices_sha256":array_hash(p.cells[rows],"<i4"),
            "maximum_registration_distance_m":float(distance.max()),
            "resolved_unique_p1_label_cells":len(cell_arrays["resolved_child_tet_rows"]),
            "tied_p1_label_cells":len(cell_arrays["unresolved_child_tet_rows"]),
            "p1_path":str(primary), "p1_sha256":primary_sha,
            "p1_one_hop_path":str(full), "p1_one_hop_sha256":sha(full),
            "role_geometry_path":str(geom), "role_geometry_sha256":volume_sha,
            "state_path":str(state_path), "state_sha256":state_sha,
            "seconds":time.monotonic()-start, "physical_admission":False,
        }
        (output/f"{role.lower()}-p1-receipt.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
