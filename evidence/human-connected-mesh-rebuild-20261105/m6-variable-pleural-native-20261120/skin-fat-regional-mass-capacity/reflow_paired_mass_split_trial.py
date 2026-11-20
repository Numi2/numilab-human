"""Reassign invalid current-frame mass routes with exact regional and typed support."""
import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np

from screen_paired_mass_routes import HERE, sha

def csr(rows, count):
    order = np.argsort(rows, kind="stable")
    ends = np.r_[0, np.cumsum(np.bincount(rows, minlength=count))]
    return order, ends

def incident(index, order, offsets):
    return order[offsets[index]:offsets[index + 1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall-limit", type=float, default=90.)
    parser.add_argument("--supplement", type=int, default=0)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--density-capacity", action="store_true")
    args = parser.parse_args()
    if args.density_capacity and not args.supplement:
        parser.error("--density-capacity requires --supplement for groups missing old routes")
    start = time.monotonic()
    deadline = start + args.wall_limit
    audit = json.loads((HERE / "full-paired-route-audit.json").read_text())
    assert set(audit["role_reports"]) == {"Skin", "Fat"}
    mass_path = HERE.parent / "resting-clearance-and-atrium-014/fine-current-paired-semantic-cycle-mass.npz"
    assert sha(mass_path) == audit["source_mass_sha256"]
    with np.load(mass_path, allow_pickle=False) as pack:
        mass = {k: pack[k] for k in pack.files}
    p = HERE.parent / "resting-clearance-and-atrium-013/fine-local-current-p1.json"
    labels = json.loads(p.read_text())["parent_inputs"]["source330_labels"]
    label_path = Path(labels["path"])
    assert sha(label_path) == labels["sha256"]
    source_table = np.loadtxt(label_path, delimiter=",", skiprows=1, dtype=np.int64)
    assert np.array_equal(source_table[:, 0], np.arange(len(source_table)))
    with label_path.open() as stream:
        header = stream.readline().strip().split(",")
    region = source_table[:, header.index("anatomical_region_id")][mass["source_parent"]]
    source_region_ids = None
    group_masks = []
    cell_domain = np.full(len(mass["group_for_cell"]), -1, dtype=np.int32)
    group_count = 0
    for receipt in audit["role_reports"].values():
        path = HERE / receipt["region_bounds_file"]
        assert sha(path) == receipt["region_bounds_sha256"]
        with np.load(path, allow_pickle=False) as support:
            rows = support["child_rows"]
            masks = support["group_and_regions"]
            group = support["child_mask_index"]
            region_ids = support["region_ids"]
        assert np.all(cell_domain[rows] == -1)
        assert np.array_equal(masks[group, 0], mass["group_for_cell"][rows])
        if source_region_ids is not None:
            assert np.array_equal(source_region_ids, region_ids)
        source_region_ids = region_ids
        cell_domain[rows] = group + group_count
        group_masks.append(masks)
        group_count += len(masks)
    assert np.all(cell_domain >= 0)
    masks = np.concatenate(group_masks, axis=0)
    assert len(masks) == group_count
    child_mass = mass["element_mass_kg"]
    old_count = int(mass["group_for_cell"].max()) + 1
    old_mass = np.bincount(mass["group_for_cell"], weights=child_mass, minlength=old_count)
    old_edge_order, old_edge_offsets = csr(mass["group"], old_count)
    region_digits = np.searchsorted(source_region_ids, region)
    assert np.array_equal(source_region_ids[region_digits], region)
    old_carriers = mass["carrier_row"]
    old_flow = mass["mass_flow_kg"]
    # Coarse recipient groups may share a regional mask yet have disjoint
    # exact source tuples. Split only groups carrying an inadmissible route.
    order, offsets = csr(cell_domain, group_count)
    split = []
    for dest, mask in enumerate(masks):
        old_group = int(mask[0])
        routes = incident(old_group, old_edge_order, old_edge_offsets)
        digit = region_digits[old_carriers[routes]]
        admissible = ((mask[1 + digit // 64] >> (digit % 64).astype(np.uint64)) & np.uint64(1)) != 0
        rows = incident(dest, order, offsets)
        if len(rows) > 1 and np.any((~admissible) & (old_flow[routes] > 0)):
            child_groups = np.arange(group_count, group_count + len(rows) - 1)
            cell_domain[rows[1:]] = child_groups
            split.extend([mask] * len(child_groups))
            group_count += len(child_groups)
    if split:
        masks = np.concatenate((masks, np.asarray(split)), axis=0)
    group_mass = np.bincount(cell_domain, weights=child_mass, minlength=group_count)
    source_count = len(region)
    rebuilt_carriers, rebuilt_groups, original_edges, rebuilt_flow = [], [], [], []
    supplier_surplus = np.zeros(source_count)
    group_shortage = np.zeros(group_count)
    removed_edges = 0
    groups_without_arcs = []
    for dest, mask in enumerate(masks):
        old_group = int(mask[0])
        candidate = incident(old_group, old_edge_order, old_edge_offsets)
        donor = old_carriers[candidate]
        digit = region_digits[donor]
        legitimate = ((mask[1 + digit // 64] >> (digit % 64).astype(np.uint64)) & np.uint64(1)) != 0
        fraction = group_mass[dest] / old_mass[old_group]
        initial = old_flow[candidate] * fraction
        removed = initial[~legitimate]
        np.add.at(supplier_surplus, donor[~legitimate], removed)
        group_shortage[dest] = removed.sum()
        removed_edges += int(np.count_nonzero(removed > 0))
        allowed = candidate[legitimate]
        if not len(allowed):
            groups_without_arcs.append(dest)
        rebuilt_carriers.extend(old_carriers[allowed].tolist())
        rebuilt_groups.extend([dest] * len(allowed))
        original_edges.extend(allowed.tolist())
        rebuilt_flow.extend(initial[legitimate].tolist())
    supplementation = None
    added = set()
    if args.supplement:
        from paired_route_projection import PairedRouteProjection
        supplementation = PairedRouteProjection(mass, audit, cell_domain)
        # Candidates connect both sides of the cut. Validate exact typed
        # projection and carrier distance lazily before carrying positive mass.
        targets = np.flatnonzero(group_shortage > 1e-24)
        suppliers = np.flatnonzero(supplier_surplus > 1e-24)
        group_material = np.zeros(group_count, dtype=np.int32)
        for role, receipt in audit["role_reports"].items():
            with np.load(HERE / receipt["region_bounds_file"], allow_pickle=False) as data:
                group_material[cell_domain[data["child_rows"]]] = 15 if role == "Skin" else 16
        distances_from = supplementation.supplier_positions
        distances_to = supplementation.group_centers
        source_material = supplementation.tuples[:, 0]
        for dest in targets:
            mask = masks[dest]
            supported = (((mask[1 + region_digits // 64] >> (region_digits % 64).astype(np.uint64)) &
                         np.uint64(1)) != 0) & (source_material == group_material[dest])
            candidates = np.flatnonzero(supported)
            distances = np.linalg.norm(distances_from[candidates] - distances_to[dest], axis=1)
            nearest = np.argsort(distances, kind="stable")
            served_types = {}
            supported_types = set().union(*(entry.keys() for entry in supplementation.direct_support(dest)))
            for candidate_offset in nearest:
                if distances[candidate_offset] > supplementation.limit:
                    break
                carrier = int(candidates[candidate_offset])
                kind = tuple(int(item) for item in supplementation.tuples[carrier])
                if kind in supported_types and served_types.get(kind, 0) < 6:
                    served_types[kind] = served_types.get(kind, 0) + 1
                    added.add((carrier, int(dest)))
        for carrier in suppliers:
            digit = region_digits[carrier]
            supported = (((masks[:, 1 + digit // 64] >> np.uint64(digit % 64)) & np.uint64(1)) != 0) & (
                source_material[carrier] == group_material)
            candidates = np.flatnonzero(supported)
            distances = np.linalg.norm(distances_to[candidates] - distances_from[carrier], axis=1)
            nearest = np.argsort(distances, kind="stable")
            for offset in nearest[:args.supplement]:
                if distances[offset] <= supplementation.limit:
                    added.add((int(carrier), int(candidates[offset])))
        added.difference_update(zip(rebuilt_carriers, rebuilt_groups))
        for carrier, dest in sorted(added):
            rebuilt_carriers.append(carrier)
            rebuilt_groups.append(dest)
            original_edges.append(-1)
            rebuilt_flow.append(0.)
    donor = np.asarray(rebuilt_carriers, dtype=np.int32)
    receiver = np.asarray(rebuilt_groups, dtype=np.int32)
    edge_source = np.asarray(original_edges, dtype=np.int32)
    flow = np.asarray(rebuilt_flow)
    source_order, source_ends = csr(donor, source_count)
    receiver_order, receiver_ends = csr(receiver, group_count)
    positive_shortage = supplier_surplus > 0
    shortage_start_kg = float(supplier_surplus.sum())
    unique_suppliers = int(np.count_nonzero(positive_shortage))
    unique_groups = int(np.count_nonzero(group_shortage > 0))
    print("initialized", len(flow), "edges", unique_suppliers, "surplus carriers and",
          unique_groups, "short recipient groups", flush=True)

    # A path exchanges equal mass at every interior carrier and group.
    # Source and recipient deficits come directly from omitted positive flow.
    carrier_seen = np.zeros(source_count, dtype=np.int32)
    group_seen = np.zeros(group_count, dtype=np.int32)
    carrier_previous = np.full(source_count, -1, dtype=np.int32)
    group_previous = np.full(group_count, -1, dtype=np.int32)
    epoch = 0
    moves = 0
    blocked = []
    verified = {}
    invalid = set()
    supplier_order = np.flatnonzero(positive_shortage)
    supplier_order = supplier_order[np.argsort(-supplier_surplus[supplier_order], kind="stable")]
    def certify_forward(edge):
        prior = edge_source[edge]
        if supplementation is None or (prior >= 0 and old_flow[prior] > 0) or edge in verified:
            return True
        proof = supplementation.verify(donor[edge], receiver[edge])
        if proof is None:
            return False
        verified[edge] = proof
        return True

    density_reflow = None
    if args.density_capacity:
        from paired_mass_density_reserve import close_with_density_reserve, density_bounds, distribute_mass
        min_cell, max_cell = density_bounds(supplementation.points, supplementation.cells, supplementation.material)
        group_lower = np.bincount(cell_domain, weights=min_cell)
        group_upper = np.bincount(cell_domain, weights=max_cell)
        supplier_surplus, mass_after, density_reflow = close_with_density_reserve(
            flow, donor, receiver, source_order, source_ends, receiver_order, receiver_ends,
            supplier_surplus, group_mass, group_lower, group_upper, certify_forward, deadline)
        group_shortage = np.maximum(0., group_lower - mass_after)
        cover = (mass_after >= group_lower * (1. - 1e-10)) & (mass_after <= group_upper * (1. + 1e-10))
        density_reflow["groups_outside_density_bounds"] = int(np.count_nonzero(~cover))
        if cover.all():
            updated_mass = distribute_mass(child_mass, cell_domain, mass_after, min_cell, max_cell)
            assert np.all(updated_mass >= min_cell * (1. - 1e-10))
            assert np.all(updated_mass <= max_cell * (1. + 1e-10))
            group_mass = np.bincount(cell_domain, weights=updated_mass)
            density_reflow["cells_changed"] = int(np.count_nonzero(child_mass != updated_mass))
            density_reflow["maximum_element_mass_change_kg"] = float(np.max(np.abs(child_mass - updated_mass)))
            density_reflow["total_absolute_mass_change_kg"] = float(np.abs(child_mass - updated_mass).sum())
            child_mass = updated_mass
    else:
        for supplier in supplier_order:
            while supplier_surplus[supplier] > 1e-24 and time.monotonic() - start < args.wall_limit:
                epoch += 1
                queue = deque([int(supplier)])
                carrier_seen[supplier] = epoch
                target = -1
                while queue and target < 0:
                    from_carrier = queue.popleft()
                    for edge in incident(from_carrier, source_order, source_ends):
                        to_group = int(receiver[edge])
                        if group_seen[to_group] == epoch or edge in invalid:
                            continue
                        group_seen[to_group] = epoch
                        group_previous[to_group] = edge
                        if group_shortage[to_group] > 1e-24:
                            target = to_group
                            break
                        for back in incident(to_group, receiver_order, receiver_ends):
                            neighbor = int(donor[back])
                            if flow[back] > 1e-24 and carrier_seen[neighbor] != epoch:
                                carrier_seen[neighbor] = epoch
                                carrier_previous[neighbor] = back
                                queue.append(neighbor)
                if target < 0:
                    blocked.append({"carrier": int(supplier), "region": int(region[supplier]),
                                    "surplus_kg": float(supplier_surplus[supplier]),
                                    "reachable_groups": int(np.count_nonzero(group_seen == epoch))})
                    break
                forward, reverse = [], []
                dest = target
                while True:
                    edge = int(group_previous[dest])
                    forward.append(edge)
                    c = int(donor[edge])
                    if c == supplier:
                        break
                    back = int(carrier_previous[c])
                    reverse.append(back)
                    dest = int(receiver[back])
                if supplementation is not None:
                    failed = False
                    for edge in forward:
                        if not certify_forward(edge):
                            invalid.add(edge)
                            failed = True
                            break
                    if failed:
                        continue
                moved = min(float(supplier_surplus[supplier]), float(group_shortage[target]),
                             *(float(flow[edge]) for edge in reverse))
                assert moved > 0
                flow[forward] += moved
                if reverse:
                    flow[reverse] -= moved
                supplier_surplus[supplier] -= moved
                group_shortage[target] -= moved
                moves += 1
    assert np.all(flow >= 0)
    expected = np.bincount(old_carriers, weights=old_flow, minlength=source_count)
    after = np.bincount(donor, weights=flow, minlength=source_count)
    assigned = np.bincount(receiver, weights=flow, minlength=group_count)
    relative = np.abs(after - expected) / expected
    relative_group = np.abs(assigned - group_mass) / group_mass
    passed = bool(relative.max() <= 1e-10 and relative_group.max() <= 1e-10 and
                  (density_reflow is None or not density_reflow["groups_outside_density_bounds"]))
    output = {
        "initial_forbidden_positive_group_arcs": removed_edges,
        "initial_missing_mass_kg": shortage_start_kg,
        "vertices_without_old_graph_support": groups_without_arcs[:50],
        "vertices_without_old_graph_support_count": len(groups_without_arcs),
        "initial_missing_carriers": unique_suppliers,
        "initial_missing_recipient_groups": unique_groups,
        "remaining_total_carrier_surplus_kg": float(supplier_surplus.sum()),
        "remaining_total_group_shortage_kg": float(group_shortage.sum()),
        "maximum_relative_carrier_mass_error": float(relative.max()),
        "maximum_relative_group_mass_error": float(relative_group.max()),
        "augmentations": moves,
        "density_capacity": density_reflow,
        "split_damaged_domain_count": len(split),
        "visited_blocked_carriers": blocked[:50],
        "visited_blocked_count": len(blocked),
        "previously_zero_edges_used": int(np.count_nonzero((flow > 0) & (edge_source >= 0) & (old_flow[edge_source] == 0))),
        "new_projection_edges_proposed": len(added),
        "new_projection_edges_used": int(np.count_nonzero((flow > 0) & (edge_source < 0))),
        "projection_edges_rejected": len(invalid),
        "new_validated_routes": [proof for edge, proof in sorted(verified.items()) if flow[edge] > 0],
        "graph_feasible": passed,
        "scope": "existing graph and current one-hop region closure, exact projection/distance and native admission separate",
        "elapsed_seconds": time.monotonic() - start,
    }
    if supplementation is not None:
        output["projection_owner_sha256"] = supplementation.projection_owner_sha256
        output["maximum_typed_projection_distance_m"] = supplementation.projection_limit
    if passed:
        output_path = HERE / ("paired-density-closed-flow-candidate.npz" if density_reflow is not None else
                              "paired-region-closed-flow-candidate.npz")
        nodal_mass = (np.bincount(supplementation.cells.ravel(), weights=np.repeat(child_mass*.25, 4),
                                  minlength=len(supplementation.points))
                      if density_reflow is not None else mass["nodal_mass_kg"])
        with output_path.open("xb") as stream:
            np.savez_compressed(stream, carrier_row=donor, group=receiver, group_for_cell=cell_domain,
                source_parent=mass["source_parent"], mass_flow_kg=flow, element_mass_kg=child_mass,
                nodal_mass_kg=nodal_mass, prior_edge_row=edge_source)
        output["candidate_sha256"] = sha(output_path)
    pending = np.flatnonzero(group_shortage > 1e-18)
    output["unfilled_recipient_domains"] = [
        {"group": int(group), "old_group": int(masks[group, 0]), "short_kg": float(group_shortage[group])}
        for group in pending
    ]
    receipt = args.receipt or Path("supplemented-paired-route-reflow.json" if args.supplement else "paired-route-reflow.json")
    (HERE / receipt).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k not in ("visited_blocked_carriers", "new_validated_routes")}, indent=2))


if __name__ == "__main__":
    main()
