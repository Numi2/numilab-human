"""Close current-frame carrier quotas within the admitted per-cell density bounds."""
from collections import deque
import time

import numpy as np


def close_with_density_reserve(
    flow, donor, receiver, carrier_order, carrier_offsets, group_order, group_offsets,
    initial_surplus, nominal_group_mass, minimum_group_mass, maximum_group_mass,
    validate, deadline,
):
    remaining = initial_surplus.copy()
    assigned = np.bincount(receiver, weights=flow, minlength=len(nominal_group_mass))
    seen_carriers = np.zeros(len(remaining), dtype=np.int32)
    seen_groups = np.zeros(len(assigned), dtype=np.int32)
    carrier_path = np.empty(len(remaining), dtype=np.int32)
    group_path = np.empty(len(assigned), dtype=np.int32)
    epoch = 0
    eps = 1e-20
    moves = {"minimum_density": 0, "source_quota": 0}
    rejected = {"minimum_density": [], "source_quota": []}
    invalid = set()

    def carrier_edges(node):
        return carrier_order[carrier_offsets[node]:carrier_offsets[node + 1]]

    def group_edges(node):
        return group_order[group_offsets[node]:group_offsets[node + 1]]

    def eligible(edges):
        for edge in edges:
            if edge not in invalid and not validate(edge):
                invalid.add(edge)
                return False
        return True

    def assign(positive, negative, amount):
        flow[positive] += amount
        flow[negative] -= amount

    def fill_minimum(target):
        nonlocal epoch
        while minimum_group_mass[target] - assigned[target] > eps and time.monotonic() < deadline:
            epoch += 1
            queue = deque([target])
            seen_groups[target] = epoch
            source, reserve = -1, -1
            while queue and source < 0 and reserve < 0 and time.monotonic() < deadline:
                group = queue.popleft()
                if assigned[group] - minimum_group_mass[group] > eps:
                    reserve = group
                    break
                for edge in group_edges(group):
                    carrier = donor[edge]
                    if seen_carriers[carrier] == epoch or edge in invalid:
                        continue
                    seen_carriers[carrier] = epoch
                    carrier_path[carrier] = edge
                    if remaining[carrier] > eps:
                        source = carrier
                        break
                    for back in carrier_edges(carrier):
                        predecessor = receiver[back]
                        if flow[back] > eps and seen_groups[predecessor] != epoch:
                            seen_groups[predecessor] = epoch
                            group_path[predecessor] = back
                            queue.append(predecessor)
            if reserve < 0 and source < 0:
                return False
            negative = []
            if reserve >= 0:
                back = int(group_path[reserve])
                negative.append(back)
                cursor = donor[back]
                available = assigned[reserve] - minimum_group_mass[reserve]
            else:
                cursor = source
                available = remaining[source]
            positive = []
            while True:
                edge = int(carrier_path[cursor])
                positive.append(edge)
                group = receiver[edge]
                if group == target:
                    break
                back = int(group_path[group])
                negative.append(back)
                cursor = donor[back]
            if not eligible(positive):
                continue
            amount = min(available, minimum_group_mass[target] - assigned[target],
                         *(flow[i] for i in negative))
            assign(positive, negative, amount)
            assigned[target] += amount
            if reserve >= 0:
                assigned[reserve] -= amount
            else:
                remaining[source] -= amount
            moves["minimum_density"] += 1
        return minimum_group_mass[target] - assigned[target] <= eps

    def fill_quota(source):
        nonlocal epoch
        while remaining[source] > eps and time.monotonic() < deadline:
            epoch += 1
            queue = deque([source])
            seen_carriers[source] = epoch
            target = -1
            while queue and target < 0 and time.monotonic() < deadline:
                carrier = queue.popleft()
                for edge in carrier_edges(carrier):
                    group = receiver[edge]
                    if seen_groups[group] == epoch or edge in invalid:
                        continue
                    seen_groups[group] = epoch
                    group_path[group] = edge
                    if maximum_group_mass[group] - assigned[group] > eps:
                        target = group
                        break
                    for back in group_edges(group):
                        neighbor = donor[back]
                        if flow[back] > eps and seen_carriers[neighbor] != epoch:
                            seen_carriers[neighbor] = epoch
                            carrier_path[neighbor] = back
                            queue.append(neighbor)
            if target < 0:
                return False
            positive, negative = [], []
            group = target
            while True:
                edge = int(group_path[group])
                positive.append(edge)
                carrier = donor[edge]
                if carrier == source:
                    break
                back = int(carrier_path[carrier])
                negative.append(back)
                group = receiver[back]
            if not eligible(positive):
                continue
            amount = min(remaining[source], maximum_group_mass[target] - assigned[target],
                         *(flow[i] for i in negative))
            assign(positive, negative, amount)
            assigned[target] += amount
            remaining[source] -= amount
            moves["source_quota"] += 1
        return remaining[source] <= eps

    required = np.maximum(0., minimum_group_mass - assigned)
    targets = np.flatnonzero(required > eps)
    for target in targets[np.argsort(-required[targets], kind="stable")]:
        if not fill_minimum(target):
            rejected["minimum_density"].append(int(target))
    for source in np.argsort(-remaining, kind="stable"):
        if remaining[source] <= eps:
            break
        if not fill_quota(source):
            rejected["source_quota"].append(int(source))

    return remaining, assigned, {
        "moves": moves,
        "unreachable": rejected,
        "minimum_unfilled_mass_kg": float(np.maximum(0., minimum_group_mass - assigned).sum()),
        "maximum_excess_mass_kg": float(np.maximum(0., assigned - maximum_group_mass).sum()),
        "initial_mandatory_mass_kg": float(required.sum()),
        "initial_mandatory_group_count": len(targets),
        "rejected_typed_projection_count": len(invalid),
    }


def density_bounds(points, tets, materials):
    volume = np.empty(len(tets))
    for offset in range(0, len(tets), 32768):
        section = slice(offset, offset + 32768)
        xyz = points[tets[section]].astype(np.float64)
        dx = xyz[:, 1:] - xyz[:, :1]
        volume[section] = np.einsum("ij,ij->i", dx[:, 0], np.cross(dx[:, 1], dx[:, 2])) / 6.
    assert np.all(np.isfinite(volume) & (volume > 0))
    assert set(np.unique(materials)) == {15, 16}
    return volume * np.where(materials == 15, 900., 850.), volume * np.where(materials == 15, 1100., 950.)


def distribute_mass(original, groups, assigned, minimum, maximum):
    prior = np.bincount(groups, weights=original)
    missing = assigned - prior
    bound = np.where(missing[groups] >= 0., maximum, minimum)
    capacity = bound - original
    aggregate = np.bincount(groups, weights=capacity)
    ratio = np.divide(missing, aggregate, out=np.zeros_like(missing), where=aggregate != 0)
    return original + ratio[groups] * capacity
