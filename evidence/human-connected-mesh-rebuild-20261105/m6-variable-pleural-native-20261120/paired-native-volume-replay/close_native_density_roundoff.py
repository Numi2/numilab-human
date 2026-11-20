"""Close the paired allocation against the production reader's reference volumes."""
from pathlib import Path
from collections import deque
import hashlib
import json
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
prior = root.parent / "resting-clearance-and-atrium-016"
sys.path.insert(0, str(prior))
# The warm allocator uses an absolute graph threshold for gram-scale
# regional repairs. This replay only moves floating-point roundoff mass.
owner = prior / "paired_mass_density_reserve.py"
solver = owner.read_text()
for original, exact in (
    ("eps = 1e-20", "eps = 1e-28"),
    ("minimum_group_mass * 2e-12", "minimum_group_mass * 2e-14"),
):
    assert solver.count(original) == 1
    solver = solver.replace(original, exact)
scope = {}
exec(compile(solver, str(owner), "exec"), scope)
close_with_density_reserve = scope["close_with_density_reserve"]

original_path = root / "paired-density-closed-flow-candidate.npz"
volume_path = root / "native-mass-reference-f64.bin"
mesh_path = prior / "repaired-tetgen/mesh.ele"
result_path = root / "paired-native-density-closed-flow.npz"
report_path = root / "paired-native-density-closure.json"
assert not result_path.exists() and not report_path.exists()
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

with np.load(original_path) as data:
    output = dict(data)
cells = np.loadtxt(mesh_path, skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=np.int32)
volume = np.fromfile(volume_path, dtype=np.float64)
assert len(cells) == len(volume) == len(output["element_mass_kg"])
assert np.all(np.isfinite(volume) & (volume > 0))
skin = cells[:, 4] == 15
assert np.array_equal(np.unique(cells[:, 4]), [15, 16])
lower = np.where(skin, 900., 850.)
upper = np.where(skin, 1100., 950.)
# Leave a reserve for aggregate rounding against the exact native volume.
reserve_kg_m3 = 3.e-10
minimum = (lower + reserve_kg_m3) * volume
maximum = (upper - reserve_kg_m3) * volume
original_mass = output["element_mass_kg"]
group = output["group_for_cell"]
group_count = int(group.max()) + 1
carrier_count = len(output["source_parent"])

original_flow = output["mass_flow_kg"]
positive = np.flatnonzero(original_flow > 0.)
flow = original_flow[positive].copy()
donor = output["carrier_row"][positive]
receiver = output["group"][positive]
archived = output["archived_compensation"][positive]
original_quota = np.bincount(donor, weights=flow, minlength=carrier_count)
initial_group = np.bincount(receiver, weights=flow, minlength=group_count)
group_lower = np.bincount(group, weights=minimum, minlength=group_count)
group_upper = np.bincount(group, weights=maximum, minlength=group_count)
assert np.all(group_upper > group_lower)
def csr(ids, count):
    return np.argsort(ids, kind="stable"), np.r_[0, np.cumsum(np.bincount(ids, minlength=count))]

donor_order, donor_offsets = csr(donor, carrier_count)
receiver_order, receiver_offsets = csr(receiver, group_count)
# Remove any overfull amounts without changing which source carriers own the
# surplus. Only previously authenticated positive routes can redistribute it.
shrink = np.divide(np.minimum(initial_group, group_upper), initial_group,
                   out=np.ones_like(initial_group), where=initial_group > 0.)
flow *= shrink[receiver]
surplus = np.bincount(donor, weights=original_flow[positive] - flow,
                      minlength=carrier_count)
print("group deficits", np.count_nonzero(initial_group < group_lower),
      "maximum kg", max(group_lower - initial_group), "total kg", sum(np.maximum(group_lower-initial_group, 0.)),
      "above maximum", np.count_nonzero(initial_group > group_upper),
      "total excess kg", sum(np.maximum(initial_group-group_upper, 0.)), flush=True)
# Build an alternating traversal from groups with usable reserve. Route
# through the existing positive graph only. The existing allocator handles
# any residual that can't use the tree after one of its edges is depleted.
assigned = np.bincount(receiver, weights=flow, minlength=group_count)
eps = 1.e-28

def reserve_tree(slack):
    carrier_link = np.full(carrier_count, -1, dtype=np.int32)
    group_link = np.full(group_count, -1, dtype=np.int32)
    roots = np.flatnonzero(slack > 2.e-13)
    group_link[roots] = -2
    queue = deque(roots)
    while queue:
        group_id = queue.popleft()
        for edge in receiver_order[receiver_offsets[group_id]:receiver_offsets[group_id+1]]:
            carrier_id = donor[edge]
            if carrier_link[carrier_id] >= 0 or flow[edge] <= eps:
                continue
            carrier_link[carrier_id] = edge
            for other in donor_order[donor_offsets[carrier_id]:donor_offsets[carrier_id+1]]:
                g = receiver[other]
                if group_link[g] == -1:
                    group_link[g] = other
                    queue.append(g)
    return carrier_link, group_link

def route_to_reserve(slack, pending, start_from_carrier):
    carrier_link, group_link = reserve_tree(slack)
    count = 0
    for start in np.flatnonzero(pending > eps):
        carrier_id = start if start_from_carrier else (
            int(donor[group_link[start]]) if group_link[start] >= 0 else -1)
        step = carrier_link[carrier_id] if carrier_id >= 0 else -1
        if step < 0:
            continue
        positive = [] if start_from_carrier else [group_link[start]]
        negative = []
        while True:
            (positive if start_from_carrier else negative).append(step)
            reached = receiver[step]
            cross = group_link[reached]
            if cross == -2:
                break
            (negative if start_from_carrier else positive).append(cross)
            step = carrier_link[donor[cross]]
        amount = min(pending[start], slack[reached], *(flow[k] for k in negative))
        if amount <= eps:
            continue
        flow[positive] += amount
        flow[negative] -= amount
        slack[reached] -= amount
        pending[start] -= amount
        count += 1
    return count

direct = route_to_reserve(group_upper - assigned, surplus, True)
assigned = np.bincount(receiver, weights=flow, minlength=group_count)
lower_direct = route_to_reserve(assigned - group_lower, np.maximum(group_lower - assigned, 0.), False)
assigned = np.bincount(receiver, weights=flow, minlength=group_count)
print("tree releases", direct, "tree exchanges", lower_direct,
      "remaining lower deficit kg", np.maximum(group_lower-assigned, 0.).sum(),
      "remaining donor kg", surplus.sum(), flush=True)
remaining, assigned, adjustments = close_with_density_reserve(
    flow, donor, receiver, donor_order, donor_offsets, receiver_order, receiver_offsets,
    surplus, initial_group, group_lower, group_upper, lambda _: True,
    time.monotonic() + 12.)
final_quota = np.bincount(donor, weights=flow, minlength=carrier_count)
print("moves", adjustments["moves"], "minimum unmatched", sum(np.maximum(group_lower-assigned,0.)),
      "maximum unmatched", max(group_lower-assigned), "leftover carrier kg", remaining.max(),
      "failed counts", {key: len(value) for key,value in adjustments["unreachable"].items()},
      "first misses", {key: value[:6] for key,value in adjustments["unreachable"].items()},
      flush=True)
assert np.all(flow >= 0.)
assert np.isclose(flow[archived].sum(), original_flow[positive][archived].sum(), rtol=1.e-6)
assert flow[archived].sum() <= .000325
quota_error = np.abs(final_quota - original_quota)
print("largest donor mass error kg", quota_error.max(), "remaining sum", remaining.sum(),
      "quota errors over tolerance", np.count_nonzero(quota_error > np.maximum(1.e-20, original_quota * 2.e-13)),
      flush=True)
assert quota_error.max() <= 1.e-14 and remaining.sum() <= 1.e-14
mass = np.clip(original_mass, minimum, maximum)
initial_cell_group = np.bincount(group, weights=mass, minlength=group_count)
difference = assigned - initial_cell_group
capacity = np.where(difference[group] >= 0., maximum, minimum) - mass
group_capacity = np.bincount(group, weights=capacity, minlength=group_count)
fraction = np.divide(difference, group_capacity, out=np.zeros_like(difference),
                     where=group_capacity != 0.)
# Cap the projection at each cell's physical capacity. If scalar reduction
# puts a group on its bound, concentrate no roundoff onto its smallest cell.
mass += capacity * np.clip(fraction[group], 0., 1.)
actual_group = np.bincount(group, weights=mass, minlength=group_count)
predicted_density = mass * (assigned/actual_group)[group] / volume
print("predicted exact-volume density outliers",
      np.count_nonzero((predicted_density <= lower) | (predicted_density >= upper)),
      "predicted worst error kg/m3",
      max(np.maximum(lower-predicted_density, predicted_density-upper)), flush=True)
density = mass / volume
bad = (density <= lower) | (density >= upper)
print("nominal density outliers", np.count_nonzero(bad), "worst delta kg/m3",
      max(np.maximum(lower-density,density-upper)), flush=True)
for cell in np.flatnonzero(bad)[:20]:
    g = group[cell]
    edge_ids = np.flatnonzero(output["group"] == g)
    print("density witness", int(cell), int(g), float(density[cell]),
          "quota deficit", float(group_lower[g] - assigned[g]),
          "quota excess", float(assigned[g] - group_upper[g]),
          "group members", np.count_nonzero(group == g), "positive/total eligible candidate edges",
          np.count_nonzero(original_flow[edge_ids]), len(edge_ids), "edges",
          [(int(output["carrier_row"][e]), float(original_flow[e])) for e in edge_ids[:20]], flush=True)
assert not np.any(bad)
cell_group = np.bincount(group, weights=mass, minlength=group_count)
assert np.all(np.abs(assigned - cell_group) <= np.maximum(1.e-20, assigned * 2.e-12))
output["mass_flow_kg"][positive] = flow
output["element_mass_kg"] = mass
output["nodal_mass_kg"] = np.bincount(cells[:, :4].ravel() - 1, weights=np.repeat(.25 * mass, 4),
                                     minlength=len(output["nodal_mass_kg"]))
np.savez_compressed(result_path, **output)
summary = dict(
    source_sha256=digest(original_path), native_volume_sha256=digest(volume_path),
    candidate_sha256=digest(result_path), reserve_kg_m3=reserve_kg_m3,
    density_margin_kg_m3=float(np.minimum(density-lower, upper-density).min()),
    adjusted_child_count=int(np.count_nonzero(mass != original_mass)),
    maximum_cell_mass_change_kg=float(np.max(np.abs(mass - original_mass))),
    total_absolute_cell_mass_change_kg=float(np.abs(mass-original_mass).sum()),
    max_carrier_mass_change_kg=float(np.max(np.abs(final_quota - original_quota))),
    maximum_unallocated_carrier_kg=float(remaining.max()), total_mass_kg=float(mass.sum()),
    compensation_mass_kg=float(flow[archived].sum()),
    positive_routes_before=len(positive), positive_routes_after=int(np.count_nonzero(flow)),
    direct_releases=direct, direct_exchanges=lower_direct, route_adjustments=direct + lower_direct + sum(adjustments["moves"].values()), **adjustments,
)
report_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
