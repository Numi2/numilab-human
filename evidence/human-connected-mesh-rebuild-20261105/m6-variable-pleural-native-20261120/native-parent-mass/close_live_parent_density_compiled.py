"""Close the paired allocation against the production reader's reference volumes."""
from pathlib import Path
from add_live_parent_routes import add_live_routes
import argparse
from collections import deque
import hashlib
import json
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
import subprocess

prior = root.parent / "resting-clearance-and-atrium-016"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference-floor", action="store_true", help="Use the recertified absolute-volume geometry")
parser.add_argument("--neighbors", type=int, default=12)
args = parser.parse_args()
mesh_path = prior / "repaired-tetgen/mesh.ele"
if args.reference_floor:
    original_path = root / "live-parent-warm-quota-flow.npz"
    volume_path = root / "absolute-volume-repaired-tetgen/native-reference-volumes.bin"
    result_path = root / "paired-floor-live-parent-density-closed-flow.npz"
    report_path = root / "paired-floor-live-parent-density-closure.json"
else:
    original_path = root / "paired-density-closed-flow-candidate.npz"
    volume_path = root / "native-mass-reference-f64.bin"
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

group_lower = np.bincount(group, weights=minimum, minlength=group_count)
group_upper = np.bincount(group, weights=maximum, minlength=group_count)
add_live_routes(root, output, group_lower, group_upper, args.neighbors)
original_flow = output["mass_flow_kg"]
positive = np.arange(len(original_flow))
flow = original_flow.copy()
donor = output["carrier_row"]
receiver = output["group"]
archived = output["archived_compensation"]
original_quota = np.bincount(donor, weights=flow, minlength=carrier_count)
initial_group = np.bincount(receiver, weights=flow, minlength=group_count)
assert np.all(group_upper > group_lower)

state = root / "live-parent-density-problem.bin"
solved = root / "live-parent-density-solution.bin"
diagnostic = root / "live-parent-density-cut.bin"
with state.open("wb") as stream:
    for array, dtype in (
        ([carrier_count, group_count, len(flow)], np.uint32),
        (donor, np.uint32), (receiver, np.uint32), (flow, np.float64),
        (original_quota, np.float64), (group_lower, np.float64), (group_upper, np.float64),
        (archived, np.uint8),
    ):
        np.asarray(array, dtype=dtype).tofile(stream)
subprocess.run([root / "close_live_parent_density_compiled", state, solved, diagnostic], check=True)
state.unlink()
flow = np.fromfile(solved, dtype=np.float64)
assert len(flow) == len(donor) and np.all(flow >= 0)
assert flow[archived].sum() <= .000325
assigned = np.bincount(receiver, weights=flow, minlength=group_count)
final_quota = np.bincount(donor, weights=flow, minlength=carrier_count)
quota_error = np.abs(final_quota - original_quota)
print("max carrier closure kg", quota_error.max(), "original compensation kg", original_flow[archived].sum(),
      "final compensation kg", flow[archived].sum(), flush=True)
unmet = np.maximum(group_lower-assigned, assigned-group_upper)
misses = np.flatnonzero(unmet > 1.e-15)
print("unmet groups", len(misses), "unmet total kg", unmet[misses].sum(),
      "worst kg", unmet.max(), flush=True)
if len(misses):
    for idx in misses[np.argsort(unmet[misses])[-12:]]:
        route_ids = np.flatnonzero(receiver == idx)
        print("unmet witness", int(idx), float(unmet[idx]), int(np.count_nonzero(group==idx)),
              int(len(route_ids)), "above upper", bool(assigned[idx]>group_upper[idx]), flush=True)
    np.savez_compressed(root / "live-parent-density-cut-witnesses.npz", bad_group=misses,
                        group_mass=assigned, route_carrier=donor, route_group=receiver, route_mass=flow)
    raise ValueError("Retained support graph cannot close the captured donor ledger")
assert quota_error.max() <= 1.e-14
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

solved.unlink()
summary = dict(
    source_sha256=digest(original_path), native_volume_sha256=digest(volume_path),
    candidate_sha256=digest(result_path), reserve_kg_m3=reserve_kg_m3,
    density_margin_kg_m3=float(np.minimum(density-lower, upper-density).min()),
    max_carrier_mass_error_kg=float(quota_error.max()), total_mass_kg=float(mass.sum()),
    compensation_mass_kg=float(flow[archived].sum()), positive_routes=int(np.count_nonzero(flow)),
)
report_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
