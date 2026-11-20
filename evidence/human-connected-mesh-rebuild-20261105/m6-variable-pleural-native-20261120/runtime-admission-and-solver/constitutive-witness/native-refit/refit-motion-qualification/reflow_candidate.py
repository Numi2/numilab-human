"""Conserve the captured carrier ledger on the refitted tissue reference."""
from hashlib import file_digest
import json
from pathlib import Path
import subprocess

import numpy as np

out = Path(__file__).resolve().parent
root = out.parent
mesh = root / "constitutive-witness-repaired-tetgen"
refit = json.loads((mesh / "receipt.json").read_text())
base_volume = root / "absolute-volume-repaired-tetgen/native-reference-volumes.bin"
base_flow = root / "paired-floor-live-parent-density-closed-flow.npz"
base_receipt = json.loads((root / "paired-floor-live-parent-density-closure.json").read_text())


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


assert digest(base_flow) == base_receipt["candidate_sha256"]
assert digest(base_volume) == base_receipt["native_volume_sha256"]
assert digest(mesh / "mesh.ele") == refit["mesh_ele_sha256"]
assert digest(mesh / "mesh.node") == refit["mesh_node_sha256"]
volume_path = out / "constitutive-witness-repaired.volumes"
volumes = np.fromfile(volume_path, dtype=np.float64)
old_volumes = np.fromfile(base_volume, dtype=np.float64)
cells = np.loadtxt(mesh / "mesh.ele", skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=np.int32)
affected = np.flatnonzero(volumes != old_volumes)
assert len(volumes) == len(cells) == refit["cells"]
assert set(affected) <= set(refit["patches"][0]["affected_cells"])
assert np.all(np.isfinite(volumes) & (volumes > 1e-18))

with np.load(base_flow, allow_pickle=False) as data:
    allocation = dict(data)
group = allocation["group_for_cell"]
old_flow = allocation["mass_flow_kg"]
old_mass = allocation["element_mass_kg"]
carrier = allocation["carrier_row"]
receiver = allocation["group"]
archived = allocation["archived_compensation"]
count = len(allocation["source_parent"])
group_count = int(group.max()) + 1
quota = np.bincount(carrier, weights=old_flow, minlength=count)
initial = np.bincount(receiver, weights=old_flow, minlength=group_count)
skin = cells[:, 4] == 15
lower_density, upper_density = np.where(skin, 900., 850.), np.where(skin, 1100., 950.)
reserve = base_receipt["reserve_kg_m3"]
minimum = (lower_density + reserve) * volumes
maximum = (upper_density - reserve) * volumes
lower = np.bincount(group, weights=minimum)
upper = np.bincount(group, weights=maximum)
positive = np.flatnonzero(old_flow > 0)
input_path = out / "refit-density-problem.bin"
output_path = out / "refit-density-solution.bin"
cut_path = out / "refit-density-cut.bin"
with input_path.open("wb") as stream:
    for array, dtype in (
        ([count, group_count, len(positive)], np.uint32),
        (carrier[positive], np.uint32), (receiver[positive], np.uint32),
        (old_flow[positive], np.float64), (quota, np.float64),
        (lower, np.float64), (upper, np.float64), (archived[positive], np.uint8),
    ):
        np.asarray(array, dtype=dtype).tofile(stream)
subprocess.run([root / "close_live_parent_density_compiled", input_path, output_path, cut_path],
               check=True)
flow = old_flow.copy()
flow[positive] = np.fromfile(output_path, dtype=np.float64)
assert np.all(np.isfinite(flow) & (flow >= 0))
assert np.all(flow[archived] <= old_flow[archived])
final_quota = np.bincount(carrier, weights=flow, minlength=count)
carrier_error = float(np.abs(quota - final_quota).max())
assert carrier_error <= 1e-14
assigned = np.bincount(receiver, weights=flow, minlength=group_count)
group_error = np.maximum(lower - assigned, assigned - upper)
failed = np.flatnonzero(group_error > 1e-15)
if len(failed):
    print("Exhausted existing positive routes", [(int(i), float(group_error[i])) for i in failed[:16]], flush=True)
    raise SystemExit(1)

mass = old_mass.copy()
changed_groups = np.union1d(group[affected], np.flatnonzero(initial != assigned))
rebalance = np.isin(group, changed_groups)
mass[rebalance] = np.clip(mass[rebalance], minimum[rebalance], maximum[rebalance])
difference = assigned - np.bincount(group, weights=mass)
capacity = np.where(difference[group] >= 0., maximum, minimum) - mass
capacity[~rebalance] = 0.
available = np.bincount(group, weights=capacity)
fraction = np.divide(difference, available, out=np.zeros_like(difference), where=available != 0)
mass += capacity * np.clip(fraction[group], 0., 1.)
actual = np.bincount(group, weights=mass)
assert np.all(np.abs(assigned - actual) <= np.maximum(1e-20, assigned * 2e-12))
density = mass * (assigned / actual)[group] / volumes
assert np.all((density > lower_density) & (density < upper_density))
allocation["mass_flow_kg"] = flow
allocation["element_mass_kg"] = mass
allocation["nodal_mass_kg"] = np.bincount(
    cells[:, :4].ravel() - 1, weights=np.repeat(.25 * mass, 4),
    minlength=len(allocation["nodal_mass_kg"]))
new_flow = out / "refit-density-closed-flow.npz"
np.savez_compressed(new_flow, **allocation)
report = dict(
    scope="candidate static density closure using already positive carrier edges",
    source_sha256=digest(base_flow), native_volume_sha256=digest(volume_path),
    candidate_sha256=digest(new_flow), affected_volume_rows=len(affected),
    moved_groups=len(changed_groups), changed_carrier_edges=int(np.count_nonzero(flow != old_flow)),
    previously_positive_edges=int(len(positive)), new_carrier_edges=0,
    relocated_mass_kg=float(np.abs(flow - old_flow).sum() / 2),
    reserve_kg_m3=reserve, minimum_density_margin_kg_m3=float(
        np.minimum(density - lower_density, upper_density - density).min()),
    max_carrier_mass_error_kg=carrier_error, total_mass_kg=float(mass.sum()),
    witness_cell_density_kg_m3=float(density[refit["patches"][0]["target_cell_rows"][0]]),
    native_admitted=False,
)
(out / "refit-density-closure.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
