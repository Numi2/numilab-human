"""Close current carrier quotas using only routes already admitted by the fit."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

output = Path.cwd()
previous = output.parent / 'resting-clearance-and-atrium-011'
sys.path.insert(0, str(previous))
os.chdir(previous)

from candidate import COMPILER, load
from paired_reference_carriers import paired_reference_carriers

receipt = json.loads((COMPILER/'mass-admission/mass-receipt.json').read_text())
def pinned(key):
    row = receipt['input_sha256'][key]
    path = Path(row['path'])
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
    return path

_, _, parent_mass, _ = paired_reference_carriers(
    COMPILER, pinned('source330_nodes'), pinned('source330_tets'))
ids, weights = [], []
for material in ('skin', 'fat'):
    with np.load(pinned(f'carrier_{material}_packet'), allow_pickle=False) as packet:
        ids.extend(np.repeat(packet['global_source330_tet_rows'],
                             np.diff(packet['parent_offsets']).astype(int)))
        weights.extend(packet['carrier_weights'])
centroid_only = np.setdiff1d(np.arange(len(parent_mass)), ids)
ids = np.r_[ids, centroid_only]
weights = np.r_[weights, np.ones(len(centroid_only))]
quota = parent_mass[ids] * weights
assert (quota > 0.).all()

input_stem = 'fine-current-paired-carrier-quota-mass-center-fit'
with np.load(input_stem+'.npz', allow_pickle=False) as raw:
    data = dict(raw)
carrier = data['carrier_row']
group = data['group']
cell_group = data['group_for_cell']
assert np.array_equal(data['source_parent'], ids)
base_flow = data['mass_flow_kg']
delivered = np.bincount(carrier, weights=base_flow, minlength=len(quota))
old_error = np.abs(delivered/quota - 1.)
zero = np.flatnonzero(delivered == 0.)
flow = np.zeros_like(base_flow)
positive = delivered[carrier] > 0.
flow[positive] = base_flow[positive]*(quota[carrier[positive]]/delivered[carrier[positive]])

_, xyz, cells, material, _, _, _, _, _ = load()
with np.load('fine-mass-local-position-delta.npz', allow_pickle=False) as delta:
    xyz[delta['changed_rows']] = delta['positions_f32']
volume = np.empty(len(cells))
for start in range(0,len(cells),32768):
    portion = slice(start,start+32768)
    vertices = xyz[cells[portion]].astype(float)
    edges = vertices[:,1:] - vertices[:,:1]
    volume[portion] = np.einsum('ij,ij->i',edges[:,0],np.cross(edges[:,1],edges[:,2]))/6.
count = int(group.max()+1)
group_volume = np.bincount(cell_group, weights=volume, minlength=count)
group_material = np.full(count, -1, dtype=int)
group_material[cell_group] = material
assert np.array_equal(material, group_material[cell_group])
maximum = np.where(group_material == 15, 1100., 950.)
minimum = np.where(group_material == 15, 900., 850.)
used = np.bincount(group, weights=flow, minlength=count)
# Preserve a saturated group's mass by moving an existing carrier's flow
# to another admitted receiver. Each internal vertex on the alternating
# path has equal inflow and outflow.
carrier_order = np.argsort(carrier, kind='stable')
carrier_offsets = np.r_[0, np.cumsum(np.bincount(carrier,minlength=len(quota)))]
group_order = np.argsort(group,kind='stable')
group_offsets = np.r_[0, np.cumsum(np.bincount(group,minlength=count))]
limit = maximum * group_volume
exhausted = []
rerouted = 0
for row in zero[np.argsort(-quota[zero], kind='stable')]:
    amount = quota[row]
    queue = deque([int(row)])
    from_carrier = {int(row): None}
    from_group = {}
    sink = None
    while queue and sink is None:
        source = queue.popleft()
        lo, hi = carrier_offsets[source:source+2]
        for edge in carrier_order[lo:hi]:
            receiver = int(group[edge])
            if receiver in from_group:
                continue
            from_group[receiver] = edge
            if limit[receiver]-used[receiver] >= amount:
                sink = receiver
                break
            gl, gh = group_offsets[receiver:receiver+2]
            donors = group_order[gl:gh]
            for back in donors[flow[donors] >= amount]:
                previous_source = int(carrier[back])
                if previous_source not in from_carrier:
                    from_carrier[previous_source] = back
                    queue.append(previous_source)
    if sink is None:
        exhausted.append(int(row))
        continue
    used[sink] += amount
    receiver = sink
    while True:
        edge = from_group[receiver]
        flow[edge] += amount
        back = from_carrier[int(carrier[edge])]
        if back is None:
            break
        rerouted += 1
        flow[back] -= amount
        receiver = int(group[back])

new_delivery = np.bincount(carrier,weights=flow,minlength=len(quota))
new_error = np.abs(new_delivery/quota - 1.)
density = np.bincount(group,weights=flow,minlength=count)/group_volume
report = dict(
    carrier_count=len(quota), prior_rows_above_relative_tolerance=int(np.count_nonzero(old_error>1e-10)),
    prior_zero_delivery_rows=len(zero), prior_undelivered_zero_mass_kg=float(quota[zero].sum()),
    restored_zero_delivery_rows=len(zero)-len(exhausted), exhausted_zero_rows=exhausted,
    rerouted_existing_edges=rerouted, max_relative_error=float(new_error.max()), largest_prior_error_kg=float(abs(delivered-quota).max()),
    min_density_margin=float((density-minimum).min()), max_density_margin=float((maximum-density).min()),
    expanded_positive_routes=int(np.bincount(cell_group)[group[flow>0.]].sum()),
    native_installed=False,
)
(output/'relative-carrier-closure.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
assert not exhausted and new_error.max() <= 1e-10
assert np.all(density >= minimum*(1.-1e-10)) and np.all(density <= maximum*(1.+1e-10))
data['mass_flow_kg'] = flow
data['element_mass_kg'] = volume*density[cell_group]
data['nodal_mass_kg'] = np.bincount(cells.ravel(),weights=np.repeat(data['element_mass_kg']*.25,4),minlength=len(xyz))
np.savez_compressed(output/'fine-current-paired-relative-carrier-mass.npz', **data)
