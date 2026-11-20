"""Independently check the fine carrier allocation and lumped nodal masses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from candidate import COMPILER, load
from paired_reference_carriers import paired_reference_carriers

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--paired-reference', action='store_true')
centering = parser.add_mutually_exclusive_group()
centering.add_argument('--material-center', action='store_true')
centering.add_argument('--material-center-fit', action='store_true')
options = parser.parse_args()

_, xyz, cells, material, _, _, _, _, _ = load()
with np.load('fine-mass-local-position-delta.npz', allow_pickle=False) as patch:
    xyz[patch['changed_rows']] = patch['positions_f32']
mass_name = 'fine-current-paired-carrier-quota-mass' if options.paired_reference else 'fine-exact-carrier-quota-mass'
verification_name = 'fine-current-paired-mass-verification' if options.paired_reference else 'fine-mapped-nodal-mass-verification'
suffix = '-center' if options.material_center else '-center-fit' if options.material_center_fit else ''
p = Path(f'{mass_name}{suffix}.npz')
target = Path(f'{verification_name}{suffix}.json')
with np.load(p, allow_pickle=False) as matrix:
    source_parent = matrix['source_parent']
    flow = matrix['mass_flow_kg']
    carrier = matrix['carrier_row']
    group = matrix['group']
    cell_groups = matrix['group_for_cell']
    element_mass = matrix['element_mass_kg']
    nodal_mass = matrix['nodal_mass_kg']
receipt = json.loads((COMPILER/'mass-admission/mass-receipt.json').read_text())['input_sha256']

def source(name):
    entry = receipt[name]
    path = Path(entry['path'])
    with path.open('rb') as packet:
        assert hashlib.file_digest(packet, 'sha256').hexdigest() == entry['sha256']
    return path

if options.paired_reference:
    source_node, source_tetrahedra, target_mass, _ = paired_reference_carriers(
        COMPILER, source('source330_nodes'), source('source330_tets'))
else:
    source_node = np.loadtxt(source('source330_nodes'), skiprows=1, delimiter=',', usecols=(5,6,7))
    source_tetrahedra = np.loadtxt(source('source330_tets'), skiprows=1, delimiter=',', dtype=int, usecols=(3,4,5,6))
    with np.load('fine-authenticated-transport-graph.npz', allow_pickle=False) as graph:
        target_mass = graph['available_kg']

source_points = source_node[source_tetrahedra]
canonical_ids, canonical_weights, canonical_xyz = [], [], []
for role in ('skin', 'fat'):
    with np.load(source(f'carrier_{role}_packet'), allow_pickle=False) as packet:
        parents = np.repeat(packet['global_source330_tet_rows'],
                            np.diff(packet['parent_offsets']).astype(int))
        canonical_ids.extend(parents)
        canonical_weights.extend(packet['carrier_weights'])
        canonical_xyz.extend(np.einsum('ij,ijk->ik', packet['carrier_barycentric'], source_points[parents]))
fallback = np.setdiff1d(np.arange(len(target_mass)), canonical_ids)
canonical_ids = np.r_[canonical_ids, fallback]
canonical_weights = np.r_[canonical_weights, np.ones(len(fallback))]
canonical_xyz = np.concatenate((np.asarray(canonical_xyz), source_points[fallback].mean(axis=1)))
assert np.array_equal(canonical_ids, source_parent)
expected_carrier = target_mass[canonical_ids]*canonical_weights
actual_carrier = np.bincount(carrier, weights=flow, minlength=len(expected_carrier))
assert np.all(np.abs(actual_carrier-expected_carrier) <= np.maximum(1e-13,1e-10*expected_carrier))

assert len(cells) == len(element_mass) == len(cell_groups)
assert len(xyz) == len(nodal_mass)
assert np.isfinite(flow).all() and np.all(flow>=0.)
assert np.isfinite(element_mass).all() and np.all(element_mass>0.)
assert np.isfinite(nodal_mass).all() and np.all(nodal_mass>0.)
observed_parent = np.bincount(source_parent[carrier],weights=flow,minlength=len(target_mass))
tolerance=np.maximum(1e-13,1e-10*target_mass)
assert np.all(np.abs(observed_parent-target_mass)<=tolerance)
observed_groups=np.bincount(group,weights=flow)
mapped_groups=np.bincount(cell_groups,weights=element_mass)
assert np.all(np.abs(observed_groups-mapped_groups)<=1e-13)
# The allocation is proportional within each incidence group. Every positive
# route therefore reaches every cell in its group and must pass the bound.
child_xyz = xyz[cells].mean(axis=1, dtype=float)
cell_order = np.argsort(cell_groups, kind='stable')
cell_offset = np.r_[0, np.cumsum(np.bincount(cell_groups))]
positive = np.flatnonzero(flow > 0.)
ordered_routes = positive[np.argsort(group[positive], kind='stable')]
route_offset = np.r_[0, np.cumsum(np.bincount(group[positive], minlength=len(cell_offset)-1))]
maximum_distance = 0.
for g in range(len(cell_offset)-1):
    routes = ordered_routes[route_offset[g]:route_offset[g+1]]
    source_xyz = canonical_xyz[carrier[routes]]
    rows = cell_order[cell_offset[g]:cell_offset[g+1]]
    for start in range(0,len(rows),8192):
        destination = child_xyz[rows[start:start+8192]]
        distance_squared = np.sum((destination[:,None,:] - source_xyz)**2, axis=2)
        maximum_distance = max(maximum_distance, float(np.sqrt(distance_squared.max())))
limit = json.loads((COMPILER/'mass-admission/mass-receipt.json').read_text())['transport_distance_limit_m']
assert maximum_distance <= limit + 1e-12
source_com = expected_carrier @ canonical_xyz/expected_carrier.sum()
child_com = element_mass @ child_xyz / element_mass.sum()
source_material = np.loadtxt(source('source330_tets'), skiprows=1, delimiter=',', dtype=int, usecols=(1,))[canonical_ids]
center_by_material = {}
for role in (15,16):
    source_role = source_material == role
    child_role = material == role
    a = expected_carrier[source_role] @ canonical_xyz[source_role] / expected_carrier[source_role].sum()
    b = element_mass[child_role] @ child_xyz[child_role] / element_mass[child_role].sum()
    center_by_material[role] = float(np.linalg.norm(b-a))
if options.material_center:
    assert max(center_by_material.values()) < 1.e-9

expected_nodal=np.bincount(cells.ravel(),weights=np.repeat(element_mass*.25,4),minlength=len(xyz))
assert np.array_equal(expected_nodal,nodal_mass)
volume=np.empty(len(cells))
missing=np.zeros(2,dtype=np.int64)
minimum=np.full(2,np.inf)
for start in range(0,len(cells),32768):
    s=slice(start,start+32768)
    v=xyz[cells[s]].astype(float)
    ed=v[:,1:]-v[:,:1]
    det=np.einsum('ij,ij->i',ed[:,0],np.cross(ed[:,1],ed[:,2]))
    volume[s]=det/6.
    edge_i=np.asarray([0,0,0,1,1,2])
    edge_j=np.asarray([1,2,3,2,3,3])
    e=v[:,edge_i]-v[:,edge_j]
    scale=np.einsum('nij,nij->ni',e,e).max(axis=1)**1.5
    for role,k in ((15,0),(16,1)):
        q=det[material[s]==role]/scale[material[s]==role]
        if len(q):
            minimum[k]=min(minimum[k],float(q.min()))
            missing[k]+=np.count_nonzero(q<1e-5)
assert not np.any(missing)
densities=element_mass/volume
material_summary={}
for role,lo,hi in ((15,900.,1100.),(16,850.,950.)):
    select=material==role
    density=densities[select]
    assert np.all(density>=lo*(1.-1e-10)) and np.all(density<=hi*(1.+1e-10))
    material_summary[role]={'cell_count':int(select.sum()),'mass_kg':float(element_mass[select].sum()),
        'minimum_density_kg_m3':float(density.min()),'maximum_density_kg_m3':float(density.max())}
with p.open('rb') as f:
    mass_sha=hashlib.file_digest(f,'sha256').hexdigest()
cross=Path('fine-mass-local-boundary-audit.log')
counts=[int(line.split()[-1]) for line in cross.read_text().splitlines() if line.startswith('pair ')]
assert len(counts)==20 and sum(counts)==0
report={
    'verification':'offline numerical geometry and mass',
    'native_installed':False,
    'source_budget':'current_paired_reference' if options.paired_reference else 'archived_source330_reference',
    'source_carrier_max_mass_error_kg':float(np.abs(actual_carrier-expected_carrier).max()),
    'positions_sha256':hashlib.sha256(xyz.astype('<f4',copy=False).tobytes()).hexdigest(),
    'connectivity_sha256':hashlib.sha256(cells.astype('<i4',copy=False).tobytes()).hexdigest(),
    'mass_sha256':mass_sha,
    'boundary_audit_sha256':hashlib.sha256(cross.read_bytes()).hexdigest(),
    'source_parent_max_mass_error_kg':float(np.abs(observed_parent-target_mass).max()),
    'static_skin_fat_layer_crossings':int(sum(counts)),
    'inverted_or_low_quality_cells':int(missing.sum()),
    'minimum_normalized_determinant_by_material':dict(zip((15,16),minimum.tolist())),
    'positive_route_maximum_distance_m':maximum_distance,
    'transport_distance_limit_m':limit,
    'source_to_child_center_of_mass_offset_m':float(np.linalg.norm(child_com-source_com)),
    'material_center_of_mass_offset_m':center_by_material,
    'nodal_mass_kg':float(nodal_mass.sum()),
    'element_mass_kg':float(element_mass.sum()),'material':material_summary,
}
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
