"""Re-solve the repaired fine volume with the compiler's conserved carrier quotas."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, hstack, vstack

from candidate import COMPILER, load
from paired_reference_carriers import paired_reference_carriers

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--paired-reference', action='store_true')
parser.add_argument('--diagnose', action='store_true')
centering = parser.add_mutually_exclusive_group()
centering.add_argument('--conserve-material-center', action='store_true')
centering.add_argument('--minimize-material-center-error', action='store_true')
parser.add_argument('--extra-receiver-rows', type=Path)
options = parser.parse_args()
stem = ('fine-current-paired-carrier-quota-mass' if options.paired_reference
        else 'fine-exact-carrier-quota-mass')
if options.conserve_material_center:
    stem += '-center'
if options.minimize_material_center_error:
    stem += '-center-fit'

old, xyz, cells, material, origins, cavities, _, old_cells, old_material = load()
with np.load('fine-mass-local-position-delta.npz', allow_pickle=False) as patch:
    xyz[patch['changed_rows']] = patch['positions_f32']
with np.load('fine-authenticated-transport-graph.npz', allow_pickle=False) as graph:
    old_parent = graph['donor']
    old_group = graph['group']
    cell_group = graph['group_for_cell']
    parent_mass = graph['available_kg']
    volume = graph['volume_m3']
assert np.all(cell_group >= 0)
receipt = json.loads((COMPILER/'mass-admission/mass-receipt.json').read_text())

def pinned(name):
    record=receipt['input_sha256'][name]
    path=Path(record['path'])
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256'],name
    return path

source_nodes = pinned('source330_nodes')
source_tets = pinned('source330_tets')
reference = {'budget': 'archived_source330_reference'}
if options.paired_reference:
    node, tet, parent_mass, reference = paired_reference_carriers(COMPILER, source_nodes, source_tets)
else:
    node = np.loadtxt(source_nodes, skiprows=1, delimiter=',', usecols=(5,6,7))
    tet = np.loadtxt(source_tets, skiprows=1, delimiter=',', dtype=int, usecols=(3,4,5,6))
parent_tet=node[tet]
parent_centroid=parent_tet.mean(axis=1)
pa,po,pw=[],[],[]
for role in ('skin','fat'):
    with np.load(pinned(f'carrier_{role}_packet'), allow_pickle=False) as p:
        parents=np.repeat(p['global_source330_tet_rows'], np.diff(p['parent_offsets']).astype(int))
        weight=p['carrier_weights']
        bary=p['carrier_barycentric']
        assert np.all(np.abs(bary.sum(axis=1)-1.)<=2e-12) and np.all(weight>0.)
        pa.extend(parents)
        po.extend(np.einsum('ra,raj->rj',bary,parent_tet[parents]))
        pw.extend(weight)
weights=np.asarray(pw)
carrier_parent=np.asarray(pa,dtype=np.int32)
carrier_xyz=np.asarray(po)
weight_sum=np.bincount(carrier_parent,weights=weights,minlength=len(parent_mass))
has_carriers=weight_sum>0.
assert np.max(np.abs(weight_sum[has_carriers]-1.))<2e-12
moment_error=0.
for component in range(3):
    weighted_center=np.bincount(carrier_parent,weights=carrier_xyz[:,component]*weights,minlength=len(parent_mass))
    moment_error=max(moment_error,float(np.abs(weighted_center[has_carriers]-parent_centroid[has_carriers,component]).max()))
assert moment_error<1e-11
fallback=np.flatnonzero(~has_carriers)
carrier_xyz=np.vstack((carrier_xyz,parent_centroid[fallback]))
weights=np.r_[weights,np.ones(len(fallback))]
carrier_parent=np.r_[carrier_parent,fallback]
carrier_mass=parent_mass[carrier_parent]*weights
carrier_by_parent=[[] for _ in parent_mass]
for row,source in enumerate(carrier_parent):
    carrier_by_parent[source].append(row)
carrier_radius=np.zeros(len(parent_mass))
np.maximum.at(carrier_radius,carrier_parent,np.linalg.norm(carrier_xyz-parent_centroid[carrier_parent],axis=1))
print(json.dumps({'expanded_carriers':len(carrier_xyz),'checkpoint_source_mass_kg':float(parent_mass.sum()),
                  'maximum_source_weighted_center_error_m':moment_error}),flush=True)

limit=float(receipt['transport_distance_limit_m'])
child_centroid=xyz[cells].mean(axis=1,dtype=float)
assignments=np.full(len(cells),-1,dtype=np.int32)
new_keys={}
key_list=[]
all_rows=np.argsort(cell_group,kind='stable')
starts=np.r_[0,np.cumsum(np.bincount(cell_group))]
donor_starts=np.r_[0,np.cumsum(np.bincount(old_group))]
pruned_rows=0
def assign(rows,key):
    g=new_keys.get(key)
    if g is None:
        g=len(key_list)
        new_keys[key]=g
        key_list.append(key)
    assignments[rows]=g

for group in range(len(starts)-1):
    rows=all_rows[starts[group]:starts[group+1]]
    assert len(rows)>0
    donors=old_parent[donor_starts[group]:donor_starts[group+1]]
    points=child_centroid[rows]
    certificate=np.all(np.linalg.norm(points[:,None]-parent_centroid[donors],axis=2)+carrier_radius[donors]<=(limit-1e-12),axis=1)
    source_slots=np.asarray(sorted(slot for donor in donors for slot in carrier_by_parent[donor]),dtype=np.int32)
    typed=int(material[rows[0]])
    if np.any(certificate):
        assign(rows[certificate],(typed,tuple(source_slots.tolist())))
    for row in rows[~certificate]:
        distances=np.linalg.norm(carrier_xyz[source_slots]-child_centroid[row],axis=1)
        retained=source_slots[distances<=limit]
        pruned_rows+=int(len(retained)<len(source_slots))
        if len(retained):
            assign(row,(typed,tuple(retained.tolist())))

uncovered = np.flatnonzero(assignments < 0)
uncovered_source = {
    int(row): ([int(origins[row])] if origins[row] >= 0
               else list(map(int, cavities[int(origins[row])])))
    for row in uncovered
}
extra_count = 0
receivers = dict(uncovered_source)
if options.extra_receiver_rows is not None:
    for row in json.loads(options.extra_receiver_rows.read_text()):
        if row not in receivers:
            receivers[row] = ([int(origins[row])] if origins[row] >= 0
                              else list(map(int, cavities[int(origins[row])])))
if options.paired_reference and receivers:
    if options.extra_receiver_rows is None:
        target = Path('fine-paired-route-targets.txt')
        target.write_text(''.join(f'C {row}\n' for row in sorted(set(
            source for sources in receivers.values() for source in sources))))
    old_to_uncovered = defaultdict(list)
    for row, ranked in receivers.items():
        for source in ranked:
            old_to_uncovered[source].append(row)
    original_node = np.loadtxt(source_nodes, skiprows=1, delimiter=',', usecols=(5,6,7))
    source_cells = original_node[tet]
    labels = np.loadtxt(COMPILER/'volume-export/volume/child.labels.csv', skiprows=1,
                        delimiter=',', dtype=int, usecols=(1,2,3))
    additional = defaultdict(set)
    first = 0
    for role, tag in (('skin', 15), ('fat', 16)):
        detail = Path(f'fine-paired-{role}-carrier-witnesses.csv')
        with np.load(pinned(f'carrier_{role}_packet'), allow_pickle=False) as p:
            source_count = len(p['carrier_barycentric'])
            if detail.is_file():
                old_positions = np.einsum('ra,raj->rj', p['carrier_barycentric'],
                                           source_cells[carrier_parent[first:first+source_count]])
                with detail.open(newline='') as stream:
                    for fields in csv.reader(stream):
                        _, side, _, slot, _, ancestor, _, receiver, material_tag = fields[:9]
                        if int(receiver) not in old_to_uncovered:
                            continue
                        slot, ancestor, receiver = int(slot), int(ancestor), int(receiver)
                        assert int(material_tag) == tag == old_material[receiver] and side.lower() == role
                        assert ancestor == carrier_parent[first+slot]
                        historical_receiver = old[old_cells[receiver]].mean(axis=0, dtype=float)
                        historical_carrier = np.asarray(fields[19:22], dtype=float)
                        assert np.linalg.norm(old_positions[slot]-historical_carrier) < 1e-11
                        assert np.linalg.norm(historical_receiver-np.asarray(fields[22:25],dtype=float)) < 1e-6
                        assert abs(np.linalg.norm(historical_carrier-historical_receiver)-float(fields[25])) < 1e-6
                        for row in old_to_uncovered[receiver]:
                            if np.all(labels[receivers[row]] == labels[receiver]):
                                additional[row].add(ancestor)
        first += source_count
    for row, donors in additional.items():
        slots = np.asarray(sorted(slot for donor in donors for slot in carrier_by_parent[donor]), dtype=np.int32)
        slots = slots[np.linalg.norm(carrier_xyz[slots]-child_centroid[row], axis=1) <= limit]
        if len(slots):
            prior = np.asarray(key_list[assignments[row]][1], dtype=np.int32) if assignments[row] >= 0 else np.empty(0, dtype=np.int32)
            union = np.union1d(prior, slots)
            if len(union) > len(prior):
                assign(row, (int(material[row]), tuple(union.tolist())))
                extra_count += len(union)-len(prior)
    occupied = np.unique(assignments[assignments >= 0])
    remap = np.full(len(key_list), -1, dtype=np.int32)
    remap[occupied] = np.arange(len(occupied))
    assigned = assignments >= 0
    assignments[assigned] = remap[assignments[assigned]]
    key_list = [key_list[group] for group in occupied]

report={'scope':'offline_exact_carrier_capacity_trial', 'complete_motion_certificate':False,
        'native_mass_admission':False,'transport_distance_limit_m':limit,
        'reference':reference,'source_parent_count':len(parent_mass),'carrier_count':len(carrier_mass),
        'source_mass_kg':float(parent_mass.sum()),'source_carrier_weight_moment_max_m':moment_error,
        'additional_exact_carrier_edges':extra_count,'pruned_child_positions':pruned_rows,'groups':len(key_list),
        'children_without_admitted_carrier':int(np.count_nonzero(assignments<0)),
        'uncovered_receiver_rows':uncovered_source,
        'still_uncovered_receiver_rows':np.flatnonzero(assignments<0).tolist()}
output=Path(f'{stem}.json')
def save_report():
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if np.any(assignments<0):
    report['status']='route_infeasible'
    save_report()
    raise SystemExit(0)

ng=len(key_list)
group_role=np.asarray([m for m,_ in key_list])
group_volume=np.bincount(assignments,weights=volume,minlength=ng)
carrier_slot=np.array([c for _,key in key_list for c in key],dtype=np.int32)
group_slot=np.repeat(np.arange(ng,dtype=np.int32),[len(key) for _,key in key_list])
cols=np.arange(len(carrier_slot))
supplied=np.bincount(carrier_slot,minlength=len(carrier_mass))>0
report['carrier_without_admitted_child']=int(np.count_nonzero(~supplied & (carrier_mass>0)))
report['missing_carrier_capacity_kg']=float(carrier_mass[~supplied].sum())
report['positive_candidate_edges']=len(cols)
if np.any(~supplied & (carrier_mass>0)):
    report['status']='route_infeasible'
    save_report()
    raise SystemExit(0)
carrier_matrix=coo_matrix((np.ones(len(cols)),(carrier_slot,cols)),shape=(len(carrier_mass),len(cols))).tocsr()
group_matrix=coo_matrix((np.ones(len(cols)),(group_slot,cols)),shape=(ng,len(cols))).tocsr()
interval=vstack((group_matrix,-group_matrix),format='csr')
lower=np.where(group_role==15,900.,850.)
upper=np.where(group_role==15,1100.,950.)
scale=1e6
if options.conserve_material_center or options.minimize_material_center_error:
    # The auxiliary masses retain the sparse flow equations. Only the group
    # variables enter the center-of-mass rows.
    source_roles = np.loadtxt(source_tets, skiprows=1, delimiter=',', dtype=int, usecols=(1,))
    center = carrier_mass @ carrier_xyz / carrier_mass.sum()
    weighted = (child_centroid-center)*volume[:,None]
    group_center = np.stack([np.bincount(assignments, weights=weighted[:,i], minlength=ng)
                             for i in range(3)], axis=1) / group_volume[:,None]
    source_moments = (carrier_xyz-center)*carrier_mass[:,None]
    moment = np.vstack([np.where(group_role[None,:]==role,group_center.T,0.)
                        for role in (15,16)])
    moments = np.concatenate([source_moments[source_roles[carrier_parent]==role].sum(axis=0)
                              for role in (15,16)])
    equations = vstack((
        hstack((carrier_matrix,coo_matrix((len(carrier_mass),ng)))),
        hstack((group_matrix,-eye(ng))),
        hstack((coo_matrix((6,len(cols))),coo_matrix(moment))),
    ),format='csr')
    costs = np.zeros(len(cols)+ng)
    bounds = np.vstack((np.c_[np.zeros(len(cols)),np.full(len(cols),np.inf)],
                        np.c_[group_volume*lower,group_volume*upper]*scale))
    if options.minimize_material_center_error:
        # Only center error has slack. Source quotas and physical density
        # caps remain hard constraints.
        slack = vstack((coo_matrix((len(carrier_mass)+ng,12)),
                        hstack((eye(6),-eye(6)))),format='csr')
        equations = hstack((equations,slack),format='csr')
        costs = np.r_[costs,np.ones(12)]
        bounds = np.vstack((bounds,np.tile([0.,np.inf],(12,1))))
    solution = linprog(
        costs, A_eq=equations,
        b_eq=np.r_[carrier_mass,np.zeros(ng),moments]*scale,
        bounds=bounds,
        method='highs-ipm',options={'primal_feasibility_tolerance':1e-9,'time_limit':65.})
    report['material_center_conservation_requested'] = options.conserve_material_center
    if options.minimize_material_center_error and solution.success:
        report['minimum_sum_material_moment_error_kg_m'] = float(solution.fun/scale)
else:
    solution=linprog(np.zeros(len(cols)), A_eq=carrier_matrix,b_eq=carrier_mass*scale,
                     A_ub=interval,b_ub=np.r_[group_volume*upper,-group_volume*lower]*scale,
                     bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9})
report['optimizer_status']=solution.message
report['status']=('capacity_feasible' if solution.success else
                  'capacity_infeasible' if solution.status==2 else 'capacity_inconclusive')
if options.diagnose and not solution.success and not options.conserve_material_center:
    deficit = linprog(
        np.r_[np.zeros(len(cols)),np.ones(2*ng)],
        A_eq=hstack((carrier_matrix,coo_matrix((len(carrier_mass),2*ng))),format='csr'),
        b_eq=carrier_mass*scale,
        A_ub=hstack((interval,-eye(2*ng)),format='csr'),
        b_ub=np.r_[group_volume*upper,-group_volume*lower]*scale,
        bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9})
    if deficit.success:
        bad=np.flatnonzero(np.maximum(deficit.x[-2*ng:-ng],deficit.x[-ng:])>1e-6)
        report['minimum_density_bound_deficit_kg']=float(deficit.fun/scale)
        report['affected_groups']=len(bad)
        report['affected_cells']=int(np.isin(assignments,bad).sum())
        report['maximum_receiver_violation_kg']=float(deficit.x[-2*ng:].max()/scale)
        report['group_deficits']=[
            {'group':int(g),'over_capacity_kg':float(deficit.x[-2*ng+g]/scale),
             'under_capacity_kg':float(deficit.x[-ng+g]/scale),
             'cells':np.flatnonzero(assignments==g).tolist()} for g in bad]
        np.savez_compressed(f'{stem}-deficit.npz',
            receiver_group=assignments,source_parent=carrier_parent,
            carrier_potential=deficit.eqlin.marginals,
            group_potential=deficit.ineqlin.marginals[:ng]-deficit.ineqlin.marginals[ng:],
            deficit_groups=bad)
    else:
        report['deficit_diagnosis_status']=deficit.message

if solution.success:
    flow=solution.x[:len(cols)]/scale
    delivered=carrier_matrix@flow
    rel_error=np.abs(delivered-carrier_mass)/np.maximum(carrier_mass,1e-300)
    density=(group_matrix@flow)/group_volume
    element_mass=volume*density[assignments]
    nodal=np.bincount(cells.ravel(),weights=np.repeat(element_mass/4.,4),minlength=len(xyz))
    position_moment=np.einsum('i,ij->j',element_mass,child_centroid)/element_mass.sum()
    report.update(maximum_carrier_closure_kg=float(np.abs(delivered-carrier_mass).max()),
                  maximum_carrier_closure_relative=float(rel_error.max()),
                  child_density_min_kg_m3={str(k):float(density[group_role==k].min()) for k in (15,16)},
                  child_density_max_kg_m3={str(k):float(density[group_role==k].max()) for k in (15,16)},
                  child_mass_kg=float(element_mass.sum()),nodal_mass_kg=float(nodal.sum()),
                  child_center_of_mass_m=position_moment.tolist())
    np.savez_compressed(f'{stem}.npz',
        carrier_row=carrier_slot, group=group_slot, group_for_cell=assignments,
        source_parent=carrier_parent, mass_flow_kg=flow, element_mass_kg=element_mass,nodal_mass_kg=nodal)
save_report()
