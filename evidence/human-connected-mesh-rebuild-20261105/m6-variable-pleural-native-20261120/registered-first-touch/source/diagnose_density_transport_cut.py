"""Check if the already certified donor routes can transport physically bounded mass."""
from collections import defaultdict
import math
import hashlib
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, hstack, vstack

from candidate import COMPILER, load

old,x,tets,mat,origins,cavities,_,old_tets,old_mat=load()
with np.load('fine-reference-measures.npz',allow_pickle=False) as data:
    v=data['volume_m3']
assert len(v)==len(mat)
with np.load('fine-mass-local-position-delta.npz', allow_pickle=False) as repair:
    x[repair['changed_rows']] = repair['positions_f32']
    v[repair['affected_rows']] = repair['volume_m3']

maximum_route_distance=.187444
source=COMPILER/'mass-allocation/allocator-output/carrier-trace.csv.zst'
source_count=28632
original_to_current=np.full(len(old_tets),-1,dtype=np.int32)
kept=np.flatnonzero(origins>=0)
original_to_current[origins[kept]]=kept
replacement_ancestors=defaultdict(list)
for i in np.flatnonzero(origins<0):
    for ancestor in cavities[int(origins[i])]:
        replacement_ancestors[ancestor].append(i)
new_centroid=x[tets].mean(axis=1,dtype=float)
old_centroid=old[old_tets].mean(axis=1,dtype=float)

lower=v*np.where(mat==15,900.,850.)
upper=v*np.where(mat==15,1100.,950.)
available=np.zeros(source_count)
support=[None]*len(tets)
new_support=defaultdict(set)
all_mass=hashlib.sha256()
manifest=json.loads((COMPILER/'mass-admission/run-manifest.json').read_text())
receipt=json.loads((COMPILER/'mass-admission/mass-receipt.json').read_text())
pins=receipt['input_sha256']
def source_file(name):
    instance=pins[name]
    path=Path(instance['path'])
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==instance['sha256'],name
    return path

node=np.loadtxt(source_file('source330_nodes'),skiprows=1,delimiter=',',usecols=(5,6,7))
reference_tet=np.loadtxt(source_file('source330_tets'),skiprows=1,delimiter=',',dtype=int,usecols=(3,4,5,6))
parents=node[reference_tet]
centroid=parents.mean(axis=1)
anchors=[]
anchor_parent=[]
for role in ('skin','fat'):
    with np.load(source_file(f'carrier_{role}_packet'),allow_pickle=False) as carrier:
        tet_ids=np.repeat(carrier['global_source330_tet_rows'],np.diff(carrier['parent_offsets']).astype(int))
        xyz=np.einsum('ra,raj->rj',carrier['carrier_barycentric'],parents[tet_ids])
        anchors.extend(map(tuple,xyz))
        anchor_parent.extend(map(int,tet_ids))
centroid=list(map(tuple,centroid))
old_center=list(map(tuple,old_centroid))
new_center=list(map(tuple,new_centroid))
n=0
discarded=0
blocked=[]
largest_defect=0.
p=subprocess.Popen(['/opt/homebrew/bin/zstd','-qcd',str(source)],stdout=subprocess.PIPE)
header=next(p.stdout)
all_mass.update(header)
assert header.startswith(b'flow_edge_row,child_row,expanded_parent_row,source_parent_row,carrier_row,')
last_child=-1
usable=[]
def finish(old):
    if old>=0 and original_to_current[old]>=0:
        support[original_to_current[old]]=tuple(sorted(set(usable)))
for line in p.stdout:
    all_mass.update(line)
    _,child,_,source_id,carrier_id,mass,_,distance=line.strip().split(b',')
    i,j,carrier=int(child),int(source_id),int(carrier_id)
    available[j]+=float(mass)
    assert carrier<0 or anchor_parent[carrier]==j
    xyz=centroid[j] if carrier<0 else anchors[carrier]
    source_error=abs(math.dist(old_center[i],xyz)-float(distance))
    largest_defect=max(largest_defect,source_error)
    if i!=last_child:
        assert i>last_child
        finish(last_child)
        usable=[]
        last_child=i
    k=original_to_current[i]
    if k>=0:
        if math.dist(new_center[k],xyz)<=maximum_route_distance:
            usable.append(j)
        else:
            discarded+=1
            blocked.append([i,int(k),j,carrier,float(mass),math.dist(new_center[k],xyz)])
    else:
        for c in replacement_ancestors[i]:
            if math.dist(new_center[c],xyz)<=maximum_route_distance:
                new_support[c].add(j)
            else:
                discarded+=1
    n+=1
finish(last_child)
assert p.wait()==0
for i,donors in new_support.items():
    support[i]=tuple(sorted(donors))
assert all_mass.hexdigest()==receipt['output_sha256']['carrier_trace']['sha256']
print(json.dumps(dict(pinned_carrier_trace_sha256=all_mass.hexdigest(),
                     max_original_distance_error_m=largest_defect, carrier_positions=len(anchors))),flush=True)
assert largest_defect<1e-6,largest_defect
assert np.isfinite(v).all() and np.all(v>0)

# Read already-authorized one-hop centroid slots only for short ranges of
# recipients implicated by the necessary capacity checks. Never mint a slot
# from nearest-neighbor distance alone; prior material and region are required.
labels=np.loadtxt(COMPILER/'volume-export/volume/child.labels.csv',skiprows=1,
                  dtype=int,delimiter=',',usecols=(1,2,3))
mandatory=np.zeros(source_count)
capacity=np.zeros(source_count)
for k,donors in enumerate(support):
    if not donors:
        continue
    if len(donors)==1:
        mandatory[donors[0]]+=lower[k]
    capacity[list(donors)]+=upper[k]
shortfall=set(np.flatnonzero(mandatory>available+1e-12))
surplus=set(np.flatnonzero(available>capacity+1e-12))
target=set(k for k,g in enumerate(support) if not g or (len(g)==1 and g[0] in shortfall))
active_parent={int(origins[k]):[k] for k in target if origins[k]>=0}
for k in target:
    if origins[k]>=0:
        continue
    old_rows=cavities[int(origins[k])]
    if np.all(labels[old_rows]==labels[old_rows[0]]):
        for row in old_rows:
            active_parent.setdefault(row,[]).append(k)

path=COMPILER/'mass-support/source-centroid-slots.csv.zst'
proof=json.loads(path.with_suffix('.zst.recovery.json').read_text())
digest=hashlib.sha256()
proc=subprocess.Popen(['/opt/homebrew/bin/zstd','-qcd',str(path)],stdout=subprocess.PIPE)
header=next(proc.stdout)
digest.update(header)
assert header.startswith(b'source_parent_row,receiver_child_tet_row,source_material,')
extended=defaultdict(set)
supra=0
for line in proc.stdout:
    digest.update(line)
    source_id,old_id,fields=line.split(b',',2)
    i=int(old_id)
    k=int(source_id)
    if i not in active_parent and k not in surplus:
        continue
    d=fields.strip().split(b',')
    assert len(d)==20
    if int(d[0])!=int(old_mat[i]) or tuple(map(int,d[16:19]))!=tuple(labels[i]):
        continue
    if available[k]<=0. or int(d[15])>1:
        continue
    eligible=active_parent.get(i,[])
    if k in surplus and original_to_current[i]>=0 and not eligible:
        eligible=[int(original_to_current[i])]
    position=np.asarray(d[7:10],dtype=float)
    for j in eligible:
        if k in (support[j] or ()):
            continue
        if np.linalg.norm(position-new_centroid[j])<=maximum_route_distance:
            extended[j].add(k)
        else:
            supra+=1
assert proc.wait()==0
assert digest.hexdigest() in json.dumps(proof)
for j,donors in extended.items():
    support[j]=tuple(sorted(donors|set(support[j] or ())))
print(json.dumps(dict(centroid_slot_catalog_sha256=digest.hexdigest(),targeted_receiver_rows=len(target),
                 receiver_rows_with_added_slots=len(extended),additional_safe_edges=sum(map(len,extended.values())),
                 outside_distance_limit=supra)),flush=True)


alternate=defaultdict(set)
by_role={}
verified=0
for role,typed,first in (('skin',15,0),('fat',16,48573)):
    witness=source_file(f'carrier_witness_{role}')
    path=Path(f'fine-{role}-alternate-witnesses.csv')
    count=0
    with path.open(newline='') as f:
        for r in csv.reader(f):
            if len(r)!=26:
                raise ValueError(r)
            record,side,kind,carrier_index,local,donor,_,old_row,material,*detail=r
            k,i,carrier=int(donor),int(old_row),int(carrier_index)+first
            assert int(material)==typed==int(old_mat[i]) and side.lower()==role
            assert int(k)==anchor_parent[carrier]
            xyz=tuple(map(float,r[19:22]))
            assert math.dist(xyz,anchors[carrier])<1e-11
            assert math.dist(tuple(map(float,r[22:25])),old_center[i])<1e-6
            assert abs(math.dist(xyz,old_center[i])-float(r[25]))<1e-6
            verified+=1
            row=int(original_to_current[i])
            if row<0:
                # An existing witness identifies its old receiver surface.
                # Mixed ancestry cannot inherit a precise geometric label here.
                permitted=[j for j in replacement_ancestors[i]
                           if np.all(labels[cavities[int(origins[j])]]==labels[i])]
            else:
                permitted=[row]
            for j in permitted:
                if k not in (support[j] or ()) and available[k]>0. and math.dist(xyz,new_center[j])<=maximum_route_distance:
                    alternate[j].add(k)
            count+=1
    by_role[role]={'source_catalog':str(witness),'witness_count':count}
for j,donors in alternate.items():
    support[j]=tuple(sorted(donors|set(support[j] or ())))
print(json.dumps(dict(authenticated_witness_rows=verified,usable_receiver_rows=len(alternate),
                 additional_routes=sum(map(len,alternate.values())),by_role=by_role)),flush=True)

# Eliminating receiver variables with identical donor incidence is exact for
# capacity feasibility. The sum of intervals is the interval of admissible
# group mass, and selecting one density per group recovers valid child bounds.
groups={}
keys=[]
group_for_cell=np.full(len(mat),-1,dtype=np.int32)
for i,donors in enumerate(support):
    if donors:
        key=(int(mat[i]),donors)
        if key not in groups:
            groups[key]=len(keys)
            keys.append(key)
        group_for_cell[i]=groups[key]
del support
occupied=group_for_cell>=0
ng=len(groups)
kg=group_for_cell[occupied]
lo=np.bincount(kg,weights=lower[occupied],minlength=ng)
hi=np.bincount(kg,weights=upper[occupied],minlength=ng)
singleton_groups=np.array([k for k,(_,g) in enumerate(keys) if len(g)==1])
singleton_parents=np.array([keys[k][1][0] for k in singleton_groups])
singleton_minimum=np.bincount(singleton_parents,weights=lo[singleton_groups],minlength=source_count)
destination_capacity=np.zeros(source_count)
for k,(_,donors) in enumerate(keys):
    destination_capacity[list(donors)]+=hi[k]
overflow=np.maximum(0.,singleton_minimum-available)
unroutable=np.maximum(0.,available-destination_capacity)
report=dict(original_positive_flow_rows=n,retained_route_sets=ng,
            source_flow_sha256=all_mass.hexdigest(),distance_pruned_edges=discarded,
            cells_without_safe_route=int(np.count_nonzero(~occupied)),
            mass_to_cover_unrouted_cells_kg=float(lower[~occupied].sum()),
            infeasible_singleton_donors=int(np.count_nonzero(overflow>1e-12)),
            shortage_for_singleton_children_kg=float(overflow.sum()),
            donors_with_insufficient_total_capacity=int(np.count_nonzero(unroutable>1e-12)),
            overcapacity_donor_mass_kg=float(unroutable.sum()),
            operating_point='exact pinned source positions plus prior scoped centroid slots',
            result='infeasible_subgraph',
            local_volume_patch_sha256=hashlib.sha256(Path('fine-mass-local-position-delta.npz').read_bytes()).hexdigest(),
            no_unwitnessed_donor_edges=True,native_mass_admission=False)
report['shortage_donor_rows']=[[int(i),float(overflow[i])] for i in np.flatnonzero(overflow>1e-12)]
report['overcapacity_donor_rows']=[[int(i),float(unroutable[i])] for i in np.flatnonzero(unroutable>1e-12)]
report['blocked_positive_routes']=blocked
report_path=Path('fine-density-transport-cut.json')
report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
# A failed capacity bound already proves that this support subset cannot pass.
if not (report['cells_without_safe_route'] or report['infeasible_singleton_donors']
        or report['donors_with_insufficient_total_capacity']):
    slots=np.array([j for _,g in keys for j in g],dtype=np.int32)
    index=np.repeat(np.arange(ng),[len(g) for _,g in keys])
    ne=len(slots)
    cols=np.arange(ne)
    donor=coo_matrix((np.ones(ne),(slots,cols)),shape=(source_count,ne)).tocsr()
    group=coo_matrix((np.ones(ne),(index,cols)),shape=(ng,ne)).tocsr()
    interval=vstack((group,-group)).tocsr()
    print(json.dumps({'transport_lp_edges':int(ne),'transport_lp_groups':ng}),flush=True)
    solution=linprog(np.zeros(ne),A_eq=donor,b_eq=available*1e3,A_ub=interval,b_ub=np.r_[hi,-lo]*1e3,
                     bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9})
    if not solution.success:
        absence = coo_matrix((source_count, 2*ng)).tocsr()
        slack = -eye(2*ng,format='csr')
        relaxed = linprog(np.r_[np.zeros(ne),np.ones(2*ng)],
            A_eq=hstack((donor,absence),format='csr'),b_eq=available*1e3,
            A_ub=hstack((interval,slack),format='csr'),b_ub=np.r_[hi,-lo]*1e3,
            bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9})
        print(json.dumps({'minimal_slack_status':relaxed.message}),flush=True)
        assert relaxed.success
        group_adjustment = relaxed.x[ne:].reshape(2,ng)*1e-3
        donor_potential = relaxed.eqlin.marginals
        group_potential = relaxed.ineqlin.marginals[:ng] - relaxed.ineqlin.marginals[ng:]
        target_groups = np.flatnonzero(np.any(group_adjustment>1e-12,axis=0))
        report['minimum_total_density_violation_kg'] = float(group_adjustment.sum())
        report['affected_density_groups'] = int(len(target_groups))
        report['largest_violations'] = [
            dict(group=int(group_id),source_parents=keys[group_id][1],
                 upper_violation_kg=float(group_adjustment[0,group_id]),
                 lower_violation_kg=float(group_adjustment[1,group_id]),
                 original_cells=int(np.count_nonzero(group_for_cell==group_id)))
            for group_id in target_groups[np.argsort(-group_adjustment[:,target_groups].sum(axis=0))[:20]]
        ]
        np.savez_compressed('fine-density-dual-deficit.npz',group_adjustment=group_adjustment,
                            donor_potential=donor_potential,group_potential=group_potential,
                            group_for_cell=group_for_cell,original_to_current=original_to_current)
    report['result']='feasible' if solution.success else 'optimizer_infeasible'
    report['optimizer_status']=solution.message
    if solution.success:
        routed=solution.x*1e-3
        donor_error=np.abs(donor@routed-available).max()
        group_mass=group@routed
        group_volume=np.bincount(kg,weights=v[occupied],minlength=ng)
        values=group_mass[kg]*v/group_volume[kg]
        report['maximum_donor_residual_kg']=float(donor_error)
        report['maximum_cell_bound_residual_kg']=float(max(np.maximum(0.,lower-values).max(),
                                                           np.maximum(0.,values-upper).max()))
        np.savez_compressed('fine-volume-balanced-mass-flow.npz',donor=slots,group=index,kg=routed,
                            group_for_cell=group_for_cell,element_mass_kg=values)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
