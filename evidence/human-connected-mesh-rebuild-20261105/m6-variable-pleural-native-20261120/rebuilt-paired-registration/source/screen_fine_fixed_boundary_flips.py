"""Construct disjoint fine-volume flip candidates with fixed points and the original quality gate."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

from check_current_parent_volume import volumes
from rebind_current_skin import sha256

script=Path.home()/'Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/bounded625-local-cavity-repair'
sys.path.insert(0,str(script))
from local_fixed_boundary_cavity_repair_625 import FACE_PATTERN, _ring

data=Path('fine-transfer-screen-map.npz')
with np.load(data,allow_pickle=False) as x:
    geometry=(x['old_world_m']+x['fine_node_delta_m']).astype(np.float32).astype(np.float64)
    roles=x['roles']
compiled=Path('../captured-fat-sliver-495/proposal/compiler/volume-export/volume/mesh.ele')
rows=np.loadtxt(compiled,skiprows=1,dtype=np.int32)
tets,material=rows[:,1:5]-1,rows[:,5]
assert len(tets)==1448830
q=np.empty(len(tets))
for first in range(0,len(tets),120_000):
    s=slice(first,first+120_000)
    q[s]=volumes(geometry,tets[s])[1]
invalid=np.flatnonzero(q<=0)
assert len(invalid)==195
min_quality=1e-5
corners=tets.ravel()
index=np.argsort(corners,kind='stable')//4
offsets=np.r_[0,np.cumsum(np.bincount(corners,minlength=len(geometry)))]

def hits(nodes):
    corner=np.asarray(nodes,dtype=int)
    lists=[index[offsets[node]:offsets[node+1]] for node in corner]
    first=min(lists,key=len)
    return set(int(cell) for cell in first if all(np.any(lst==cell) for lst in lists))

def faces(tet):
    return tet[FACE_PATTERN]

def chain(tetrahedra):
    oriented=Counter()
    for tet in tetrahedra:
        for a,b,c in faces(tet):
            orientation=1 if sum((a>b,a>c,b>c))%2==0 else -1
            oriented[tuple(sorted((int(a),int(b),int(c))))]+=orientation
    return {f:v for f,v in oriented.items() if v}

def make_proposals(row):
    cell=tets[row]
    for face in faces(cell):
        guard=hits(face)
        if len(guard)!=2:
            continue
        a,b,c=sorted(map(int,face))
        d,e=[int(v) for r in sorted(guard) for v in tets[r] if v not in face]
        if d==e or hits((d,e)):
            continue
        yield '2-3',guard,np.asarray(((d,e,a,b),(d,e,b,c),(d,e,c,a)),int)
    for edge in combinations(cell,2):
        guard=hits(edge)
        if len(guard) not in (3,4):
            continue
        edge=tuple(map(int,edge))
        ring=_ring(tets[sorted(guard)],edge)
        if not ring:
            continue
        a,b,c,*ending=ring
        d,e=edge
        if not ending:
            if hits((a,b,c))-guard:
                continue
            yield '3-2',guard,np.asarray(((a,b,c,d),(a,b,c,e)),int)
            continue
        x=ending[0]
        for diagonal,separator in (((a,c),((a,b,c),(a,c,x))),
                                   ((b,x),((b,c,x),(b,x,a)))):
            if hits(diagonal)-guard or any(hits(f)-guard for f in separator):
                continue
            yield '4-4',guard,np.asarray([(*f,node) for f in separator for node in (d,e)],int)

used=set()
staged_keys=set()
selected=[]
tested=Counter()
for row in invalid:
    if int(row) in used:
        continue
    best=None
    for kind,retired,replacement in make_proposals(row):
        tested[kind]+=1
        ri=sorted(retired)
        if np.any(material[ri]!=material[row]) or retired&used:
            continue
        old=tets[ri]
        dv,vq=volumes(geometry,replacement)
        flip=dv<0
        replacement[flip]=replacement[flip][:,[1,0,2,3]]
        if np.any(np.abs(vq)<min_quality) or chain(old)!=chain(replacement):
            continue
        keys={tuple(sorted(map(int,t))) for t in replacement}
        if len(keys)!=len(replacement) or keys&staged_keys or any(hits(key)-retired for key in keys):
            continue
        old_volume=volumes(geometry,old)[0].sum()
        if not np.isclose(old_volume,np.abs(dv).sum(),rtol=1e-11,atol=1e-22):
            continue
        old_bad=int(np.count_nonzero(q[ri]<=0))
        score=(old_bad,float(np.abs(vq).min()))
        if best is None or score>best[0]:
            best=(score,retired,keys,dict(kind=kind,retired=ri,material_id=int(material[row]),
                                       new=replacement.tolist(),removed_inversions=old_bad,
                                       replacement_normalized_det_min=score[1],
                                       parent_union_from_retired_current_rows=ri))
    if best:
        _,retired,keys,entry=best
        selected.append(entry)
        used.update(retired)
        staged_keys.update(keys)
remaining=[int(row) for row in invalid if row not in used]
rest=np.flatnonzero((q>0)&(q<min_quality))
ready=dict(input_fine_map_sha256=sha256(data),base_topology_sha256=sha256(compiled),
           positive_minimum_gate=min_quality, tested=dict(tested),accepted_flips=len(selected),
           starting_inverted=len(invalid),remaining_inverted=len(remaining),
           remaining_all_exterior=int(np.count_nonzero(np.all(roles[tets[remaining]]=='O',axis=1))),
           remaining_inverted_rows=remaining,remaining_positive_subthreshold=int(sum(row not in used for row in rest)),
           replacements=selected,externally_certified=False,semantic_mass_compiled=False,physical_admission=False)
Path('fine-fixed-boundary-flips.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps({k:v for k,v in ready.items() if k not in ('replacements','remaining_inverted_rows')}))
