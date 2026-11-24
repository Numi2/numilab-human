"""Try exact-boundary material cavity reconstructions around the native rolled-back element."""
from itertools import combinations
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from material_cavity_flips import boundary,faces
from supported_material_admission import geom
from retessellate_witness import solve
from compile_connected_joint import digest
def altitude(x,t):
    p=x[t];d=abs(np.linalg.det(p[:,1:]-p[:,:1]));areas=[]
    for i in range(4):
        f=np.delete(p,i,axis=1);areas.append(np.linalg.norm(np.cross(f[:,1]-f[:,0],f[:,2]-f[:,0]),axis=1))
    return d/np.maximum.reduce(areas)
def run(work):
    v=work.parent/'qualified-material'
    out=work/'witness-cavity'
    out.mkdir(exist_ok=True)
    d=json.loads((work/'native-motion.json').read_bytes())
    e=d['failure']['tet']
    get=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3))
    x=get(v/'mesh.node');y=get(work/'selected-pose/current.node')
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    t=raw[:,:4]-1;lab=np.loadtxt(v/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8')
    if t[e].tolist()!=d['failure']['physical_nodes']:raise ValueError('Wrong native witness')
    near=np.flatnonzero(np.isin(t,t[e]).sum(1)>=1)
    same=np.all(lab[:,1:4]==lab[e,1:4],axis=1)
    adj=defaultdict(set)
    for k in near[same[near]]:
        for face in combinations(sorted(t[k]),3):adj[face].add(int(k))
    seeds=[{e}]
    for _ in range(2):
        new=[]
        for cluster in seeds:
            for c in cluster:
                for f in combinations(sorted(t[c]),3):
                    for n in adj[f]-cluster:new.append(frozenset(cluster|{n}))
        seeds+=sorted(set(new),key=lambda r:(len(r),sorted(r)))
    # Add complete connected edge and interior-node stars. Every material face and
    # neighboring cell outside each tested cavity is fixed exactly.
    for edge in combinations(t[e],2):
        s=near[np.isin(t[near],edge).sum(1)==2]
        if same[s].all():seeds.append(frozenset(s))
    s=near[(t[near]==t[e,0]).any(1)]
    if same[s].all():seeds.append(frozenset(s))
    # Combine all face neighbors and complete edge stars. A nearly collinear
    # internal face can be locked until both sides and the adjacent edge open.
    regular=set().union(*(adj[f] for f in combinations(sorted(t[e]),3)))
    extended=[regular]
    for side in combinations(t[e],2):
        star=set(near[np.isin(t[near],side).sum(1)==2])
        if all(same[list(star)]):extended.append(regular | star)
    extended.extend([regular|set(s), regular|set().union(*extended)])
    seeds.extend(map(frozenset,extended))
    attempts=[];best=None
    for subset in sorted(set(map(frozenset,seeds)),key=lambda v:(len(v),sorted(v))):
        rows=np.array(sorted(subset));old=t[rows];nodes=np.unique(old)
        if len(rows)<2 or len(nodes)>18:continue
        other=np.flatnonzero(np.isin(t,nodes).sum(1)>=2);ex=np.setdiff1d(other,rows)
        edges={edge for row in t[ex] for edge in combinations(sorted(set(row)&set(nodes)),2)}
        existing={edge for row in old for edge in combinations(sorted(row),2)}
        banned=edges-existing
        targets={tuple(sorted(t[e]))}
        # Apply the existing 2x altitude gate to all replacement cells.
        prop=solve(x,y,old,np.ones(len(x),bool),banned,targets)
        rec=dict(cells=rows.tolist(),candidate=prop is not None)
        if prop is not None:
            h=min(altitude(x,prop).min(),altitude(y,prop).min())
            rec.update(min_altitude_um=float(h*1e6),elements=len(prop))
            if boundary(old)!=boundary(prop):raise ValueError('Boundary moved')
            if best is None or h>best[0]:best=(h,rows,prop)
        attempts.append(rec)
    report=dict(witness=e,measured_initial_J=d['failure']['initial_J'],prior_minimum_altitude_um=float(altitude(x,t[e:e+1])[0]*1e6),attempts=attempts)
    if best:
        h,rows,prop=best
        np.savez_compressed(out/'candidate.npz',old_cells=rows,new_tetrahedra=prop)
        report.update(best_minimum_altitude_um=h*1e6,old_cells=rows.tolist(),new_cells=len(prop),mesh=digest(v/'mesh.ele'),candidate=digest(out/'candidate.npz'))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k!='attempts'})
if __name__=='__main__':run(Path(__file__).resolve().parent)
