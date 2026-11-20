"""Carry the source-carrier flow across the local connectivity changes.

Identity is retained for unchanged cell rows. For a changed cavity we carry
each incoming source-parent mass separately into its positive new volumes.
No rigid donation, constitutive calibration or native activation occurs here.
"""
from collections import defaultdict
import csv
import hashlib
import json
import subprocess

import numpy as np

from candidate import COMPILER, load

old,points,tets,material,origins,changed,_,old_tets,old_material=load()
ledger=COMPILER/"mass-allocation/allocator-output"
source=ledger/"flow.csv.zst"
old_count=len(old_tets)
def volumes(x,t):
    value=np.empty(len(t))
    for lo in range(0,len(t),80000):
        block=slice(lo,lo+80000)
        x0,x1,x2,x3=np.moveaxis(x[t[block]].astype(float),1,0)
        value[block]=np.einsum("ij,ij->i",x1-x0,np.cross(x2-x0,x3-x0))/6
    return value

v=volumes(points,tets)
v0=volumes(old,old_tets)
assert np.all(v>0) and np.all(v0>0)
old_m=np.zeros(old_count,dtype=np.float64)
source_m=np.zeros(28632,dtype=np.float64)
retired=set(np.flatnonzero(~np.isin(np.arange(old_count),origins[origins>=0])).tolist())
parents_by_original=defaultdict(dict)
digester=hashlib.sha256()
proc=subprocess.Popen(["/opt/homebrew/bin/zstd","-qcd",str(source)],stdout=subprocess.PIPE)
line=next(proc.stdout)
digester.update(line)
assert line.startswith(b"child_row,source_parent_row,mass_kg,")
for raw in proc.stdout:
    digester.update(raw)
    child,parent,mass,*_=raw.split(b",")
    i,j,m=int(child),int(parent),float(mass)
    assert 0<=i<old_count and 0<=j<len(source_m) and m>0 and np.isfinite(m)
    old_m[i]+=m
    source_m[j]+=m
    if i in retired:
        row=parents_by_original[i]
        row[j]=row.get(j,0.)+m
assert proc.wait()==0
receipt=json.loads((COMPILER/"mass-admission/mass-receipt.json").read_text())
pins=receipt["output_sha256"]
assert digester.hexdigest() in json.dumps(pins),str(pins)[:700]
assert np.all(old_m>0)
# The replacement groups form small separated connected components. Later
# flips may retire part of an earlier group; merge all final descendants that
# inherit any common original parent before distributing that parent's mass.
new_ids=np.flatnonzero(origins<0)
group_to_children=defaultdict(list)
remaining_groups=set(int(r) for r in np.unique(origins[new_ids]))
for child in new_ids:
    group_to_children[int(origins[child])].append(int(child))
relation=defaultdict(set)
for group in remaining_groups:
    for parent in changed[group]:
        if parent in retired:
            relation[parent].add(group)
assert len(relation)==len(retired)
new_m=np.zeros(len(origins))
stable=origins>=0
new_m[stable]=old_m[origins[stable]]
source_after=source_m.copy()
assigned={}
cavity_count=0
while remaining_groups:
    root=min(remaining_groups)
    queue=[root]
    connected=set()
    former=set()
    while queue:
        group=queue.pop()
        if group in connected:
            continue
        connected.add(group)
        for parent in changed[group]:
            if parent not in retired or parent in former:
                continue
            former.add(parent)
            queue.extend(sorted(relation[parent]))
    remaining_groups.difference_update(connected)
    assert former
    c=sorted(i for g in connected for i in group_to_children[g])
    indices=np.asarray(c)
    outgoing={}
    for parent in sorted(former):
        for owner,value in parents_by_original[parent].items():
            outgoing[owner]=outgoing.get(owner,0.)+value
            source_after[owner]-=value
    assert len(set(material[indices]))==1 and np.all(old_material[list(former)]==material[c[0]])
    weights=v[indices]/v[indices].sum()
    for c,w in zip(c,weights):
        assigned[c]={j:m*w for j,m in outgoing.items()}
        for j,m in assigned[c].items():
            new_m[c]+=m
            source_after[j]+=m
    cavity_count+=1
assert len(assigned)==len(new_ids)
assert np.all(new_m>0) and np.max(np.abs(source_after-source_m))<1e-13
assert abs(new_m.sum()-old_m.sum())<1e-12
nodal=np.bincount(tets.ravel(),weights=np.repeat(new_m*.25,4),minlength=len(points))
assert min(nodal)>0 and abs(nodal.sum()-new_m.sum())<1e-12
def tally(m,mat):
    return {str(i):float(m[mat==i].sum()) for i in (15,16)}
rho=new_m/v
by_mat={}
for label in (15,16):
    k=material==label
    by_mat[str(label)]={"old_nominal_density_kg_m3":float(np.median((old_m/v0)[old_material==label])),
                        "candidate_effective_density_quantiles_kg_m3":np.quantile(rho[k],[0,.001,.01,.5,.99,.999,1]).tolist()}
result=dict(source_flow_sha256=digester.hexdigest(),old_mass_kg=tally(old_m,old_material),
            new_mass_kg=tally(new_m,material),source_parent_mass_max_error_kg=float(np.max(np.abs(source_after-source_m))),
            nodal_mass_kg=float(nodal.sum()),original_rows_removed=len(retired),
            replacement_rows=len(new_ids),independent_cavities=cavity_count,
            cells=len(tets),nodes=len(points),density_audit=by_mat,
            current_graph_optimizer_replay=False, native_mass_admission=False)
with open("fine-cavity-mass-flow.csv","w",newline="") as out:
    rows=csv.writer(out,lineterminator="
")
    rows.writerow(("child_row","source_parent_row","mass_kg"))
    for i,donors in sorted(assigned.items()):
        for source,m in sorted(donors.items()):
            rows.writerow((i,source,format(m,".17g")))
np.savez_compressed("fine-source-carried-mass.npz",original_child_row=origins,element_mass_kg=new_m,nodal_mass_kg=nodal)
with open("fine-source-carried-mass.json","w") as out:
    json.dump(result,out,indent=2)
print(json.dumps(result))
