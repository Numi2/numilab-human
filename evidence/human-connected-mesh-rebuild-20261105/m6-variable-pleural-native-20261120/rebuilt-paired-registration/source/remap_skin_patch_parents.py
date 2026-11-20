"""Remap scalar face anchors through the current mesh's boundary-matched retriangulations."""
import json
from pathlib import Path

import numpy as np

from audit_skin_face_descendants import previous, current, out_old, out_new, report
from rebind_current_skin import sha256

def polygon_chart(boundary):
    successor={a:b for a,b in boundary}
    assert len(successor)==len(boundary) and len(set(successor.values()))==len(boundary)
    first=min(successor)
    vertices=[first]
    while successor[vertices[-1]]!=first:
        vertices.append(successor[vertices[-1]])
        assert len(vertices)<=len(boundary)
    assert len(vertices)==len(boundary)
    angles=2*np.pi*np.arange(len(vertices))/len(vertices)
    return dict(zip(vertices,np.stack((np.cos(angles),np.sin(angles)),axis=1)))

def cross(u,v):
    return u[...,0]*v[...,1]-u[...,1]*v[...,0]

def area(p):
    return 0.5*float(cross(p,np.roll(p,-1,axis=0)).sum())

def overlap(subject, clip):
    p=subject
    for a,b in zip(clip,np.roll(clip,-1,axis=0)):
        if not len(p):
            return 0.
        d=cross(b-a,p-a)
        result=[]
        for i,j in enumerate(np.roll(np.arange(len(p)),-1)):
            i_inside=d[i]>=0
            j_inside=d[j]>=0
            if i_inside:
                result.append(p[i])
            if i_inside!=j_inside:
                result.append(p[i]+(p[j]-p[i])*(d[i]/(d[i]-d[j])))
        p=np.asarray(result)
    return area(p) if len(p)>=3 else 0.

def coordinates(point,triangle):
    e1,e2=triangle[1:]-triangle[0]
    u,v=np.linalg.solve(np.column_stack((e1,e2)),point-triangle[0])
    return np.asarray([1-u-v,u,v])

source=Path('paired-registration-final/registration.npz')
with np.load(source,allow_pickle=False) as payload:
    registration={k:payload[k] for k in payload.files}
parents=registration['current_surface_face_parent']
old_parents=registration['old_surface_face_parent']
assert not report['unmatched_face_parents']
matches={}
for row in report['patches']:
    old=out_old[row['old_component']]
    new=out_new[row['new_component']]
    chart=polygon_chart(old['boundary'])
    assert old['verts']==new['verts']==set(chart)
    original=np.asarray([[chart[int(v)] for v in previous[f]] for f in old['rows']])
    replacement=np.asarray([[chart[int(v)] for v in current[f]] for f in new['rows']])
    assert min(map(area,original))>0 and min(map(area,replacement))>0
    weights=np.asarray([[overlap(p,q) for q in replacement] for p in original])
    assert np.max(np.abs(weights.sum(axis=1)-np.asarray(list(map(area,original)))))<1e-12
    assert np.max(np.abs(weights.sum(axis=0)-np.asarray(list(map(area,replacement)))))<1e-12
    for i,face in enumerate(old['rows']):
        anchor=original[i].mean(axis=0)
        coords=np.asarray([coordinates(anchor,q) for q in replacement])
        candidates=np.flatnonzero(np.min(coords,axis=1)>-1e-12)
        assert len(candidates) and np.all(weights[i,candidates]>1e-12)
        preferred=min(candidates,key=lambda j:(-round(float(weights[i,j]),12),new['rows'][j]))
        matches[face]=dict(original_face=int(face),relative_partition_overlap=[
            [new['rows'][j],float(weights[i,j]/area(original[i]))]
            for j in np.flatnonzero(weights[i]>1e-12)],
            current_face=int(new['rows'][preferred]),
            anchor_reference='historical fine triangle centroid in common directed boundary chart',
            anchor_barycentric=coords[preferred].tolist())

selected=[]
for row in np.flatnonzero(parents<0):
    entry=matches[int(old_parents[row])]
    parents[row]=entry['current_face']
    selected.append(dict(coarse_row=int(row),**entry))
assert len(selected)==65 and np.all(parents>=0) and np.all(parents<len(current))
out=Path('paired-registration-with-face-ancestry')
out.mkdir(exist_ok=False)
np.savez_compressed(out/'registration.npz',**registration)
evidence=dict(exact_retained_face_parents=len(parents)-len(selected),
              remapped_face_parents=len(selected),old_current_patch_pairs=len(report['patches']),
              scalar_representative_and_overlap=selected,unresolved_face_parents=0,
              registration_sha256=sha256(out/'registration.npz'),runtime_admitted=False)
(out/'receipt.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps({k:v for k,v in evidence.items() if k!='scalar_representative_and_overlap'}))
