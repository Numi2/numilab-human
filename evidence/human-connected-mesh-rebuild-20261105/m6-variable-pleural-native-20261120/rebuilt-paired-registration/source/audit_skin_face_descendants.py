"""Identify changed NHSKIN face patches by exact directed boundary lineage."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from rebind_current_skin import oriented,sha256

sys.path.insert(0,str(Path('../resting-clearance-and-atrium-006')))
from fit_clearance_patches import decode

receipt=json.loads(Path('skin-source-rebind-002/receipt.json').read_text())
paths=[Path(p) for p in receipt['inputs_sha256'] if p.endswith('.nhskin')]
assert all(sha256(p)==receipt['inputs_sha256'][str(p)] for p in paths)
old,new=[decode(p.read_bytes()) for p in paths]
previous,current=[x['inds'] for x in (old,new)]
old_keys=[oriented(f) for f in previous]
new_keys=[oriented(f) for f in current]
a,b=Counter(old_keys),Counter(new_keys)
assert all(n==1 for n in (a-b).values()) and all(n==1 for n in (b-a).values())
removed_keys=set(a-b)
added_keys=set(b-a)
removed=[i for i,k in enumerate(old_keys) if k in removed_keys]
added=[i for i,k in enumerate(new_keys) if k in added_keys]

def components(faces, changed):
    edges=defaultdict(list)
    for row in changed:
        u,v,w=map(int,faces[row])
        for src,dst in ((u,v),(v,w),(w,u)):
            edges[tuple(sorted((src,dst)))].append((src,dst,row))

    neighbors=defaultdict(set)
    for connections in edges.values():
        if len(connections)>1:
            ids={r for _,_,r in connections}
            for r in ids:
                neighbors[r].update(ids-{r})
    todo=set(changed)
    out={}
    while todo:
        component=set()
        stack=[min(todo)]
        while stack:
            row=stack.pop()
            if row in todo:
                todo.remove(row)
                component.add(row)
                stack.extend(neighbors[row])
        local_edges={tuple(sorted((int(u),int(v)))) for r in component
                     for u,v in zip(faces[r],np.roll(faces[r],-1))}
        boundary=frozenset((u,v) for e in local_edges if len(edges[e])==1 for u,v,_ in edges[e])
        verts=set(map(int,faces[sorted(component)].ravel()))
        nonmanifold=sum(len(edges[e])>2 or (len(edges[e])==2 and edges[e][0][:2]!=edges[e][1][:2][::-1])
                        for e in local_edges)
        out[min(component)]=dict(rows=sorted(component),boundary=boundary,verts=verts,
                                 euler=len(verts)-len(local_edges)+len(component),nonmanifold=nonmanifold)
    return out

out_old,out_new=components(previous,removed),components(current,added)
by_signature=defaultdict(list)
for i,c in out_new.items():
    if c['boundary'] and not c['nonmanifold']:
        by_signature[c['boundary'],c['euler']].append((i,c))
missing=json.loads(Path('paired-registration-final/receipt.json').read_text())['unresolved_face_rows']
remaining={v['old_face'] for v in missing}
rows=[]
for i,comp in out_old.items():
    affected=remaining & set(comp['rows'])
    if not affected:
        continue
    matches=by_signature[comp['boundary'],comp['euler']] if not comp['nonmanifold'] else []
    match=matches[0] if len(matches)==1 else None
    rows.append(dict(old_component=i,old_faces=comp['rows'],affected_face_parents=sorted(affected),
                     old_boundary_edges=len(comp['boundary']),old_nonmanifold_edges=comp['nonmanifold'],
                     new_component=match[0] if match else None,
                     new_faces=match[1]['rows'] if match else [],
                     shared_interior_vertices=(len(comp['verts']&match[1]['verts']) if match else None),
                     complete_vertex_set_retained=match is not None and comp['verts']==match[1]['verts']))
report=dict(removed_faces=len(removed),added_faces=len(added),old_components=len(out_old),
            new_components=len(out_new),exact_changed_patch_matches=sum(row['new_component'] is not None for row in rows),
            unmatched_face_parents=sorted(remaining-set().union(*(set(row['affected_face_parents']) for row in rows
                                  if row['new_component'] is not None))),patches=rows)
Path('skin-face-descendants-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='patches'}))
