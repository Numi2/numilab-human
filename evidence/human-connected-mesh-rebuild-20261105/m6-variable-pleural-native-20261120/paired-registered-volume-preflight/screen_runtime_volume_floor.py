"""Find exact material-boundary preserving cavities for the native absolute volume floor."""
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
mesh = root.parent / "resting-clearance-and-atrium-016/repaired-tetgen"
positions = np.loadtxt(mesh/"mesh.node", skiprows=1, usecols=(1, 2, 3))
cells = np.loadtxt(mesh/"mesh.ele", skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=np.int32)
tets,material = cells[:,:4]-1,cells[:,4]
data = np.load(root/"paired-native-density-closed-flow.npz")
groups = data["group_for_cell"]
volume = np.fromfile(root/"native-mass-reference-f64.bin", dtype=np.float64)
failure = np.flatnonzero(volume <= 1.e-18)
faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
edges = tuple(combinations(range(4),2))

def measure(rows):
    x=positions[np.asarray(rows)]
    delta=x[:,1:]-x[:,:1]
    det=np.einsum("ij,ij->i",delta[:,0],np.cross(delta[:,1],delta[:,2]))
    span=np.max([np.linalg.norm(x[:,a]-x[:,b],axis=1) for a,b in edges],axis=0)
    return det/6.,abs(det)/span**3

def boundary(rows):
    out=Counter()
    for tet in rows:
        for face in faces:
            a,b,c=map(int,tet[list(face)])
            sign= -1 if (int(a>b)+int(a>c)+int(b>c))%2 else 1
            out[tuple(sorted((a,b,c)))]+=sign
    return {row: value for row,value in out.items() if value}

def key(rows):
    return sorted(tuple(sorted(tet)) for tet in rows)

reports=[]
for target in failure:
    n=tets[target]
    patch=np.flatnonzero(np.any(np.isin(tets,n),axis=1))
    nearby=tets[patch]
    face_bank,edge_bank=defaultdict(list),defaultdict(list)
    for i,tet in zip(patch,nearby):
        for f in combinations(sorted(tet),3):
            face_bank[f].append(int(i))
        for e in combinations(sorted(tet),2):
            edge_bank[e].append(int(i))
    candidates=[]
    tested=set()
    def add(old,new):
        if target not in old or len(set(material[old])) != 1:
            return
        oldkey=tuple(sorted(old))
        new=np.asarray(new,dtype=np.int32)
        identity=(oldkey,tuple(key(new)))
        if identity in tested:
            return
        tested.add(identity)
        if len({tuple(sorted(tet)) for tet in new}) != len(new):
            return
        old_tets=tets[old]
        v,q=measure(new)
        if not np.all(abs(v)>1.01e-18) or not np.all(q>=1.001e-5):
            return
        invert=v<0
        new[invert,:2]=new[invert][:,:2][:,::-1]
        if boundary(new)!=boundary(old_tets):
            return
        if abs(abs(v).sum()-volume[old].sum())>1e-10*volume[old].sum():
            return
        incident_external=set(key(nearby[np.isin(patch,old,invert=True)]))
        if any(tuple(sorted(tet)) in incident_external for tet in new):
            return
        candidates.append({"retired":oldkey,"inserted":new.tolist(),"quality_min":float(q.min()),
                           "volume_min_m3":float(abs(v).min()),
                           "group_count":len(set(groups[old])),
                           "old_groups":groups[old].tolist(),"material":int(material[old[0]])})
    for face,owners in face_bank.items():
        if len(owners)==2 and target in owners:
            a,b=(next(int(u) for u in tets[r] if u not in face) for r in owners)
            if a!=b and tuple(sorted((a,b))) not in edge_bank:
                add(owners,[(a,b,face[i],face[(i+1)%3]) for i in range(3)])
    for edge,owners in edge_bank.items():
        if target not in owners:
            continue
        ring=set.union(*(set(tets[r]) for r in owners))-set(edge)
        if len(owners)==3 and len(ring)==3:
            add(owners,[(*sorted(ring),u) for u in edge])
        if len(owners)==4 and len(ring)==4:
            ring_edges=sorted(tuple(sorted(set(tets[r])-set(edge))) for r in owners)
            for diagonal in combinations(sorted(ring),2):
                if diagonal in ring_edges:
                    continue
                cycles=[(edge[0],u) for u in sorted(ring-set(diagonal))]
                cycles += [(edge[1],u) for u in sorted(ring-set(diagonal))]
                add(owners,[(*diagonal,*pair) for pair in cycles])
    candidates.sort(key=lambda r:(r["group_count"],-r["volume_min_m3"],-r["quality_min"]))
    reports.append({"cell":int(target),"local_rows":len(patch),"candidate_count":len(candidates),
                    "candidates":candidates[:10],
                    "neighbor_tets":[{"row":row,"material":int(material[row]),"group":int(groups[row]),
                                      "volume_m3":v} for face,owners in face_bank.items() if target in owners
                                     for row in owners if row!=target for v in [float(volume[row])] ]})
result={"status":"SCREENED","volume_floor_m3":1.e-18,"failures":reports}
(root/"runtime-volume-floor-cavity-screen.json").write_text(json.dumps(result,indent=2)+"\n")
print([(r["cell"],r["candidate_count"],[(x["group_count"],x["volume_min_m3"]) for x in r["candidates"]]) for r in reports])
