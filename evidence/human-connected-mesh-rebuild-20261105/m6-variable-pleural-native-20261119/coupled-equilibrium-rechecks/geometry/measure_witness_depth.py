"""Measure signed crossing depth on retained native geometry."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent
P=HERE.with_name("resting-clearance-and-atrium-007")
G=HERE.with_name("resting-clearance-and-atrium-006")
sys.path[:0]=[str(P),str(G)]
from fit_edge_dependency_repair import capture_in_memory
from fit_skin_tissue_depth import nearest_faces
from register_tissue_bellies import READER, SKIN

run=P/"recomputed-support-native"
reference=json.loads((run/"step-8000-skin-exact.json").read_text())
witness=run/"native/accepted-geometry/step-8000.mrvpack"
bank_hits={(b["semantic"],b["stable_id"]):b for b in reference["surfaces"] if b["count"]}
with capture_in_memory(witness) as (data, bodies, receipt):
    assert receipt["pack_file_sha256"]==reference["accepted_pack_sha256"]
    surfaces={k:(p,f) for k,_,p,f in READER["primitives"](data,bodies) if k==SKIN or k in bank_hits}
skin,skin_faces=surfaces[SKIN]
triangles=skin[skin_faces].astype(float)*1000.
index=cKDTree(triangles.mean(axis=1))
depth=[]
for key,report in bank_hits.items():
    target, faces=surfaces[key]
    pairs=np.array(report["triangle_pairs"])
    test_ids=np.unique(faces[pairs[:,1]])
    xyz=target[test_ids].astype(float)*1000.
    n,d=[],[]
    for lo in range(0,len(xyz),512):
        f,bary,norm,signed,_=nearest_faces(xyz[lo:lo+512],index,triangles)
        foot=np.einsum("vi,vij->vj",bary,triangles[f])
        length=np.linalg.norm(xyz[lo:lo+512]-foot,axis=1)
        valid=(signed>.0001)&(signed>=.85*length)
        n.extend(signed[valid])
        d.extend(test_ids[lo:lo+512][valid])
    signed=np.asarray(n)
    out=dict(bank=list(key),pairs=report["count"],witness_vertices=len(test_ids),
             positively_oriented=len(signed),
             mm_above_4=int((signed>4).sum()),mm_above_10=int((signed>10).sum()))
    if len(signed):
        out["quantiles_mm"]=np.quantile(signed,[.5,.9,.99,1.]).tolist()
        out["deepest_vertex"]=int(d[int(np.argmax(signed))])
        out["deepest_world_xyz_m"]=target[out["deepest_vertex"]].astype(float).tolist()
    depth.append(out)
result=dict(accepted_pack_sha256=reference["accepted_pack_sha256"],scope="Signed nearest sampled skin triangle. Exact intersection pairs are independently retained. This proximity measure does not certify containment.",banks=sorted(depth,key=lambda r:r["pairs"],reverse=True))
(HERE/"signed-witness-depths.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result["banks"][:10]))
