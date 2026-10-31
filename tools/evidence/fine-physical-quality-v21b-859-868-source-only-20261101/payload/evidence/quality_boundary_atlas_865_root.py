from pathlib import Path
import json,numpy as np,hashlib
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");atlas=json.loads((H/"quality-atlas-864-root.json").read_text())
out={}
for role,a in atlas.items():
 with np.load(a["input"]["path"],allow_pickle=False) as z:
  p=np.asarray(z["points_world_m_f32"],np.float32);t=np.asarray(z["tetrahedra"],np.int64);b=np.asarray(z["desired_boundary_faces"],np.int64)
 rows=np.asarray([r for g in a["bad_clusters"] for r in g["rows"]],np.int64);bt=t[rows];boundarynodes=np.zeros(len(p),bool);boundarynodes[np.unique(b)]=True
 bf={tuple(x) for x in np.sort(b,axis=1).tolist()}
 details=[]
 for row,tet in zip(rows,bt):
  faces=[tet[np.array(f)] for f in ((0,2,1),(0,1,3),(0,3,2),(1,2,3))]
  hits=[f for f in faces if tuple(sorted(f.tolist())) in bf]
  x=p[tet].astype(np.float64);edges=sorted(float(np.linalg.norm(x[i]-x[j])) for i in range(4) for j in range(i+1,4))
  details.append({"row":int(row),"boundary_vertices":int(boundarynodes[tet].sum()),"boundary_faces":len(hits),"edge_lengths_m":edges,"points":p[tet].tolist()})
 out[role]={"bad_count":len(rows),"boundary_node_counts":{str(i):sum(d["boundary_vertices"]==i for d in details) for i in range(5)},"boundary_face_counts":{str(i):sum(d["boundary_faces"]==i for d in details) for i in range(5)},"details":details}
 print(role,out[role]["boundary_node_counts"],out[role]["boundary_face_counts"],flush=True)
dest=H/"quality-boundary-atlas-865-root.json";dest.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(str(dest),hashlib.sha256(dest.read_bytes()).hexdigest())

