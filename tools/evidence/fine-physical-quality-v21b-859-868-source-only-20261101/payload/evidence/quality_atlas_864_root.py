from pathlib import Path
import json,numpy as np,hashlib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
s=json.loads((H/"all-quality-859-qualified-root/quality-working-state.json").read_text())
out={}
for role in ("Skin","Fat"):
 with np.load(s[role+"_current"]["path"],allow_pickle=False) as z:
  print(role,"keys",z.files,flush=True)
  p=np.asarray(z["points_world_m_f32"] if "points_world_m_f32" in z else z["points_world_m"],np.float32)
  t=np.asarray(z["tetrahedra"],np.int64)
 x=p[t].astype(np.float64);a=x[:,1]-x[:,0];b=x[:,2]-x[:,0];c=x[:,3]-x[:,0]
 d=np.einsum("ij,ij->i",a,np.cross(b,c))
 e=np.zeros(len(t))
 for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
  e=np.maximum(e,np.linalg.norm(x[:,i]-x[:,j],axis=1))
 quality=d/e**3;bad=np.flatnonzero((d/6<=1e-18)|(quality<1e-5)|~np.isfinite(quality))
 bt=t[bad];nodes,inv=np.unique(bt,return_inverse=True)
 inc=csr_matrix((np.ones(bt.size,dtype=np.int8),(np.repeat(np.arange(len(bt)),4),inv.ravel())),shape=(len(bt),len(nodes)))
 n,labels=connected_components(inc@inc.T,directed=False)
 groups=[]
 for k in range(n):
  rows=bad[labels==k];v=t[rows]
  groups.append({"count":len(rows),"rows":rows.tolist(),"centroid_m":p[v].mean(axis=(0,1)).tolist(),"min_q":float(quality[rows].min()),"max_q":float(quality[rows].max()),"min_det6":float(d[rows].min()),"max_edge_m":float(e[rows].max())})
 groups.sort(key=lambda a:(-a["count"],a["rows"][0]))
 out[role]={"failures":len(bad),"volume_fail":int(np.count_nonzero(d[bad]/6<=1e-18)),"quality_fail":int(np.count_nonzero(quality[bad]<1e-5)),"bad_clusters":groups,"input":s[role+"_current"]}
 print(role,len(bad),"clusters",[(g["count"],g["centroid_m"]) for g in groups],flush=True)
dest=H/"quality-atlas-864-root.json";dest.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print("ATLAS",str(dest),hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)

