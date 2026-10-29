from pathlib import Path
import csv,hashlib,json,collections
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
A=P/"reduced-physical-boundary-400/attempt-044-minimal-coupled-bone32-buffer"
BASE=P/"reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz"
MAP=P/"conforming-fit-006-native-assets-545/run-002-operational-reference/registered-assets/volume/combined-node-map.csv"
OUT=A/"inner-half-skin-clipped-domain-002"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def faces(t):
 for k in range(4):
  f=[int(t[j]) for j in range(4) if j!=k]
  if k%2:f[1],f[2]=f[2],f[1]
  yield f
def mesh(p,t,phi,source_rows):
 # Original source vertices remain distinct. Only shared mesh edges share a cut point.
 cut_edges=sorted({tuple(sorted((int(x),int(y)))) for tt in t for i,x in enumerate(tt) for y in tt[i+1:] if phi[x]!=phi[y]})
 pp=list(map(list,p));edgeid={}
 for x,y in cut_edges:
  edgeid[(x,y)]=len(pp);pp.append(list((p[x]+p[y])*.5))
 def cut(x,y):return edgeid[tuple(sorted((int(x),int(y))))]
 pp=np.asarray(pp,np.float64);inc=collections.defaultdict(list)
 for ci,tt in enumerate(t):
  for f in faces(tt):inc[tuple(sorted(f))].append((ci,f))
 if any(len(v)>2 for v in inc.values()):raise ValueError("source tet boundary incidence")
 out=[];parents=[];kind=[]
 def emit(poly,ci,ty):
  if len(poly)<3:return
  # Stable fan, preserving polygon orientation. Shared faces use common edge IDs.
  j=min(range(len(poly)),key=lambda k:poly[k]);poly=poly[j:]+poly[:j]
  for k in range(1,len(poly)-1):
   out.append([poly[0],poly[k],poly[k+1]]);parents.append(int(source_rows[ci]));kind.append(ty)
 for key,v in sorted(inc.items()):
  if len(v)!=1:continue
  ci,f=v[0];poly=[]
  for j,x in enumerate(f):
   y=f[(j+1)%3]
   if phi[x]==0:poly.append(x)
   if phi[x]!=phi[y]:poly.append(cut(x,y))
  emit(poly,ci,0)
 for ci,tt in enumerate(t):
  lo=sorted(int(x) for x in tt if phi[x]==0);hi=sorted(int(x) for x in tt if phi[x]==1)
  if not lo or not hi:continue
  if len(lo)==1:poly=[cut(lo[0],x) for x in hi]
  elif len(hi)==1:poly=[cut(hi[0],x) for x in lo]
  else:poly=[cut(hi[0],lo[0]),cut(hi[0],lo[1]),cut(hi[1],lo[1]),cut(hi[1],lo[0])]
  xyz=pp[poly];n=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0])
  direction=p[hi].mean(axis=0)-p[lo].mean(axis=0)
  if np.dot(n,direction)<0:poly=poly[::-1]
  emit(poly,ci,1)
 F=np.asarray(out,np.int64);used=np.unique(F);re=np.full(len(pp),-1,np.int64);re[used]=np.arange(len(used))
 return pp[used],re[F],np.asarray(parents),np.asarray(kind),used,cut_edges
def verify(p,f):
 x=p[f];cr=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0])
 if np.any(np.all(cr==0,axis=1)):raise ValueError("degenerate F64 mask facet")
 e=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
 _,cnt=np.unique(e,axis=0,return_counts=True)
 if np.any(cnt!=2):raise ValueError("clipped mask is not closed edge-manifold")
 directed=collections.Counter(tuple(map(int,k)) for k in np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]))
 if any(directed[(b,a)]!=c for (a,b),c in directed.items()):raise ValueError("inconsistent directed edge winding")
 return float(np.einsum("ij,ij->i",x[:,0],np.cross(x[:,1],x[:,2])).sum()/6)
def main():
 if OUT.exists():raise FileExistsError(OUT)
 assert sha(BASE)=="888eb12450d0e69fb0ad3f5998159ab7a0a20d3e70b13c42ba416c61cbbede63"
 assert sha(MAP)=="b3bf0870519d81c3ff8c98d15ce61c107905beec21615da85505f2da2bd03bcf"
 # Analytic single-tet clipping checks: 1/2/3 outside corners leave 7/8, 1/2, 1/8.
 unit=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
 for n,fraction in [(1,7/8),(2,.5),(3,1/8)]:
  ph=np.zeros(4,int);ph[:n]=1
  a,b,_,_,_,_=mesh(unit,np.array([[0,1,2,3]]),ph,np.array([0]))
  assert abs(verify(a,b)-fraction/6)<1e-15
 z=np.load(BASE,allow_pickle=False);p=z["points_current_world_m_f32"].astype(np.float64);t=z["volume_tetrahedra"].astype(np.int64)
 rows=list(csv.DictReader(MAP.open()));role={}
 for r in rows:
  node=int(r["joint_node_id"])-1
  if node in role:assert role[node]==r["role"]
  role[node]=r["role"]
 sel=np.flatnonzero((z["material_index"]==15)&(z["geometric_donor_body"]==23)&(z["anatomical_region_id"]==23))
 assert all(role[int(k)] in ("O","I") for k in np.unique(t[sel]))
 phi=np.array([1 if role[k]=="O" else 0 for k in range(len(p))],int)
 a,f,parents,kind,used,edges=mesh(p,t[sel],phi,sel);volume=verify(a,f)
 x=p[t[sel]];v=np.einsum("ij,ij->i",x[:,1]-x[:,0],np.cross(x[:,2]-x[:,0],x[:,3]-x[:,0]))/6
 assert np.all(v>0)
 nout=phi[t[sel]].sum(axis=1);fractions=np.array([1.,.875,.5,.125,0.]);analytic=float(np.dot(v,fractions[nout]))
 assert abs(volume-analytic)<1e-15
 out_source_nodes=np.array([int(x) if x<len(p) else -1 for x in used])
 edge_ends=np.array([[-1,-1] if x<len(p) else edges[int(x)-len(p)] for x in used],np.int64)
 # Midpoints are source-parent barycentric constructions; this is an authoring mask only.
 outphi=np.array([phi[x] if x<len(p) else .5 for x in used])
 assert np.all(outphi<=.5)
 f32=a.astype(np.float32).astype(np.float64);tx=f32[f];zero=int(np.sum(np.all(np.cross(tx[:,1]-tx[:,0],tx[:,2]-tx[:,0])==0,axis=1)))
 OUT.mkdir()
 np.savez_compressed(OUT/"clipped-inner-skin-domain.npz",points_world_m_f64=a,faces_f64=f,face_source_tet_rows=parents,face_kind=kind,vertex_original_source_nodes=out_source_nodes,vertex_source_edge_nodes=edge_ends,vertex_layer_coordinate=outphi)
 j={"status":"SOURCE_ONLY_CLIPPED_INNER_SKIN_MASK_READY","construction":"Exact clipping of current006 positive Skin23 tets by O=1,I=0 barycentric layer coordinate <=0.5. It is only a buffer-donor eligibility mask; physical mesh and original mass are unchanged.","analytical_single_tet_cases":"PASS 1/2/3 O corners","points":len(a),"faces":len(f),"source_tets":len(sel),"closed_edge_manifold":True,"consistent_edge_winding":True,"volume_m3":volume,"analytic_cell_clip_volume_m3":analytic,"minimum_original_source_tet_volume_m3":float(v.min()),"f32_zero_facets":zero,"retained_original_O_vertices":int(np.sum(outphi==1)),"runtime_created":0,"accepted_steps":0,"physical_admission":False,"inputs":{str(x):sha(x) for x in [BASE,MAP,Path(__file__)]}}
 (OUT/"receipt.json").write_text(json.dumps(j,indent=2)+"\n");print(json.dumps(j))
if __name__=="__main__":main()
