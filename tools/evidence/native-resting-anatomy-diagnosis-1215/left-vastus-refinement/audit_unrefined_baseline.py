from pathlib import Path
import hashlib,json,struct,sys
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009")
N=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/current-anatomy-20261005/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue")
CAPS=[(R/"native-lung1178-thumb1187-smoke-1191/native-run",0),(R/"native-lung1178-thumb1187-smoke-1191/native-run",10000),(R/"native-flat-reference-40s-1201/native-run",20000)]
sys.path.insert(0,"/Users/n/numi-human-free-apex-two-family-1178/src")
from numilab_human import common_atlas_skin_clearance as cl, cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def pin(p):return {"path":str(p),"sha256":sha(p),"bytes":Path(p).stat().st_size}
def q(v,f):
 d={}; vv=[]; ids=np.empty(len(v),np.int64)
 for i,p in enumerate(v):
  k=np.asarray(p,dtype="<f4").tobytes()
  if k not in d:d[k]=len(vv);vv.append(np.asarray(p,dtype="<f4").copy())
  ids[i]=d[k]
 return np.asarray(vv,dtype="<f4"),ids[np.asarray(f,dtype=np.int64)]
def audit(v,f,label):
 qv,qf=q(v,f);keys=[ci.float32_point_lattice_key(p) for p in qv];records=ci._records(keys,qf.tolist())
 t=analyze_topology(qv.astype(float).tolist(),qf.tolist())
 x=ci._audit_pair(records,records,same_surface=True)
 return {"label":label,"vertices":len(v),"faces":len(f),"quotient_vertices":len(qv),"topology_summary":{k:v for k,v in t.items() if not isinstance(v,(list,dict))},"topology_defect_counts":{k:len(v) for k,v in t.items() if isinstance(v,list)},"exact_self_intersections":x}
raw=N.read_bytes();_,abi,nr,nb,nv,ni,*_=struct.unpack_from("<8s6I32s",raw);recs=np.frombuffer(raw,dtype="<u4",count=nr*8,offset=64).reshape(-1,8);r=recs[recs[:,6]==64][0]
fb,bc,fv,nvv,fi,nidx,*_=map(int,r);voff=64+nr*32+nb*36
verts=np.frombuffer(raw,dtype=[("p","<f4",(3,)),("n","<f4",(3,)),("idx","<u4",(4,)),("w","<f4",(4,))],count=nvv,offset=voff+fv*56)
ioff=64+nr*32+nb*36+nv*56
faces=np.frombuffer(raw,dtype="<u4",count=nidx,offset=ioff+fi*4).reshape(-1,3).astype(np.int64)-fv
rows=[audit(verts["p"],faces,"raw-source-NHTISS-F32" )]
for run,step in CAPS:
 pack=run/"accepted-geometry"/f"step-{step}.mrvpack"; pos,surfs,_=cl._pack_surfaces(pack,{(51005,64)})
 capf=np.asarray(surfs[(51005,64)]["faces"],dtype=np.int64); local_to_global=np.full(len(verts),-1,np.int64)
 for sf,cf in zip(faces,capf,strict=True):
  for li,gi in zip(sf,cf,strict=True):
   li=int(li);gi=int(gi)
   if local_to_global[li] not in (-1,gi):raise RuntimeError("inconsistent face-to-pack map")
   local_to_global[li]=gi
 if np.any(local_to_global<0):raise RuntimeError("unused local vertices")
 rows.append(audit(np.asarray(pos,dtype="<f4")[local_to_global],faces,f"actual-native-step-{step}"))
report={"schema":"numi.human.vastus-lateralis-unrefined-baseline-exact-self-audit.v1","status":"complete","scope":"Same exact coordinate quotient and predicate as the one-edge refined candidate; no repair, tolerance, or cross-surface audit.","source":pin(N),"candidate_report":pin(Path(__file__).with_name("report.json")),"predicate":pin(Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/cardiac_cavity_intersections.py")),"captures":[pin(run/"accepted-geometry"/f"step-{step}.mrvpack")|{"step":step,"receipt":pin(run/"accepted-geometry"/f"step-{step}.receipt.json")} for run,step in CAPS],"audits":rows,"script":pin(Path(__file__))}
out=Path(__file__).with_name("unrefined-baseline-report.json");out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"report":str(out),"sha256":sha(out),"audits":[{"label":r["label"],"self_count":r["exact_self_intersections"]["count"],"candidate_pairs":r["exact_self_intersections"]["aabb_candidate_pairs"],"allowed_shared":r["exact_self_intersections"]["allowed_shared_vertex_or_edge_pairs"],"topology":r["topology_summary"]} for r in rows]},indent=2))
