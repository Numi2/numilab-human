from pathlib import Path
import hashlib,json,struct,sys
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009")
ROOT=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005")
N=ROOT/"output/current-anatomy-20261005/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
M=N.with_name("bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json")
REG=ROOT/"Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json"
ART=ROOT/"Build/skin-source-fit-recovery-20261004"
OUT=Path(__file__).resolve().parent
sys.path.insert(0,"/Users/n/numi-human-free-apex-two-family-1178/src")
from numilab_human import model

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def pin(p):return {"path":str(p),"sha256":sha(p),"bytes":Path(p).stat().st_size}
raw=N.read_bytes();_,abi,nr,nb,nv,ni,*_=struct.unpack_from("<8s6I32s",raw); rows=np.frombuffer(raw,dtype="<u4",count=nr*8,offset=64).reshape(-1,8);r=rows[rows[:,6]==64][0]
fb,bc,fv,nvv,fi,nidx,*_=map(int,r);bo=64+nr*32;vo=bo+nb*36
binds=np.frombuffer(raw,dtype=[("body","<u4"),("v","<f4",(8,))],count=bc,offset=bo+fb*36)
verts=np.frombuffer(raw,dtype=[("p","<f4",(3,)),("n","<f4",(3,)),("idx","<u4",(4,)),("w","<f4",(4,))],count=nvv,offset=vo+fv*56)
manifest=json.loads(M.read_text());surf=next(x for x in manifest["source"]["surfaces"] if x["stable_id"]==64);body_order=[x["myosim_body"] for x in surf["body_bindings"]]
reg=json.loads(REG.read_text());T=np.asarray(reg["coordinate_system"]["global_source_mm_to_myosim_world_m"],float)
_,routes,_=model._myosim_surface_route_context(ART,manifest["source"]["myosim_source_archive_sha256"]);points=routes["vaslat_l"]["route_points"]
def expected(p):
 x=T[:3,:3]@(np.asarray(p,dtype=float)*1000)+T[:3,3]
 d=[]
 for body in body_order:
  d.append(min(float(np.dot(x-np.asarray(n["world_m"],float),x-np.asarray(n["world_m"],float))) for n in points if n["body"]==body))
 raw=[1/(v+9e-6) for v in d];s=sum(raw);return np.asarray([v/s for v in raw])
slot_to_body=[int(x["body"]) for x in binds];body_to_order={int(x["core_body_index"]):j for j,x in enumerate(surf["body_bindings"])}
errors=[]
for v in verts:
 got=np.zeros(len(body_order),float)
 for slot,w in zip(v["idx"],v["w"],strict=True):
  if int(slot)!=0xffffffff and float(w)>0:got[body_to_order[slot_to_body[int(slot)]]]+=float(w)
 errors.append(np.abs(got-expected(v["p"])))
errors=np.asarray(errors)
candidate=np.load(OUT/"left-vastus-lateralis-one-edge-refinement.npz")
mid=candidate["positions"][-1]; midweights=candidate["weights"][-1]; mididx=candidate["indices"][-1]
midby=np.zeros(len(body_order),float)
for slot,w in zip(mididx,midweights,strict=True):
 if int(slot)!=0xffffffff and float(w)>0:midby[body_to_order[slot_to_body[int(slot)]]]+=float(w)
rep={"schema":"numi.human.vastus-lateralis-route-weight-refinement-check.v1","status":"complete","method":"Re-evaluate the existing nearest-per-body route-node inverse-square formula on the midpoint; independently compare that same formula against the entire existing NHTISS64 weight field after decoding lane indices.","existing_weight_formula_error":{"vertex_count":int(len(verts)),"max_abs_by_body":errors.max(axis=0).tolist(),"max_abs_any_body":float(errors.max()),"mean_abs_by_body":errors.mean(axis=0).tolist()},"midpoint":{"source_position_f32_m":mid.tolist(),"body_order":body_order,"recomputed_float64":expected(mid).tolist(),"candidate_stored_float32_by_body":midby.tolist(),"candidate_lanes_index":mididx.tolist(),"candidate_lanes_weight":midweights.tolist(),"max_abs_error_after_float32":float(np.max(np.abs(midby-expected(mid))))},"inputs":[pin(N),pin(M),pin(REG),pin(Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/model.py")),pin(OUT/"left-vastus-lateralis-one-edge-refinement.npz")],"script":pin(Path(__file__))}
p=OUT/"route-weight-validation.json";p.write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"report":str(p),"sha256":sha(p),"existing":rep["existing_weight_formula_error"],"midpoint":rep["midpoint"]},indent=2))
