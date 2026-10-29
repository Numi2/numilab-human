#!/usr/bin/env python3
from pathlib import Path
import collections,gc,hashlib,importlib.util,json,sys,time
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
G=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype"
E=G/"pair-geometry-014-linked-remesh"
OUT=Path(__file__).resolve().parent
C=E/"closed603-merge-020/candidate-f32-linked-remesh-typed.npz"
BASE=G/"candidate-f32.npz"
REPLAY=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz"
MAN=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json"
HELP=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(n,path):
 s=importlib.util.spec_from_file_location(n,path);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
started=time.monotonic();pred=load("remesh608",HELP).load_exact_owner()
assert sha(C)=="0911a3012df9f0776b54fd9276c38207383a9853d3d47c67c301a9e7b7774459"
assert sha(BASE)=="e8e332fb4ab421db6e44f0f5fe638e378646fd63438a318dcf94ad84b7a8ce96"
with np.load(BASE,allow_pickle=False) as z:oldids=set(map(int,z["face_source_original_id"]))
with np.load(C,allow_pickle=False) as z:
 points=z["points_world_m"].astype(np.float32);faces=z["faces"].astype(np.int64)
 sources=z["face_source_original_id"].astype(np.int64)
newrows=np.flatnonzero(~np.isin(sources,np.asarray(sorted(oldids),np.int64)))
newrecs,_=pred.float32_surface_records(points.astype(np.float64),faces[newrows],source_seam_groups=())
newrecs=[(r[0],r[1],r[2],1000000000000+j,r[4])for j,r in enumerate(newrecs)]
def tx(s):
 if int(s)==62050000:return 270
 if int(s)==62050001:return 309
 return (int(s)-62010000)//100
sourcepins={"candidate":sha(C),"baseline":sha(BASE),"native_replay":sha(REPLAY),"f64_bank_manifest":sha(MAN),"exact_owner":sha(pred.__file__),"script":sha(__file__)}
report={"schema":"changed-face-exact-all185-bank-interface.v1","scope":"new replacement faces only; does not certify unchanged faces, closed-solid containment, nesting, physical volume or motion","pins":sourcepins,"new_face_rows":newrows.tolist(),"new_face_count":len(newrows),"runs":[],"physical_admission":False}
def execute(mode,bankrecs,bankranges):
 t=time.monotonic();index=pred._prepare_surface_aabb(bankrecs)
 result=pred._audit_pair_prepared_first(index,newrecs,same_surface=False)
 pairs=np.asarray(result["triangle_pairs"],np.int64).reshape(-1,2)
 np.save(OUT/(mode+"-pairs.npy"),pairs,allow_pickle=False)
 bankends=np.asarray([r["end"] for r in bankranges],np.int64)
 perbank=collections.Counter();pertx=collections.Counter();sample=[]
 for boneface,dface in pairs:
  ri=int(np.searchsorted(bankends,int(boneface),side="right"));bank=bankranges[ri]
  j=int(dface)-1000000000000;row=int(newrows[j]);sid=int(sources[row]);tid=tx(sid)
  perbank[(bank["body"],bank["bone"])]+=1;pertx[tid]+=1
  if len(sample)<64:sample.append({"body":bank["body"],"bone":bank["bone"],"bank_face_row":int(boneface)-bank["start"],"candidate_face_row":row,"source_id":sid,"transaction_id":tid})
 out={"mode":mode,"bank_count":len(bankranges),"bone_faces":len(bankrecs),"exact_interface_pair_count":int(result["count"]),"aabb_candidate_pairs":int(result["aabb_candidate_pairs"]),"pairs_sha256":sha(OUT/(mode+"-pairs.npy")),"pairs_archive":str(OUT/(mode+"-pairs.npy")),"per_bank":[{"body":a,"bone":b,"pairs":n}for(a,b),n in sorted(perbank.items())],"per_transaction":[{"transaction_id":i,"pairs":n}for i,n in sorted(pertx.items())],"witness_sample":sample,"seconds":time.monotonic()-t}
 report["runs"].append(out);report["elapsed_s"]=time.monotonic()-started
 (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({"mode":mode,"pairs":int(result["count"]),"transactions":len(pertx),"seconds":out["seconds"]}),flush=True)
with np.load(REPLAY,allow_pickle=False) as z:
 bpoints=z["points_world_f32"].astype(np.float64);bf=z["faces"].astype(np.int64)
 offs=z["bank_face_offsets"];body=z["bank_body"];bone=z["bank_bone"]
 ranges=[{"body":int(body[i]),"bone":int(bone[i]),"start":int(offs[i]),"end":int(offs[i+1])}for i in range(185)]
brec,_=pred.float32_surface_records(bpoints,bf,source_seam_groups=())
execute("source-order-native-f32-replay",brec,ranges)
del brec,bpoints,bf;gc.collect()
# Pinned Float64 registration operands are represented without rounding on
# the SAME exact 2^149 integer lattice as query geometry. Reject any input
# whose denominator does not divide that lattice instead of truncating.
denominator=1<<149
def key(p):
 out=[]
 for x in p:
  n,d=float(x).as_integer_ratio()
  if denominator%d:raise ValueError("Float64 coordinate cannot share this exact lattice")
  out.append(n*(denominator//d))
 return tuple(out)
manifest=json.loads(MAN.read_text());records=[];ranges=[];nextvertex=0
for b in manifest["operands"]:
 f=Path(b["operand_path"])
 if sha(f)!=b["operand_sha256"]:raise ValueError("pinned operand changed")
 with np.load(f,allow_pickle=False) as z:
  xyz=z["points_world_m_f64"];ff=z["faces_f64"]
 start=len(records);ks=tuple(key(p)for p in xyz)
 raw=pred._records(ks,ff)
 records.extend((r[0],r[1],r[2],start+r[3],tuple(nextvertex+v for v in r[4]))for r in raw)
 ranges.append({"body":int(b["body"]),"bone":int(b["bone"]),"start":start,"end":len(records)})
 nextvertex+=len(xyz)
execute("pinned-f64-exact",records,ranges)
report["elapsed_s"]=time.monotonic()-started
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
