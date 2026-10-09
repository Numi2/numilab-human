#!/usr/bin/env python3
"""Create an offline NHSKIN ABI5 candidate from the independently verified Dirichlet matrix."""
from __future__ import annotations
import hashlib, json, struct, sys
from pathlib import Path
import numpy as np
ROOT=Path("/Users/n/numi-human-retained-delivery-20261009/skin-source-binding-1202")
BASE=Path("/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin")
BASE_SHA="b2d235e32c1c7d7f753eb83d1e8e9d045a1fd62be9c5c6da65dfde8844e2622b"
NPZ=ROOT/"dirichlet/bodyparts3d-skin-binding-solution.npz"; NPZ_SHA="8ee440a0e346ba253c32cf693661b889c622908cd7355c13ef50ae1d9187aae8"
DIR_REPORT=ROOT/"dirichlet/dirichlet-reconstruction-report.json"; DIR_REPORT_SHA="a979ed813e20544f4f0b9e9e03784d652abb5440dd60939107c360e5ca978803"
GRAPH=ROOT/"source-surface-binding-inputs.npz"; GRAPH_SHA="85ceca04ba63fc463a4515579719fe82ad03137c98891647b75e472461c96625"
BONES=ROOT/"source-surface-binding-bones.json"; BONES_SHA="4a7c19baa4f3e36d6422288e0c91d9e326d393b1a3493a2b2097f5ac6e4df42c"
AUDIT=Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/skin_surface_audit.py"); AUDIT_SHA="dbd72470bd2cfdaa398ba5d253393b755d84b5cdfd25b9dd1952bde1120d5f46"
OUT=ROOT/"counterfactual-screen-input/candidate-dirichlet.nhskin"; OUT_REPORT=ROOT/"counterfactual-screen-input/candidate-build-report.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
for p,h in ((BASE,BASE_SHA),(NPZ,NPZ_SHA),(DIR_REPORT,DIR_REPORT_SHA),(GRAPH,GRAPH_SHA),(BONES,BONES_SHA),(AUDIT,AUDIT_SHA)):
 require(Path(p).is_file() and not Path(p).is_symlink() and sha(p)==h,f"input hash mismatch: {p}")
r=json.loads(DIR_REPORT.read_text())
require(r.get("status")=="dirichlet_verified_source_candidate_differs_from_existing_payloads" and r.get("independent_verifier",{}).get("certificate",{}).get("passed") is True and r.get("checks",{}).get("independent_verifier_full_solution_exact") is True,"source report does not prove verified differing solution")
sys.path.insert(0,"/Users/n/numi-human-free-apex-two-family-1178/src")
from numilab_human.skin_surface_audit import _verified_weights
from numilab_human.skin_surface_binding import DIRICHLET_METHOD
with np.load(GRAPH,allow_pickle=False) as z:
 vertices=np.asarray(z["vertices"],dtype="<f8"); faces=np.asarray(z["faces"],dtype="<i8"); nb=int(z["binding_count"])
with np.load(NPZ,allow_pickle=False) as z:
 full64=np.asarray(z["full_weights"],dtype="<f8"); targets=np.asarray(z["seed_targets"],dtype="<u4"); gaps=np.asarray(z["seed_projection_gaps_m"],dtype="<f8")
q,local,vfull,cert=_verified_weights(NPZ.read_bytes(),vertices.tobytes(),np.asarray(faces,dtype="<u4").tobytes(),targets.tobytes(),gaps.tobytes(),nb,DIRICHLET_METHOD)
require(cert.get("passed") is True and np.array_equal(vfull,full64),"independent verifier replay failed")
require(q.shape==(54949,4) and local.shape==(54949,4) and np.isfinite(vfull).all() and np.min(vfull)>=0 and np.max(np.abs(vfull.sum(axis=1)-1))<1e-12,"invalid full matrix or diagnostics")
base=BASE.read_bytes(); H=struct.Struct("<8s5I32s")
magic,abi,nb0,nv,ni,fp,sh=H.unpack_from(base)
require(magic==b"NHSKIN1\0" and abi==5 and nb0==nb and nv==54949,"base header mismatch")
vo=H.size+nb*36; io=vo+nv*56; fo=io+ni*4
require(fo+nv*nb*4==len(base),"base payload length mismatch")
out=bytearray(base); q=np.asarray(q,dtype="<u4"); local=np.asarray(local,dtype="<f4"); full32=np.asarray(vfull,dtype="<f4")
for i in range(nv):
 o=vo+i*56+24; struct.pack_into("<4I",out,o,*map(int,q[i])); struct.pack_into("<4f",out,o+16,*map(float,local[i]))
out[fo:]=full32.tobytes()
require(out[:H.size]==base[:H.size] and out[H.size:vo]==base[H.size:vo] and out[io:fo]==base[io:fo],"header/bindings/indices changed")
for i in range(nv): require(out[vo+i*56:vo+i*56+24]==base[vo+i*56:vo+i*56+24],f"position/normal changed at {i}")
for i in range(nv):
 o=vo+i*56+24
 require(np.array_equal(np.asarray(struct.unpack_from("<4I",out,o),dtype="<u4"),q[i]) and np.array_equal(np.asarray(struct.unpack_from("<4f",out,o+16),dtype="<f4"),local[i]),f"diagnostics mismatch at {i}")
require(np.array_equal(np.frombuffer(out,dtype="<f4",count=nv*nb,offset=fo).reshape(nv,nb),full32),"full matrix mismatch")
OUT.parent.mkdir(exist_ok=True); require(not OUT.exists() and not OUT_REPORT.exists(),"refusing overwrite")
OUT.write_bytes(out)
result={"schema":"numi.human.skin-dirichlet-counterfactual-candidate.v1","status":"source_verified_offline_candidate_not_native_admitted","qualification":"Only ABI5 full weights and four-weight diagnostic fields changed. Position/normal, binding, indices, topology and header are byte-identical. Source-derived inferred field, not measured skin mechanics or native result.","inputs":{"base_1187":{"path":str(BASE),"sha256":BASE_SHA},"dirichlet_npz":{"path":str(NPZ),"sha256":NPZ_SHA},"dirichlet_report":{"path":str(DIR_REPORT),"sha256":DIR_REPORT_SHA},"source_graph":{"path":str(GRAPH),"sha256":GRAPH_SHA},"source_bones":{"path":str(BONES),"sha256":BONES_SHA},"audit_module":{"path":str(AUDIT),"sha256":AUDIT_SHA}},"verifier":{"method":DIRICHLET_METHOD,"certificate":cert,"full_solution_equal":True,"quartets_equal":True},"dimensions":{"binding_count":nb,"vertex_count":nv,"triangle_count":ni//3,"full_weights_f32_sha256":hashlib.sha256(full32.tobytes()).hexdigest()},"byte_identity":{"header":True,"binding_records":True,"position_normal_slots":True,"triangle_indices":True,"changed_regions":["four-weight diagnostic fields","full F32 matrix"]},"output":{"path":str(OUT),"bytes":len(out),"sha256":hashlib.sha256(out).hexdigest()}}
OUT_REPORT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"status":result["status"],"candidate":str(OUT),"sha256":result["output"]["sha256"],"bytes":len(out)},sort_keys=True))
