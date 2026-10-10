#!/usr/bin/env python3
import hashlib,json,struct
from pathlib import Path
import numpy as np
BASE=Path("/Users/n/numi-human-retained-delivery-20261009/source-seam-connectivity-1247/regen-4rows-001")
CAND=BASE/"candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MANIFEST=CAND.with_suffix(".manifest.json")
PARENT=Path("/Users/n/numi-human-retained-delivery-20261009/passive-biceps-micro-overlap-1225/compose-current-1cd-attempt003/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue")
SELF=BASE/"regenerated-row-exact-f32-self-audit-001.json"
OUT=BASE/"fhl-row-patches-001"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if OUT.exists():raise SystemExit("refuse existing output directory")
raw=CAND.read_bytes();magic,abi,nr,nb,nv,ni,fp,source=struct.unpack_from("<8s6I32s",raw,0)
if magic!=b"NHTISS4\0" or abi!=5 or len(raw)!=64+32*nr+36*nb+56*nv+4*ni:raise SystemExit("NHTISS ABI/size mismatch")
vo=64+32*nr+36*nb;io=vo+56*nv
records=np.frombuffer(raw,dtype="<u4",count=nr*8,offset=64).reshape(nr,8)
manifest=json.loads(MANIFEST.read_text());mrows={int(r["stable_id"]):r for r in manifest["source"]["surfaces"]}
report=json.loads(SELF.read_text());selfrows={int(r["stable_id"]):r for r in report["regenerated_rows"]}
OUT.mkdir()
patches={}
for sid in (27,28):
 r=next(row for row in records if int(row[6])==sid)
 firstb,nbind,firstv,nvert,firsti,nidx,stable,layer=map(int,r)
 if stable!=sid or nidx%3 or layer!=1:raise SystemExit(f"bad FHL row {sid}")
 v6=np.ndarray((nvert,6),dtype="<f4",buffer=raw,offset=vo+firstv*56,strides=(56,4)).copy()
 bind=np.ndarray((nvert,4),dtype="<u4",buffer=raw,offset=vo+firstv*56+24,strides=(56,4)).copy()
 weights=np.ndarray((nvert,4),dtype="<f4",buffer=raw,offset=vo+firstv*56+40,strides=(56,4)).copy()
 ix=np.frombuffer(raw,dtype="<u4",count=nidx,offset=io+firsti*4).copy()
 if ix.size and (int(ix.min())<firstv or int(ix.max())>=firstv+nvert):raise SystemExit(f"row index range {sid}")
 faces=(ix-firstv).reshape(-1,3)
 path=OUT/f"row-{sid}.npz"
 np.savez(path,vertices6=v6,binding_indices=bind,weights=weights,faces=faces.astype("<u4"))
 with np.load(path,allow_pickle=False) as z:
  if not (z["vertices6"].tobytes()==v6.tobytes() and z["binding_indices"].tobytes()==bind.tobytes() and z["weights"].tobytes()==weights.tobytes() and z["faces"].tobytes()==faces.astype("<u4").tobytes()):
   raise SystemExit(f"NPZ byte roundtrip {sid}")
 patches[str(sid)]={"path":str(path),"sha256":sha(path),"vertices6_shape":list(v6.shape),"vertices6_dtype":str(v6.dtype),"binding_indices_shape":list(bind.shape),"weights_shape":list(weights.shape),"faces_shape":list(faces.shape),"layer":mrows[sid]["layer"],"member_id":mrows[sid]["member_id"],"label":mrows[sid]["label"],"body_bindings":mrows[sid]["body_bindings"],"exact_f32_self_unallowed_pairs":selfrows[sid]["exact_predicate"]["unallowed_self_intersection_pair_count"],"exact_f32_quotient_boundary_edges":selfrows[sid]["topology_after_exact_f32_coordinate_quotient"]["boundary_edges"]}
parentmanifest=PARENT.with_suffix(".manifest.json")
inputs={str(p):sha(p) for p in (CAND,MANIFEST,PARENT,parentmanifest,SELF)}
reportout={"schema":"numi.human.fhl-source-row-patch-preparation.v1","scope":"Extract only passing FHL muscle rows 27/28 into existing NHTISS row-patch array layout. EHL rows 23/24 are deliberately excluded because their regenerated exact-F32 self-pair counts worsened. Prepared patches do not compose, update a receipt, qualify anatomy, or prove native pose clearance.","parent_payload_path":str(PARENT),"parent_payload_sha256":sha(PARENT),"parent_manifest_path":str(parentmanifest),"parent_manifest_sha256":sha(parentmanifest),"source_subset_payload_path":str(CAND),"source_subset_payload_sha256":sha(CAND),"source_subset_manifest_path":str(MANIFEST),"source_subset_manifest_sha256":sha(MANIFEST),"self_audit_report_path":str(SELF),"self_audit_report_sha256":sha(SELF),"registration_fingerprint32":f"{fp:08x}","source_archive_sha256":source.hex(),"rows":patches,"input_sha256":inputs}
out=OUT/"row-patch-preparation.json";out.write_text(json.dumps(reportout,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(out),"report_sha256":sha(out),"patches":patches},indent=2))
