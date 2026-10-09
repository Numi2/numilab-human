#!/usr/bin/env python3
"""Build a raw-source-geometry Dirichlet ABI5 candidate and owner-preflight it."""
from __future__ import annotations
import hashlib, json, os, shutil, struct, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path("/Users/n/numi-human-retained-delivery-20261009/skin-source-binding-1202")
PROJECT=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project")
SRC=PROJECT/"src"
SOURCES=ROOT/"source-inputs"
ARTIFACT=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004")
REGISTRATION=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json")
RAW_SKIN=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-weight-heldout-20261004/inputs/base/bodyparts3d-myosim-skinned-shell.nhskin")
RAW_MANIFEST=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-weight-heldout-20261004/inputs/base/bodyparts3d-myosim-skinned-shell.manifest.json")
INPUTS=ROOT/"source-surface-binding-inputs.npz"
BONES=ROOT/"source-surface-binding-bones.json"
DIRICHLET_NPZ=ROOT/"dirichlet/bodyparts3d-skin-binding-solution.npz"
DIRICHLET_REPORT=ROOT/"dirichlet/dirichlet-reconstruction-report.json"
OUT=ROOT/"raw-source-dirichlet-preflight"
HUMAN_SRC=Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human")
PAYLOAD=OUT/"bodyparts3d-myosim-skinned-shell-dirichlet.nhskin"
PROOF=OUT/"bodyparts3d-skin-binding-solution.npz"
MANIFEST=OUT/"bodyparts3d-myosim-skinned-shell.manifest.json"
AUDIT_REPORT=OUT/"skin-source-payload-preflight.json"
WRAPPER_REPORT=OUT/"authoring-and-oracle-report.json"

PINS={
 str(INPUTS):"85ceca04ba63fc463a4515579719fe82ad03137c98891647b75e472461c96625",
 str(BONES):"4a7c19baa4f3e36d6422288e0c91d9e326d393b1a3493a2b2097f5ac6e4df42c",
 str(DIRICHLET_NPZ):"8ee440a0e346ba253c32cf693661b889c622908cd7355c13ef50ae1d9187aae8",
 str(DIRICHLET_REPORT):"a979ed813e20544f4f0b9e9e03784d652abb5440dd60939107c360e5ca978803",
 str(RAW_SKIN):"7d296e45ba497f56ac4380a64605b30a794432b945ac28bf7bebf91657cfc4a1",
 str(RAW_MANIFEST):"bd8711819f789d3ef192106801833337509fa22b9bb3533eb012145b0ecdc49e",
 str(REGISTRATION):"b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e",
}
CODE_PINS={
 str(HUMAN_SRC/"skin_source_payload_preflight.py"):"f1b53b1bab9283d8fd9a88920efb8100b446153da180847a2521a32a7c80a829",
 str(HUMAN_SRC/"model.py"):"caae0f9a64db518712b491780b697c32800b13dba4b5b60ad128b67f5794c0df",
 str(HUMAN_SRC/"skin_surface_binding.py"):"33039db6031150e20a1fc1f3bba7e3d594f608dc0adcfc4a6de368a79c4102e7",
 str(HUMAN_SRC/"skin_surface_audit.py"):"dbd72470bd2cfdaa398ba5d253393b755d84b5cdfd25b9dd1952bde1120d5f46",
}

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def write_json(p,v):
 tmp=p.with_name(p.name+".tmp")
 with tmp.open("xb") as f:
  f.write(json.dumps(v,indent=2,sort_keys=True,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)

started=datetime.now(timezone.utc).isoformat(); t0=time.monotonic()
if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
for p,h in {**PINS,**CODE_PINS}.items():
 actual=sha(Path(p))
 if actual!=h: raise RuntimeError(f"pinned input changed: {p}: {actual} != {h}")
OUT.mkdir()
shutil.copyfile(DIRICHLET_NPZ,PROOF)
raw=RAW_SKIN.read_bytes()
header=struct.Struct("<8s5I32s")
magic,abi,nb,nv,ni,fp,source_hash=header.unpack_from(raw)
if magic!=b"NHSKIN1\0" or abi!=5: raise ValueError("unexpected raw payload ABI")
vertex_dtype=__import__("numpy").dtype([
 ("position","<f4",(3,)),("normal","<f4",(3,)),
 ("influence_indices","<u4",(4,)),("influence_weights","<f4",(4,))])
if vertex_dtype.itemsize!=56: raise AssertionError(vertex_dtype.itemsize)
binding_off=60;vertex_off=binding_off+36*nb;index_off=vertex_off+56*nv;weight_off=index_off+4*ni
if len(raw)!=weight_off+4*nv*nb: raise ValueError("raw payload byte-length mismatch")
if nv!=54949 or nb!=86: raise ValueError(f"unexpected payload dimensions {nv}x{nb}")

sys.path.insert(0,str(SRC))
sys.path.insert(0,str(SOURCES/"myosim"/"checkout"))
sys.path.insert(0,str(Path("/Users/n/numi-human-free-apex-two-family-1178/src")))
import numpy as np
from numilab_human.skin_surface_binding import source_surface_binding, DIRICHLET_METHOD
from numilab_human.skin_source_payload_preflight import audit, decode_payload
with np.load(INPUTS,allow_pickle=False) as z:
 vertices=np.asarray(z["vertices"]);faces=np.asarray(z["faces"])
 binding_count=int(z["binding_count"])
bones=json.loads(BONES.read_text())["bones"]
quartet,local,full,targets,gaps,evidence=source_surface_binding(vertices,faces,bones,binding_count,seed_condition="dirichlet")
with np.load(DIRICHLET_NPZ,allow_pickle=False) as z:
 saved_full=np.asarray(z["full_weights"]);saved_targets=np.asarray(z["seed_targets"]);saved_gaps=np.asarray(z["seed_projection_gaps_m"])
if evidence["method"]!=DIRICHLET_METHOD or not np.array_equal(saved_full,full) or not np.array_equal(saved_targets,targets) or not np.array_equal(saved_gaps,gaps):
 raise RuntimeError("candidate generator output differs from frozen Dirichlet solution")

payload=bytearray(raw)
records=np.frombuffer(payload,dtype=vertex_dtype,count=nv,offset=vertex_off)
records["influence_indices"][:]=np.asarray(quartet,dtype="<u4")
records["influence_weights"][:]=np.asarray(local,dtype="<f4")
full32=np.asarray(full,dtype="<f4")
np.frombuffer(payload,dtype="<f4",count=nv*nb,offset=weight_off)[:]=full32.ravel(order="C")
with PAYLOAD.open("xb") as f:f.write(payload)

manifest=json.loads(RAW_MANIFEST.read_text())
manifest["payload"]["file"]=PAYLOAD.name
manifest["payload"]["sha256"]=sha(PAYLOAD)
manifest["payload"]["bytes"]=PAYLOAD.stat().st_size
manifest["payload"]["binding_count"]=nb
manifest["payload"]["vertex_count"]=nv
manifest["payload"]["index_count"]=ni
manifest["payload"]["triangle_count"]=ni//3
manifest["coverage"]["binding_solution"]["file"]=PROOF.name
manifest["coverage"]["binding_solution"]["sha256"]=sha(PROOF)
manifest["coverage"]["binding_solution"]["bytes"]=PROOF.stat().st_size
manifest["coverage"]["binding_solution"]["full_weight_shape"]=list(full.shape)
manifest["coverage"]["source_surface_binding"]=evidence
manifest["coverage"]["maximum_float32_partition_error"]=float(np.max(np.abs(full32.sum(axis=1)-1)))
manifest["coverage"]["maximum_float32_weight_quantization_error"]=float(np.max(np.abs(full32.astype("<f8")-full)))
manifest["coverage"]["distinct_influence_quartets"]=int(np.unique(quartet,axis=0).shape[0])
write_json(MANIFEST,manifest)

decoded=decode_payload(PAYLOAD.read_bytes())
if (not np.array_equal(decoded["indices"],np.frombuffer(raw,dtype="<u4",count=ni,offset=index_off))
    or not np.array_equal(decoded["vertices_f"][:,:6],np.frombuffer(raw,dtype="<f4",count=14*nv,offset=vertex_off).reshape(nv,14)[:,:6])
    or not np.array_equal(decoded["bindings_f"],np.frombuffer(raw,dtype="<f4",count=9*nb,offset=binding_off).reshape(nb,9))):
 raise RuntimeError("raw geometry/topology/normal/binding bytes changed while authoring weights")

# This calls the real source-only owner audit. It independently rebuilds the
# source graph, runtime anchors, projections, and seed targets/gaps from archives.
result=audit(SOURCES,ARTIFACT,REGISTRATION,PAYLOAD,MANIFEST)
write_json(AUDIT_REPORT,result)
report={
 "schema":"numi.human.raw-source-dirichlet-payload-authoring.v1",
 "status":"passed_independent_source_oracle_preflight",
 "qualification":"raw source visual-shell Dirichlet counterfactual only; no current candidate geometry or posed/native clearance claim",
 "started_utc":started,"completed_utc":datetime.now(timezone.utc).isoformat(),"elapsed_wall_s":time.monotonic()-t0,
 "pinned_inputs":{p:{"sha256":h,"actual_sha256":sha(Path(p))} for p,h in PINS.items()},
 "pinned_audit_code":{p:{"sha256":h,"actual_sha256":sha(Path(p))} for p,h in CODE_PINS.items()},
 "payload":{"path":str(PAYLOAD),"sha256":sha(PAYLOAD),"bytes":PAYLOAD.stat().st_size,"raw_geometry_binding_topology_preserved":True,"patched_fields":["per-vertex top-four diagnostic influence indices","per-vertex top-four diagnostic influence weights","complete ABI5 float32 weight matrix"]},
 "binding_solution":{"path":str(PROOF),"sha256":sha(PROOF),"method":evidence["method"],"full_weight_shape":list(full.shape),"target_rows_recomputed_by_owner_but_saved_targets_match":bool(np.array_equal(targets,saved_targets)),"projection_gaps_recomputed_by_owner_but_saved_gaps_match":bool(np.array_equal(gaps,saved_gaps))},
 "manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"source_manifest_sha256":PINS[str(RAW_MANIFEST)]},
 "independent_preflight":{"path":str(AUDIT_REPORT),"sha256":sha(AUDIT_REPORT),"status":result["status"],"source_surface":result["source_surface"],"binding":result["binding"],"runtime":result["runtime"],"source_program_checks":result["source_program_checks"],"audit_code_sha256":result["audit_code_sha256"]},
}
write_json(WRAPPER_REPORT,report)
print(json.dumps({"status":report["status"],"audit_status":result["status"],"wall_s":report["elapsed_wall_s"],"payload_sha256":sha(PAYLOAD),"manifest_sha256":sha(MANIFEST),"audit_sha256":sha(AUDIT_REPORT),"report_sha256":sha(WRAPPER_REPORT)},sort_keys=True),flush=True)
