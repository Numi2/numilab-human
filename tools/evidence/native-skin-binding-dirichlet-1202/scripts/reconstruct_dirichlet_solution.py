#!/usr/bin/env python3
"""Run and independently verify the opt-in Dirichlet source-skin field."""
from __future__ import annotations
import hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path("/Users/n/numi-human-retained-delivery-20261009/skin-source-binding-1202")
INPUT=ROOT/"source-surface-binding-inputs.npz"
BONES=ROOT/"source-surface-binding-bones.json"
DEFAULT=ROOT/"reconstruction/bodyparts3d-skin-binding-solution.npz"
DEFAULT_REPORT=ROOT/"reconstruction-report.json"
CURRENT_BINDING=Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/skin_surface_binding.py")
AUDIT=Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/skin_surface_audit.py")
RAW_SKIN=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-weight-heldout-20261004/inputs/base/bodyparts3d-myosim-skinned-shell.nhskin")
CURRENT_SKIN=Path("/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin")
RAW_MANIFEST=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-weight-heldout-20261004/inputs/base/bodyparts3d-myosim-skinned-shell.manifest.json")
OUT=ROOT/"dirichlet"
NPZ=OUT/"bodyparts3d-skin-binding-solution.npz"
REPORT=OUT/"dirichlet-reconstruction-report.json"

EXPECTED={
 str(INPUT):"85ceca04ba63fc463a4515579719fe82ad03137c98891647b75e472461c96625",
 str(BONES):"4a7c19baa4f3e36d6422288e0c91d9e326d393b1a3493a2b2097f5ac6e4df42c",
 str(DEFAULT):"8abd7ac49732bc81656434e99da8d0746d71831163778e826e58b4bb30ac7a49",
 str(DEFAULT_REPORT):"07a8d3d04fb4361e0cab40ea228c5ca95184ca7900595586d5f567ef81dd92b6",
 str(CURRENT_BINDING):"33039db6031150e20a1fc1f3bba7e3d594f608dc0adcfc4a6de368a79c4102e7",
 str(AUDIT):"dbd72470bd2cfdaa398ba5d253393b755d84b5cdfd25b9dd1952bde1120d5f46",
 str(RAW_SKIN):"7d296e45ba497f56ac4380a64605b30a794432b945ac28bf7bebf91657cfc4a1",
 str(CURRENT_SKIN):"b2d235e32c1c7d7f753eb83d1e8e9d045a1fd62be9c5c6da65dfde8844e2622b",
 str(RAW_MANIFEST):"bd8711819f789d3ef192106801833337509fa22b9bb3533eb012145b0ecdc49e",
}

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(path,obj):
 tmp=path.with_name(path.name+".tmp")
 with tmp.open("xb") as f:
  f.write(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

start=datetime.now(timezone.utc).isoformat(); t0=time.monotonic()
if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
for p,h in EXPECTED.items():
 path=Path(p)
 if not path.is_file(): raise FileNotFoundError(path)
 actual=digest(path)
 if actual!=h: raise RuntimeError(f"input changed: {path}: {actual} != {h}")
OUT.mkdir()
sys.path.insert(0,"/Users/n/numi-human-free-apex-two-family-1178/src")
import numpy as np
import scipy
from numilab_human.skin_surface_binding import source_surface_binding, DIRICHLET_METHOD
from numilab_human.skin_surface_audit import _verified_weights
with np.load(INPUT,allow_pickle=False) as z:
 vertices=np.asarray(z["vertices"])
 faces=np.asarray(z["faces"])
 binding_count=int(z["binding_count"])
bones=json.loads(BONES.read_text())["bones"]
with np.load(DEFAULT,allow_pickle=False) as z:
 default_full=np.asarray(z["full_weights"])
 baseline_targets=np.asarray(z["seed_targets"])
 baseline_gaps=np.asarray(z["seed_projection_gaps_m"])
quartet,local,full,targets,gaps,evidence=source_surface_binding(
 vertices,faces,bones,binding_count,seed_condition="dirichlet")
targets=np.asarray(targets,dtype="<u4");gaps=np.asarray(gaps,dtype="<f8");full=np.asarray(full,dtype="<f8")
if not np.array_equal(targets,baseline_targets): raise RuntimeError("Dirichlet seed target rows drifted from default replay")
if not np.array_equal(gaps,baseline_gaps): raise RuntimeError("Dirichlet projection gaps drifted from default replay")
np.savez_compressed(NPZ,full_weights=np.asarray(full,dtype="<f8"),seed_targets=targets,seed_projection_gaps_m=gaps)
proof=NPZ.read_bytes()
verified_quartet,verified_local,verified_full,certificate=_verified_weights(
 proof,
 np.asarray(vertices,dtype="<f8").tobytes(order="C"),
 np.asarray(faces,dtype="<u4").tobytes(order="C"),
 targets.tobytes(order="C"),gaps.tobytes(order="C"),binding_count,DIRICHLET_METHOD)
if not (np.array_equal(verified_full,full) and np.array_equal(verified_quartet,quartet)
        and np.array_equal(verified_local,local)):
 raise RuntimeError("independent _verified_weights result differs from Dirichlet owner output")

def read_weights(path):
 data=path.read_bytes();head=__import__("struct").Struct("<8s5I32s")
 magic,abi,nb,nv,ni,fp,source=head.unpack_from(data)
 off=head.size+nb*__import__("struct").calcsize("<I8f")+nv*56+ni*4
 w=np.frombuffer(data,dtype="<f4",count=nv*nb,offset=off).reshape(nv,nb)
 return w,data
raw_w,raw_bytes=read_weights(RAW_SKIN);cur_w,cur_bytes=read_weights(CURRENT_SKIN)
f32=np.asarray(full,dtype="<f4").tobytes(order="C")
raw_match=f32==raw_w.tobytes(order="C");cur_match=f32==cur_w.tobytes(order="C")
def diff(a,b):
 d=np.asarray(a,dtype="<f8")-np.asarray(b,dtype="<f8")
 return {"changed_rows":int(np.count_nonzero(np.any(np.asarray(a)!=np.asarray(b),axis=1))),
         "max_abs_weight_delta":float(np.max(np.abs(d))),
         "mean_abs_weight_delta":float(np.mean(np.abs(d))),
         "l1_delta_max":float(np.max(np.sum(np.abs(d),axis=1)))}
report={
 "schema":"numi.human.source-skin-dirichlet-reconstruction.v1",
 "status":"dirichlet_independently_verified" if raw_match and cur_match else "dirichlet_verified_source_candidate_differs_from_existing_payloads",
 "qualification":"offline inferred source-surface Dirichlet weight field; not measured skin mechanics or native qualification",
 "started_utc":start,"completed_utc":datetime.now(timezone.utc).isoformat(),"elapsed_wall_s":time.monotonic()-t0,
 "python":sys.version,"numpy_version":np.__version__,"scipy_version":scipy.__version__,
 "source_binding_owner":{"path":str(CURRENT_BINDING),"sha256":digest(CURRENT_BINDING),"method":evidence["method"],"seed_condition":evidence["seed_condition"]},
 "independent_verifier":{"path":str(AUDIT),"sha256":digest(AUDIT),"method":DIRICHLET_METHOD,"certificate":certificate},
 "inputs":{"captured_source_graph_npz":{"path":str(INPUT),"sha256":digest(INPUT)},"captured_source_bones_json":{"path":str(BONES),"sha256":digest(BONES)},"default_reconstruction_report":{"path":str(DEFAULT_REPORT),"sha256":digest(DEFAULT_REPORT)},"raw_source_payload":{"path":str(RAW_SKIN),"sha256":digest(RAW_SKIN)},"current_1187_payload":{"path":str(CURRENT_SKIN),"sha256":digest(CURRENT_SKIN)},"raw_source_manifest":{"path":str(RAW_MANIFEST),"sha256":digest(RAW_MANIFEST)}},
 "solution":{"path":str(NPZ),"sha256":digest(NPZ),"bytes":NPZ.stat().st_size,"arrays":{"full_weights":{"shape":list(full.shape),"dtype":full.dtype.str},"seed_targets":{"shape":list(targets.shape),"dtype":targets.dtype.str},"seed_projection_gaps_m":{"shape":list(gaps.shape),"dtype":gaps.dtype.str}},"seed_targets_exact_default":True,"seed_projection_gaps_exact_default":True,"f32_full_weights_sha256":hashlib.sha256(np.asarray(full,dtype="<f4").tobytes()).hexdigest()},
 "checks":{"independent_verifier_full_solution_exact":bool(np.array_equal(verified_full,full)),"independent_verifier_quartets_exact":bool(np.array_equal(verified_quartet,quartet)),"independent_verifier_local_weights_exact":bool(np.array_equal(verified_local,local)),"raw_source_full_f32_exact":raw_match,"current_1187_full_f32_exact":cur_match,"vs_default_source_full_weight_delta":diff(full,default_full),"vs_raw_source_full_weight_delta":diff(full,raw_w),"vs_current_1187_full_weight_delta":diff(full,cur_w),"full_f32_sha256_raw":hashlib.sha256(raw_w.tobytes()).hexdigest(),"full_f32_sha256_current_1187":hashlib.sha256(cur_w.tobytes()).hexdigest()},
 "binding_evidence":evidence,
}
write_json(REPORT,report)
print(json.dumps({"status":report["status"],"wall_s":report["elapsed_wall_s"],"solution_sha256":digest(NPZ),"report":str(REPORT),"verified":report["checks"]["independent_verifier_full_solution_exact"],"raw_match":raw_match,"current_match":cur_match},sort_keys=True),flush=True)
