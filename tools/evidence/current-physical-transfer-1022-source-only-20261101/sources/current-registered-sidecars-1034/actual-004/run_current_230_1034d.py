from __future__ import annotations
import gc,hashlib,importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path

BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
TRANSFER=BASE.parent
PYTHON=Path("/Users/numi/numi-human-prep-venv-20261005/bin/python")
WRAPPER=TRANSFER/"run_geometry_serial.py"
CHECKPOINT=BASE/"current-native-1024-root/integration-checkpoint.json"
PLAN=BASE/"downstream-003-current-typed-228/current-typed-228-to-433-command-plan.json"
ADAPTER=BASE/"current-registered-sidecars-1034/export_current_registered_sidecars_1034.py"
CONTROLS=BASE/"current-registered-sidecars-1034/controls-receipt.json"
OUT=BASE/"current-registered-sidecars-1034/actual-004/registered-sidecars"
LAUNCH=Path(__file__).resolve()
EXECUTION=LAUNCH.parent/"execution.json"
HELPER_DIR=BASE/"downstream-003-current-typed-228/native-domain-217-source-005"
HELPER_PATH=HELPER_DIR/"export_child_semantic_labels_217.py"
HELPER_SHA="9cf388fd7a9a18e72c478b9e5d1f6601884bdcf46be6dacb2d04e13fa2b66115"
OWNER_PATH=BASE/"current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py"
OWNER_SHA="121e3bf44d90ba3d7b82ccaa40a26a92d1675c6d277a4409ae95f8c05ad2485d"
PARENT_RUN_MANIFEST=BASE/"downstream-003-current-typed-228/run-manifest-pre-export.json"
PARENT_RUN_MANIFEST_SHA="3e551dbb2f2d3d69ede9742d8a1af0d72c3aa64e9112f5654d8c052d2bbd16f5"
PARENT_MASS_SHA="c7823c42d64a2154a34c87cc0d4d202f0d5256f79217e74f5f62bcf1d5b31679"
LEGACY_228_RECEIPT=BASE/"downstream-003-current-typed-228/registered-sidecars-228/receipt.json"
LEGACY_228_RECEIPT_SHA="93053ef51df7ad6c222dcc4f32dfc6bfd91ac5426a6c909cb297a01764ed9b38"
PLAN_SHA="39f1e7980278fff89629ae4631289a0c5d4548fb75f11914ebadff545b41d80a"
WRAPPER_SHA="88ce1f98414e0125f0cb31a94354db40cfdaba2281f79a9040aa9872d19edf35"

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def require_pin(item,label):
 p=Path(item["path"])
 if not p.is_file(): raise FileNotFoundError(f"{label} missing: {p}")
 actual=sha(p)
 if item.get("sha256")!=actual: raise ValueError(f"{label} SHA mismatch {actual} != {item.get('sha256')}")
 return p,actual
def write_json(p,obj):
 p.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n",encoding="utf-8")

def import_and_validate_230_context(parent_path,poses_path,weights_path,plan):
 if sha(HELPER_PATH)!=HELPER_SHA: raise ValueError("217 helper dependency hash mismatch")
 if sha(OWNER_PATH)!=OWNER_SHA: raise ValueError("pinned 230 owner hash mismatch")
 if sha(PARENT_RUN_MANIFEST)!=PARENT_RUN_MANIFEST_SHA: raise ValueError("228 source330 parent manifest hash mismatch")
 if sha(LEGACY_228_RECEIPT)!=LEGACY_228_RECEIPT_SHA: raise ValueError("228 source330 parent receipt hash mismatch")
 old=json.loads(PARENT_RUN_MANIFEST.read_text())
 inputs=old.get("inputs",{})
 legacy=json.loads(LEGACY_228_RECEIPT.read_text())
 parent_receipt=legacy.get("parentCurrent",{})
 paths={"mass":parent_path,"nodes":Path(str(parent_path)+".nodes.csv"),"source":Path(str(parent_path)+".source.csv"),"tets":Path(str(parent_path)+".tets.csv"),"labels":Path(str(parent_path)+".labels.csv"),"weights":weights_path}
 legacy_keys={"mass":("massPath","massSHA256"),"nodes":("nodesPath","nodeSHA256"),"source":("sourcePath","sourceSHA256"),"tets":("tetsPath","tetsSHA256"),"labels":("labelsPath","labelsSHA256")}
 for label,path in paths.items():
  actual=sha(path)
  if label in legacy_keys:
   path_key,sha_key=legacy_keys[label]
   if parent_receipt.get(path_key)!=str(path) or parent_receipt.get(sha_key)!=actual: raise ValueError(f"source330 {label} differs from pinned 228 parent receipt")
  else:
   weight_receipt=legacy.get("parentFull86Weights",{})
   if weight_receipt.get("path")!=str(path) or weight_receipt.get("sha256")!=actual: raise ValueError("source330 full86 weights differ from pinned 228 parent receipt")
  if label!="source" and inputs.get(str(path))!=actual: raise ValueError(f"source330 {label} input differs from pinned 228 manifest")
 parent_plan=plan.get("parent",{})
 checks=(("capture_base",parent_path),("poses",poses_path))
 for key,path in checks:
  if str(path)!=parent_plan.get(key): raise ValueError(f"source330 {key} path differs from 228 command plan")
 for key,path in (("nodes_sha256",paths["nodes"]),("labels_sha256",paths["labels"]),("tets_sha256",paths["tets"]),("full86_weights_sha256",paths["weights"]),("poses_sha256",poses_path)):
  if parent_plan.get(key)!=sha(path): raise ValueError(f"source330 {key} differs from 228 command plan")
 sys.path.insert(0,str(HELPER_DIR)); sys.path.insert(0,str(OWNER_PATH.parent))
 spec=importlib.util.spec_from_file_location("pinned_current_230_owner_context_1034c",OWNER_PATH)
 if spec is None or spec.loader is None: raise RuntimeError("cannot import pinned current 230 owner")
 owner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=owner;spec.loader.exec_module(owner)
 parent,index,material_indices,parentmeta=owner.load_parent_current(parent_path)
 frames,frame_meta,bodies=owner.load_frames(poses_path)
 weights,weight_meta=owner.load_parent_full86_weights(weights_path,parent,parentmeta)
 if weights.shape!=(len(parent["points"]),86) or len(bodies)!=len(frames) or not {15,16}.issubset(material_indices):
  raise ValueError("source330 parent/pose/full86 context shape mismatch")
 result={"status":"PASS_230_OWNER_AND_SOURCE330_CONTEXT","owner_path":str(OWNER_PATH),"owner_sha256":sha(OWNER_PATH),
  "helper_path":str(HELPER_PATH),"helper_sha256":sha(HELPER_PATH),"parent_path":str(parent_path),"parent_sha256":sha(parent_path),
  "parent_nodes":int(len(parent["points"])),"parent_tets":int(len(parent["tets"])),"materials":sorted(map(int,material_indices)),
  "pose_bindings":int(len(frames)),"full86_shape":list(weights.shape),"frame_method":frame_meta.get("frameSourceMethod","legacy_quaternion"),
  "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
 del weights,weight_meta,frames,frame_meta,bodies,parent,index,material_indices,parentmeta,owner
 gc.collect()
 return result

if not PYTHON.is_file(): raise FileNotFoundError(PYTHON)
for p,label in ((WRAPPER,"serial wrapper"),(CHECKPOINT,"integration checkpoint"),
                (PLAN,"parent command plan"),(ADAPTER,"current 230 exporter"),(CONTROLS,"1034 controls")):
 if not p.is_file(): raise FileNotFoundError(f"{label} missing: {p}")
if sha(WRAPPER)!=WRAPPER_SHA: raise ValueError("serial wrapper hash differs from pinned version")
if sha(PLAN)!=PLAN_SHA: raise ValueError("228 command plan hash differs from pinned version")
checkpoint_sha=sha(CHECKPOINT)
checkpoint=json.loads(CHECKPOINT.read_text())
proof_path,proof_sha=require_pin(checkpoint["current_source_proof"],"808 source proof")
state_path,state_sha=require_pin(checkpoint["current_volume_prefix"],"818c state")
shell_path,shell_sha=require_pin(checkpoint["paired_shell_receipt"],"1022 shell receipt")
binding_path,binding_sha=require_pin(checkpoint["paired_shell_validation"],"1022 shell binding")
handoff_path,handoff_sha=require_pin(checkpoint["217_handoff"],"938 handoff receipt")
crosswalk_path,crosswalk_sha=require_pin(checkpoint["registered_active_source_crosswalk"],"938 full crosswalk")
proof=json.loads(proof_path.read_text())
if proof.get("status")!="PASS_CURRENT_EXACT_SOURCE_EVIDENCE" or proof.get("physical_admission") is not False or proof.get("native_steps")!=0:
 raise ValueError("808 proof is not current source-only PASS")
if proof.get("state")!={"path":str(state_path),"sha256":state_sha}:
 raise ValueError("808 proof state differs from integration checkpoint")
binding=json.loads(binding_path.read_text())
if binding.get("status")!="PASS_938V2_CURRENT_SOURCE_SHELL_RECEIPT_BINDING":
 raise ValueError("1022 binding status is not PASS")
if binding.get("receipt",{}).get("sha256")!=shell_sha:
 raise ValueError("1022 binding does not bind pinned shell receipt")
if binding.get("state",{}).get("sha256")!=state_sha:
 raise ValueError("1022 binding does not bind pinned 818c state")
binding_sources=binding.get("source_surfaces",{})
for role in ("O","I","D"):
 if (binding_sources.get(role,{}).get("path")!=proof.get("source_surfaces",{}).get(role,{}).get("path") or
     binding_sources.get(role,{}).get("sha256")!=proof.get("source_surfaces",{}).get(role,{}).get("sha256")):
  raise ValueError(f"1022 binding {role} source differs from 808 proof")
handoff=json.loads(handoff_path.read_text())
if handoff.get("physical_admission") is not False or handoff.get("native_steps")!=0:
 raise ValueError("938 handoff is not source-only")
controls=json.loads(CONTROLS.read_text())
if controls.get("schema")!="numi.h706.current-230-adapter-source-controls.v1" or controls["preflight"].get("status")!="PASS_CURRENT_808_1022_938_ACTIVE_ROW_PREFLIGHT":
 raise ValueError("1034 source controls do not pass for the checkpoint")
if controls["preflight"].get("state_sha256")!=state_sha or controls["preflight"].get("source_proof_sha256")!=proof_sha:
 raise ValueError("1034 control receipt is stale relative to checkpoint")
if controls["preflight"].get("shell_binding_sha256")!=binding_sha or controls["preflight"].get("shell_receipt_sha256")!=shell_sha:
 raise ValueError("1034 control receipt has stale shell pins")
if controls["preflight"].get("handoff_receipt_sha256")!=handoff_sha or controls["preflight"].get("crosswalk_sha256")!=crosswalk_sha:
 raise ValueError("1034 control receipt has stale 938 pins")
if controls["source"].get("sha256")!=sha(ADAPTER):
 raise ValueError("1034 exporter source differs from controls")
sources=proof.get("source_surfaces",{})
if set(sources)!={"O","I","D"}: raise ValueError("808 proof lacks exact O/I/D surfaces")
role_paths={}
for role in ("O","I","D"):
 p,digest=require_pin(sources[role],f"808 {role} current source")
 role_paths[role]=(p,digest)
# The 330 parent/pose assets are recovered from the existing pinned 228 command plan;
# this plan is used only for the unchanged parent domain, not for current role geometry.
plan=json.loads(PLAN.read_text())
parent=plan["parent"]
parent_path=Path(parent["capture_base"])
poses_path=Path(parent["poses"])
weights_path=Path(str(parent_path)+".weights.csv")
for p,label in ((parent_path,"source330 compiled parent"),(poses_path,"source330 poses"),(weights_path,"source330 full86 weights")):
 if not p.is_file(): raise FileNotFoundError(f"{label} missing: {p}")
if sha(poses_path)!=parent.get("poses_sha256"): raise ValueError("source330 pose hash differs from command plan")
if sha(parent_path)!=PARENT_MASS_SHA: raise ValueError("source330 parent mass file differs from authenticated 228 manifest")
context_preflight=import_and_validate_230_context(parent_path,poses_path,weights_path,plan)
if sha(weights_path)!=parent.get("full86_weights_sha256"): raise ValueError("source330 full86 weights hash differs from command plan")
if OUT.exists(): raise FileExistsError(f"refusing to reuse output directory: {OUT}")
free=shutil.disk_usage(OUT.parent).free
if free<750_000_000: raise OSError(f"source-only 230 output preflight requires >=750000000 free bytes, found {free}")
cmd=[
 str(PYTHON),str(WRAPPER),str(PYTHON),str(ADAPTER),
 "--outer",str(role_paths["O"][0]),"--inner",str(role_paths["I"][0]),"--deep",str(role_paths["D"][0]),
 "--parent",str(parent_path),"--poses",str(poses_path),"--parent-full86-weights",str(weights_path),
 "--current-state",str(state_path),"--source-proof",str(proof_path),
 "--shell-receipt",str(shell_path),"--shell-binding",str(binding_path),
 "--handoff-receipt",str(handoff_path),"--current-surface-node-map",str(crosswalk_path),
 "--out",str(OUT),"--maximum-parent-distance-m","0.02","--tie-tolerance-m","1e-8"
]
manifest={
 "schema":"numi.h706.current-230-run-1034-input-manifest.v1",
 "checkpoint":{"path":str(CHECKPOINT),"sha256":checkpoint_sha},
 "parent_command_plan":{"path":str(PLAN),"sha256":sha(PLAN)},
 "serial_wrapper":{"path":str(WRAPPER),"sha256":sha(WRAPPER)},
 "source_proof":{"path":str(proof_path),"sha256":proof_sha},
 "state":{"path":str(state_path),"sha256":state_sha},
 "shell_receipt":{"path":str(shell_path),"sha256":shell_sha},
 "shell_binding":{"path":str(binding_path),"sha256":binding_sha},
 "handoff":{"path":str(handoff_path),"sha256":handoff_sha},
 "crosswalk":{"path":str(crosswalk_path),"sha256":crosswalk_sha},
 "current_surfaces":{r:{"path":str(p),"sha256":d} for r,(p,d) in role_paths.items()},
 "source330_parent":{"path":str(parent_path),"sha256":sha(parent_path)},
 "source330_poses":{"path":str(poses_path),"sha256":sha(poses_path)},
 "source330_full86_weights":{"path":str(weights_path),"sha256":sha(weights_path)},
 "adapter":{"path":str(ADAPTER),"sha256":sha(ADAPTER),"controls_path":str(CONTROLS),"controls_sha256":sha(CONTROLS)},
 "pinned_217_import_dependency":{"path":str(HELPER_PATH),"sha256":sha(HELPER_PATH)},
 "pinned_230_owner":{"path":str(OWNER_PATH),"sha256":sha(OWNER_PATH)},
 "source330_parent_manifest":{"path":str(PARENT_RUN_MANIFEST),"sha256":sha(PARENT_RUN_MANIFEST)},
 "source330_parent_receipt":{"path":str(LEGACY_228_RECEIPT),"sha256":sha(LEGACY_228_RECEIPT)},
 "source330_context_preflight":context_preflight,
 "serial_wrapper":{"path":str(WRAPPER),"sha256":sha(WRAPPER)},
 "argv":cmd,"output_directory":str(OUT),"free_bytes_before":free,
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"
}
write_json(LAUNCH.parent/"launch-inputs.json",manifest)
if "--preflight-only" in sys.argv:
 preflight={"schema":"numi.h706.current-230-run-1034c-context-preflight.v1",
  "runner":{"path":str(LAUNCH),"sha256":sha(LAUNCH)},
  "adapter":{"path":str(ADAPTER),"sha256":sha(ADAPTER)},
  "source330_context_preflight":context_preflight,
  "status":"PASS_PINNED_INPUT_PATHS_AND_HASHES",
  "launch_inputs":{"path":str(LAUNCH.parent/"launch-inputs.json"),"sha256":sha(LAUNCH.parent/"launch-inputs.json")},
  "free_bytes_before":free,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
 write_json(LAUNCH.parent/"preflight-receipt.json",preflight)
 print(json.dumps(preflight,sort_keys=True,separators=(",",":")))
 raise SystemExit(0)
try:
 env=os.environ.copy()
 current_paths=[str(HELPER_DIR)]+[x for x in env.get("PYTHONPATH","").split(os.pathsep) if x and x!=str(HELPER_DIR)]
 env["PYTHONPATH"]=os.pathsep.join(current_paths)
 result=subprocess.run(cmd,check=False,env=env)
 if result.returncode!=0:
  write_json(EXECUTION,{"schema":"numi.h706.current-230-run-1034-execution.v1","status":"FAIL","returncode":result.returncode,
     "launch_inputs":{"path":str(LAUNCH.parent/"launch-inputs.json"),"sha256":sha(LAUNCH.parent/"launch-inputs.json")},
     "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"})
  raise SystemExit(result.returncode)
 receipt=OUT/"receipt.json"
 if not receipt.is_file(): raise RuntimeError("1034 returned success without output receipt")
 write_json(EXECUTION,{"schema":"numi.h706.current-230-run-1034-execution.v1","status":"PASS_SOURCE_ONLY",
   "launch_inputs":{"path":str(LAUNCH.parent/"launch-inputs.json"),"sha256":sha(LAUNCH.parent/"launch-inputs.json")},
   "receipt":{"path":str(receipt),"sha256":sha(receipt)},
   "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","pythonpath_import_dir":str(HELPER_DIR),"pythonpath_import_sha256":sha(HELPER_PATH)})
except BaseException as exc:
 if not EXECUTION.exists():
  write_json(EXECUTION,{"schema":"numi.h706.current-230-run-1034-execution.v1","status":"FAIL_PREFLIGHT_OR_RUNNER",
   "error_type":type(exc).__name__,"error":str(exc),"launch_inputs":{"path":str(LAUNCH.parent/"launch-inputs.json") if (LAUNCH.parent/"launch-inputs.json").is_file() else None,
   "sha256":sha(LAUNCH.parent/"launch-inputs.json") if (LAUNCH.parent/"launch-inputs.json").is_file() else None},
   "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"})
 raise
