#!/usr/bin/env python3
"""Bind the fresh mass-fit source receipt, then execute the pinned 433 refresh owners.

The command writes source-only artifacts. It does not run the native solver or mass
allocation. Every heavy owner is called once through run_geometry_serial.py.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
ROOT=H/"mass-aware-inner-I-001"
PY=Path("/Users/numi/numi-human-prep-venv-20261005/bin/python")
SERIAL=H.parent/"run_geometry_serial.py"
BASE=H/"all-quality-1020-zero-skin-fat-source-qualified-root/verified-state.json"
OWNER818C=H/"bounded625-local-cavity-repair/finalize_818c_volume_only_current_source_prefix_808.py"
OWNER818C_SHA256="9a67725d0a8db5aa0a5b0d067932203f402a9d7b71072087c208b1e2c32e51ac"
BASE_STATE_LATE_BOUND={"late_bound_from":"derived_prepare_receipt.parent_state"}
EXPORT1044_PARENT=H/"downstream-004-same-material-current-reference-229/native-domain-217-source-007/compat/current1020-433-adapter-1044/export_refined_coarse_registered_volume_433_p1_anchor_1044_signed_role_faces.py"
EXPORT1044_PARENT_SHA256="5c4a78a1310e466741341504db1800a23f728567a609367d51842dea63ccecbc"
EXPORT1044_FRESH_AUDIT_SHA256="8244851e506dcf6f5a11b75919f37116a47ef9b5e812ac97de31919470477040"
OWNERS={
 "canonical849":H/"canonical-volume-parent-unions-849-nativepreflight/build_current_manifest_849d_explicit_reference_base.py",
 "canonical834":H/"canonical-volume-parent-unions-834-nativepreflight/canonical_volume_parent_unions_834.py",
 "shell1022":H/"current-shell-1022-native/build_current_shell_1022.py",
 "finalize1022":H/"current-shell-1022-native/finalize_current_shell_receipt_1022.py",
 "bridge938":H/"current-role-p1-support-938-1025-active-source-map/current_fine_volume_to_217_938_818c_849d_active_rows.py",
 "sidecars230":H/"current-registered-sidecars-1034/export_current_registered_sidecars_1034.py",
 "p1support1027":H/"current-role-p1-support-938-1027-pinned-230-dependency/build_current_role_p1_143_support_1027_pinned_230_dependency.py",
 "p1export937":H/"current-role-p1-support-938-1028-active-node-domain/current_domain_p1_export_937_active_node_domain_1028.py",
 "register217":H/"current-native-1024-root/current-217-1036/register_current_combined_volume_nodes_217_1036.py",
 "export1037":H/"downstream-004-same-material-current-reference-229/native-domain-217-source-007/compat/current1020-433-adapter-1024/export_refined_coarse_registered_volume_433_p1_anchor_1024_active_rows_gzip.py",
 "export1043":H/"downstream-004-same-material-current-reference-229/native-domain-217-source-007/compat/current1020-433-adapter-1043/export_refined_coarse_registered_volume_433_p1_anchor_1043_cached_binding_face_diagnostic.py",
 "export1044":H/"downstream-004-same-material-current-reference-229/native-domain-217-source-007/compat/current1020-433-adapter-1044b-1060/export_refined_coarse_registered_volume_433_p1_anchor_1044b_fresh_orientation.py",
 "orientation1057":H/"current-native-1024-root/current-433-1044/audit_1022_role_surface_orientation_1057.py",
 "currentdomain937":H/"current-role-p1-support-938-1028-active-node-domain/current_domain_p1_export_937_active_node_domain_1028.py",
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p).resolve()
 if not p.is_file(): raise ValueError("missing file: "+str(p))
 return {"path":str(p),"sha256":sha(p),"size_bytes":p.stat().st_size}
def checkpin(x,label):
 if not isinstance(x,dict) or not isinstance(x.get("path"),str) or not isinstance(x.get("sha256"),str):
  raise ValueError(label+" is not a path/SHA pin")
 p=Path(x["path"]).resolve()
 if not p.is_file() or sha(p)!=x["sha256"]: raise ValueError(label+" pin mismatch: "+str(p))
 return p
def loadj(p): return json.loads(Path(p).read_text())
def require_status(d,status,label):
 if d.get("status")!=status: raise ValueError(label+" status mismatch: "+str(d.get("status")))
 if d.get("physical_admission") is not False or int(d.get("native_steps",0))!=0:
  raise ValueError(label+" overclaims physical/native execution")
def samepin(a,b,label):
 if not isinstance(a,dict) or not isinstance(b,dict) or Path(a.get("path","")).resolve()!=Path(b.get("path","")).resolve() or a.get("sha256")!=b.get("sha256"):
  raise ValueError(label+" path/SHA mismatch")
def freshout(p):
 p=Path(p).resolve()
 if p.exists(): raise FileExistsError("output must be new: "+str(p))
 p.mkdir(parents=True)
 return p
def owner_pins_from_1058(receipt):
 original=receipt.get("history",{}).get("original_prepare_receipt")
 if not isinstance(original,dict): raise ValueError("derived receipt lacks original 1058 prepare pin")
 opath=checkpin(original,"original 1058 prepare receipt")
 odoc=loadj(opath)
 if odoc.get("schema")!="numi.h706.mass-aware-I-433-refresh-prepare.v1" or odoc.get("status")!="PASS_PREPARED_SOURCE_ONLY_PENDING_806_808":
  raise ValueError("original 1058 prepare receipt schema/status mismatch")
 planpin=odoc.get("owner_plan")
 planpath=checkpin(planpin,"original 1058 owner plan")
 plan=loadj(planpath)
 pins=plan.get("owners",{})
 needed=("bridge938","sidecar230","p1support1027","p1export937","register217","export1037","export1043","export1044","shell1022","finalize_shell1022","orientation1057","serial_runner")
 name_map={"finalize1022":"finalize_shell1022"}
 resolved={}
 for k in needed:
  key=name_map.get(k,k)
  if key not in pins: raise ValueError("1058 owner plan misses "+key)
  pp=checkpin(pins[key], "1058 owner "+key)
  resolved[k]={"path":str(pp),"sha256":sha(pp)}
 for k,p in OWNERS.items():
  if k=="export1044":
   old=resolved.get(k)
   if (old is None or Path(old["path"]).resolve()!=EXPORT1044_PARENT.resolve() or
       old["sha256"]!=EXPORT1044_PARENT_SHA256 or sha(EXPORT1044_PARENT)!=EXPORT1044_PARENT_SHA256):
    raise ValueError("1058 export1044 parent is not the exact frozen source")
   if not p.is_file() or sha(p)!=EXPORT1044_FRESH_AUDIT_SHA256:
    raise ValueError("versioned fresh-orientation 1044 successor pin mismatch")
   resolved["export1044_parent"]=dict(old)
   resolved[k]={"path":str(p.resolve()),"sha256":sha(p)}
  elif k in resolved:
   if Path(resolved[k]["path"]).resolve()!=p.resolve() or sha(p)!=resolved[k]["sha256"]:
    raise ValueError("current owner differs from 1058 pinned owner "+k)
  elif k in ("canonical834","orientation1057","export1037","export1043","currentdomain937"):
   if not p.is_file(): raise ValueError("required owner missing: "+k)
   resolved[k]={"path":str(p.resolve()),"sha256":sha(p)}
 resolved["serial_runner"]=checkpin(pins["serial_runner"],"geometry serial runner")
 return planpin,plan,resolved

def bind(a):
 rp=Path(a.prepare_receipt).resolve()
 prep=loadj(rp)
 if prep.get("schema")!="numi.h706.mass-aware-I-433-refresh-prepare-1059.v1" or prep.get("status")!="PASS_PREPARED_SOURCE_ONLY_PENDING_FINAL_808":
  raise ValueError("fresh 1059 derived preparation receipt is required")
 if prep.get("physical_admission") is not False or int(prep.get("native_steps",0))!=0:
  raise ValueError("derived preparation overclaims physical/native status")
 candidate=checkpin(prep.get("candidate_state"),"final 818c candidate state")
 state=loadj(candidate)
 if state.get("physical_admission") is not False or int(state.get("native_steps",0))!=0:
  raise ValueError("final candidate state is not source-only")
 fit=checkpin(prep.get("fit_report"),"mass-fit report")
 fit_doc=loadj(fit)
 if not str(fit_doc.get("status","")).startswith("PASS") or fit_doc.get("physical_admission") is not False or int(fit_doc.get("native_steps",0))!=0:
  raise ValueError("mass-fit report is not a source-only PASS")
 cand_i=checkpin(prep.get("candidate_I"),"candidate I surface")
 if state.get("source_surfaces",{}).get("I",{}).get("path")!=str(cand_i) or state["source_surfaces"]["I"].get("sha256")!=sha(cand_i):
  raise ValueError("final state I source pin differs from fitted candidate")
 parent=checkpin(prep.get("parent_state"),"original parent state")
 parent_state=loadj(parent)
 for r in ("O","D"):
  samepin(state["source_surfaces"][r],parent_state["source_surfaces"][r],"unchanged "+r+" surface")
 pref=prep.get("metadata_prefix_818c")
 if not isinstance(pref,dict): raise ValueError("derived receipt lacks 818c metadata prefix evidence")
 owner=checkpin(pref.get("owner"),"818c owner")
 if owner.resolve()!=OWNER818C.resolve() or sha(OWNER818C)!=OWNER818C_SHA256:
  raise ValueError("818c owner is not the exact pinned metadata-only finalizer")
 report=checkpin(pref.get("report"),"818c report")
 doc=loadj(report)
 if doc.get("geometry_changed") is not False or doc.get("physical_admission") is not False or int(doc.get("native_steps",0))!=0:
  raise ValueError("818c metadata prefix is not geometry-identity/source-only")
 checkpin(pref.get("input_state"),"818c input state"); samepin(doc.get("input_state"),pref.get("input_state"),"818c report input state")
 samepin(doc.get("output_state"),prep.get("candidate_state"),"818c final output state")
 samepin(state.get("source_state"),pref.get("input_state"),"818c state ancestry")
 samepin(doc.get("producer"),pref.get("owner"),"818c producer")
 audit=checkpin(prep.get("audit_806"),"fresh 806 report")
 proof=checkpin(prep.get("proof_808"),"final 808 proof")
 adoc=loadj(audit);pdoc=loadj(proof)
 if not str(adoc.get("status","")).startswith("PASS") or not str(pdoc.get("status","")).startswith("PASS"):
  raise ValueError("806/808 evidence is not PASS")
 for d,l in ((adoc,"806"),(pdoc,"808")):
  if d.get("physical_admission") is not False or int(d.get("native_steps",0))!=0:
   raise ValueError(l+" overclaims physical/native status")
 samepin(pdoc.get("state"),prep.get("candidate_state"),"final 808 state")
 samepin(state.get("all185_806"),prep.get("audit_806"),"final state 806")
 if not str(state.get("source_audit_status","")).startswith("PASS") or not str(state.get("source_crossing_audit_status","")).startswith("PASS"):
  raise ValueError("final state is missing passing source audit/crossing status")
 a185=adoc.get("all185_receipt"); p185=pdoc.get("all185")
 if not isinstance(a185,dict) or not isinstance(p185,dict): raise ValueError("806/808 missing all185 evidence")
 for k in ("schema","record_count","native_sha256","pinned_manifest_sha256"):
  if a185.get(k)!=p185.get(k): raise ValueError("808 all185 evidence differs from 806 at "+k)
 for r in ("O","I","D"):
  samepin(pdoc.get("source_surfaces",{}).get(r),state["source_surfaces"][r],"808 "+r+" source")
 for r in ("Skin","Fat"):
  vp=checkpin(prep.get("paired_volumes",{}).get(r),r+" final volume")
  samepin(state.get(r+"_current"),prep["paired_volumes"][r],r+" current volume state")
  role=doc.get("volume_regions",{}).get(r,{})
  samepin(role.get("output"),prep["paired_volumes"][r],r+" 818c output volume")
  with np.load(vp,allow_pickle=False) as z:
   for key,want in (("typed_I_candidate_path",str(cand_i)),("typed_I_candidate_sha256",sha(cand_i))):
    if key not in z.files or np.asarray(z[key]).size!=1:
     raise ValueError(r+" final volume lacks scalar "+key)
    got=str(np.asarray(z[key]).reshape(-1)[0])
    if (str(Path(got).resolve()) if key.endswith("_path") else got)!=(str(cand_i) if key.endswith("_path") else want):
     raise ValueError(r+" final volume has stale "+key)
 planpin,oldplan,owners=owner_pins_from_1058(prep)
 out=freshout(a.out_dir)
 roles={}
 for r in ("Skin","Fat"):
  base_entry=parent_state[r+"_current"]
  checkpin(base_entry,"base "+r+" volume")
  roles[r]={"base_volume":base_entry,"current_volume":prep["paired_volumes"][r],
    "steps":[{"kind":"identity","parent":base_entry,"output":prep["paired_volumes"][r]}]}
 manifest={"schema":"numi.h706.current-volume-parent-union-rebuild-input.v1",
  "base_state":prep["parent_state"],"current_state":prep["candidate_state"],"roles":roles}
 mp=out/"canonical834-input.json";mp.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
 ownerpins={k:{"path":str(v["path"]),"sha256":v["sha256"]} for k,v in owners.items()}
 plan={"schema":"numi.h706.mass-aware-I-refresh-execution-plan.v1",
  "status":"PASS_BOUND_FRESH_806_808_818C; DOWNSTREAM_NOT_RUN",
  "source_state":prep["candidate_state"],"prepare_receipt":pin(rp),"fit_report":pin(fit),
  "audit_806":pin(audit),"proof_808":pin(proof),"metadata_prefix_818c":{"owner":pin(owner),"report":pin(report)},
  "canonical834_manifest":pin(mp),"base_state":pin(parent),"owners":ownerpins,
  "derived_owner_inputs":{"export1044":{"parent_path":str(EXPORT1044_PARENT.resolve()),
    "parent_sha256":EXPORT1044_PARENT_SHA256,
    "successor_reason":"accept fresh 1057 report and current 1022 shell receipt while retaining exact signed-face checks"}},
  "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
  "scope":"Refresh/rebind owners only; mass allocation and native execution remain pending"}
 pp=out/"execution-plan.json";pp.write_text(json.dumps(plan,sort_keys=True,indent=2)+"\n")
 receipt={"schema":"numi.h706.mass-aware-I-refresh-execution-bind.v1","status":plan["status"],
  "runner":pin(Path(__file__)),"derived_prepare_receipt":pin(rp),"original_prepare_receipt":prep["history"]["original_prepare_receipt"],"parent_state":pin(parent),
  "source_state":prep["candidate_state"],"fit_report":pin(fit),"audit_806":pin(audit),"proof_808":pin(proof),
  "metadata_prefix_818c":{"owner":pin(owner),"report":pin(report)},
  "canonical834_manifest":pin(mp),"execution_plan":pin(pp),
  "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
 path=out/"bind-receipt.json";path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"status":receipt["status"],"receipt":pin(path),"plan":pin(pp)},sort_keys=True))

def get_output(report,role):
 r=report.get("roles",{}).get(role,{})
 p=r.get("output")
 if not isinstance(p,dict): raise ValueError("834 report missing "+role+" output pin")
 checkpin(p,"834 "+role+" canonical volume")
 return p
def run_serial(stage, owner, args, out, py, serial, logs):
 argv=[str(py),str(serial),str(py),str(owner),*map(str,args)]
 logdir=out/"logs";logdir.mkdir(exist_ok=True)
 base=logdir/(stage)
 start=time.time()
 proc=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 so=base.with_suffix(".stdout.log");se=base.with_suffix(".stderr.log")
 so.write_text(proc.stdout);se.write_text(proc.stderr)
 rec={"stage":stage,"owner":pin(owner),"argv":argv,"started_unix":start,"ended_unix":time.time(),
  "returncode":proc.returncode,"stdout":pin(so),"stderr":pin(se)}
 logs.append(rec)
 write_journal(out,logs)
 if proc.returncode: raise RuntimeError(stage+" failed with exit "+str(proc.returncode)+"; stderr="+str(se))
 return rec
def write_journal(out,logs):
 p=out/"execution-journal.json"
 tmp=p.with_suffix(".tmp")
 tmp.write_text(json.dumps({"schema":"numi.h706.mass-aware-I-refresh-execution-journal.v1",
  "stages":logs,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"},sort_keys=True,indent=2)+"\n")
 os.replace(tmp,p)
def verify_owner_plan(plan):
 out={}
 for k,v in plan["owners"].items():
  out[k]=checkpin(v,"pinned owner "+k)
 for k,p in OWNERS.items():
  if k in out and out[k].resolve()!=p.resolve():raise ValueError("owner path differs from pinned 1058 plan "+k)
 return out
def copy_domain_template(template, out, qstate, volumes, hand, p1, labels, source):
 d=loadj(template)
 d["current_state"]=pin(qstate)
 h=loadj(hand)
 roles=h.get("role_meshes",{})
 for role,key in (("Skin","skin"),("Fat","fat")):
  pr=loadj(p1[role]["receipt"])
  n=roles.get(role,{})
  cells=int(n.get("active_cell_count",pr.get("counts",{}).get("cells",0)))
  if cells<=0: raise ValueError("cannot infer "+role+" cell count")
  q=d["roles"][key]
  q["cell_count"]=cells;q["material_index"]=15 if role=="Skin" else 16
  q["status"]=pr["status"]
  q["current_volume"]=pin(volumes[role])
  q["node"]=pin(Path(hand).parent/(key.lower()+".node"))
  q["ele"]=pin(Path(hand).parent/(key.lower()+".ele"))
  q["support_receipt"]=pin(p1[role]["receipt"])
  q["support_npz"]=pin(p1[role]["support"])
  q["receipt"]=q["support_receipt"]
 d["counts"]={"skin_cells":d["roles"]["skin"]["cell_count"],
  "fat_cells":d["roles"]["fat"]["cell_count"],
  "combined_cells":d["roles"]["skin"]["cell_count"]+d["roles"]["fat"]["cell_count"]}
 d.setdefault("source",{})
 for key in ("parent_mass_csv","full86_weights","projection_helper"):
  d["source"][key]=pin(source[key])
 hdir=Path(hand).resolve().parent
 combined=d["combined"]
 combined["element_map"]=pin(hdir/"skin-fat-combined-element-map.csv.gz")
 combined["handoff_receipt"]=pin(Path(hand).resolve())
 combined["combined_node"]=pin(hdir/"skin-fat-combined.node")
 combined["combined_ele"]=pin(hdir/"skin-fat-combined.ele")
 combined["combined_node_map"]=pin(hdir/"skin-fat-combined-node-map.csv.gz")
 out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(d,sort_keys=True,indent=2)+"\n")
 return out

def validate_execution_inputs(cfgpath):
 cfgpath=Path(cfgpath).resolve();cfgpin=pin(cfgpath);cfg=loadj(cfgpath)
 if cfg.get("schema")!="numi.h706.mass-aware-I-refresh-execution-inputs.v1":
  raise ValueError("execution inputs schema mismatch")
 files=cfg.get("files",{})
 required=("rim_receipt","parent_view_manifest","wall_config","base_state","parent_mass_csv",
  "full86_weights","registered_parent_poses","capture_dir_manifest","p1_domain_template","projection_helper")
 if set(files)!=set(required):raise ValueError("execution inputs must contain exactly the required file/late-binding keys")
 if files.get("base_state")!=BASE_STATE_LATE_BOUND and not isinstance(files.get("base_state"),dict):
  raise ValueError("base_state must be a file pin or the named 1059 parent-state late binding")
 pins={}
 for k in required:
  if k=="capture_dir_manifest":continue
  if k=="base_state" and files[k]==BASE_STATE_LATE_BOUND:
   pins[k]=None
  else:
   pins[k]=checkpin(files[k],k)
 if pins["wall_config"].resolve()!= (H/"bounded625-local-cavity-repair/metadata-finalized-678-v2-001/metadata-finalizer-report.json").resolve():
  raise ValueError("wall_config is not the pinned 818c config")
 if sha(pins["wall_config"])!="f17cf552296aef8a2334475e262aac1f74c56b83170a64a63de62aac756c3f57":
  raise ValueError("818c wall config hash changed")
 mp=checkpin(files["capture_dir_manifest"],"capture_dir_manifest")
 md=loadj(mp);capture_root=Path(md.get("directory","")).resolve()
 if not capture_root.is_dir() or not isinstance(md.get("files"),list) or not md["files"]:
  raise ValueError("capture directory manifest malformed")
 seen=set()
 for item in md["files"]:
  rel=Path(item["relative_path"])
  if rel.is_absolute() or ".." in rel.parts:raise ValueError("capture manifest contains unsafe relative path")
  if str(rel) in seen:raise ValueError("capture manifest repeats a relative path")
  seen.add(str(rel))
  checkpin({"path":str(capture_root/rel),"sha256":item["sha256"]},"capture "+str(rel))
 expected_capture=("compiled-parent-cell-mass.csv","compiled-parent-cell-mass.csv.nodes.csv",
  "compiled-parent-cell-mass.csv.tets.csv","compiled-parent-cell-mass.csv.labels.csv",
  "compiled-parent-cell-mass.csv.weights.csv","parent-layer-rows.csv","registered-skin-poses-100.json")
 if tuple(sorted(seen))!=tuple(sorted(expected_capture)):
  raise ValueError("capture manifest does not pin the exact seven source-330 owner inputs")
 for key,name in (("parent_mass_csv","compiled-parent-cell-mass.csv"),
                  ("full86_weights","compiled-parent-cell-mass.csv.weights.csv"),
                  ("registered_parent_poses","registered-skin-poses-100.json")):
  if pins[key].resolve()!=(capture_root/name).resolve():
   raise ValueError(key+" is not from the pinned source-330 capture")
 template=loadj(pins["p1_domain_template"])
 if template.get("schema")!="numi.native-domain-217.current-p1-owner-pins.v1":
  raise ValueError("P1 template schema mismatch")
 src=template.get("source",{})
 for key in ("parent_mass_csv","full86_weights","projection_helper"):
  checkpin(src.get(key),"P1 template source "+key)
  if src[key]["sha256"]!=sha(pins[key] if key!="projection_helper" else pins["projection_helper"]):
   raise ValueError("P1 template source differs from explicit "+key)
 py=Path(cfg.get("python_executable",str(PY))).resolve()
 if not py.is_file():raise FileNotFoundError(py)
 serial=checkpin({"path":str(SERIAL),"sha256":"88ce1f98414e0125f0cb31a94354db40cfdaba2281f79a9040aa9872d19edf35"},"geometry serial runner")
 for name,path in OWNERS.items():
  if not path.is_file():raise FileNotFoundError("pinned refresh owner missing: "+name)
 if sha(OWNER818C)!=OWNER818C_SHA256:raise ValueError("818c owner hash changed")
 if sha(EXPORT1044_PARENT)!=EXPORT1044_PARENT_SHA256 or sha(OWNERS["export1044"])!=EXPORT1044_FRESH_AUDIT_SHA256:
  raise ValueError("1044 successor/parent source pins changed")
 if sha(OWNERS["orientation1057"])!="9c9a68a7d1122e326b7504737a8d3abeba9d3c1a1ddb011f6320940df09519f1":
  raise ValueError("1057 orientation owner hash changed")
 return cfgpin,cfg,files,pins,capture_root

def preflight(a):
 cfgpin,cfg,files,pins,capture_root=validate_execution_inputs(Path(a.inputs))
 out=Path(a.report).resolve()
 if out.exists():raise FileExistsError("preflight report must be new: "+str(out))
 out.parent.mkdir(parents=True,exist_ok=True)
 report={"schema":"numi.h706.mass-aware-I-refresh-input-preflight.v1",
  "status":"PASS_INPUTS_PINNED_NO_GEOMETRY_EXECUTED",
  "runner":pin(Path(__file__)),"execution_inputs":cfgpin,
  "capture_directory":str(capture_root),"capture_file_count":7,
  "base_state_binding":files["base_state"],
  "candidate_state_binding":"derived_prepare_receipt.candidate_state",
  "expected_parent_current_relation":"parent_state comes from qualified 1059 receipt; candidate_state is a distinct current fitted state; execute enforces the base pin or exact named parent-state derivation",
  "wall_config":pin(pins["wall_config"]),"p1_domain_template":pin(pins["p1_domain_template"]),
  "static_owner_count":len(OWNERS),"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
 out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"status":report["status"],"report":pin(out)},sort_keys=True))

def execute(a):
 bindpath=Path(a.bind_receipt).resolve();br=loadj(bindpath)
 if br.get("schema")!="numi.h706.mass-aware-I-refresh-execution-bind.v1" or br.get("status")!="PASS_BOUND_FRESH_806_808_818C; DOWNSTREAM_NOT_RUN":
  raise ValueError("source bind receipt is not a passing 1059 bind")
 planpath=checkpin(br.get("execution_plan"),"execution plan");plan=loadj(planpath)
 owners=verify_owner_plan(plan)
 cfgpin,cfg,files,pins,capture_root=validate_execution_inputs(Path(a.inputs))
 source=Path(br["source_state"]["path"]).resolve();state=loadj(source)
 if sha(source)!=br["source_state"]["sha256"]:raise ValueError("bound state changed")
 out=freshout(a.out_dir);logs=[]
 write_journal(out,logs)
 py=Path(cfg.get("python_executable",str(PY))).resolve()
 if not py.is_file():raise FileNotFoundError(py)
 serial=Path(owners["serial_runner"])
 # All output paths are derived from this unique invocation directory.
 shdir=out/"shell-1022"; shbind=out/"shell-binding.json"
 run_serial("shell-1022",owners["shell1022"],["--state",source,"--rim-receipt",pins["rim_receipt"],
  "--parent-view-manifest",pins["parent_view_manifest"],"--output",shdir],out,py,serial,logs)
 shellreceipt=shdir/"receipt.json"; skin_shell=shdir/"skin-shell.npz";fat_shell=shdir/"fat-shell.npz"
 for p in (shellreceipt,skin_shell,fat_shell):
  if not p.is_file():raise ValueError("1022 did not produce expected "+str(p))
 run_serial("finalize-shell-1022",owners["finalize1022"],["--receipt",shellreceipt,"--state",source,
  "--report",shbind],out,py,serial,logs)
 if not shbind.is_file():raise ValueError("1022 finalizer did not produce binding report")
 # Build and validate the current ancestry manifest/report.
 base=checkpin(br["parent_state"],"base state")
 configured_base=files.get("base_state")
 if configured_base!=BASE_STATE_LATE_BOUND:
  samepin(configured_base,br["parent_state"],"execution config base state")
 m849=out/"canonical849"/"manifest.json";r849=out/"canonical849"/"report.json"
 run_serial("canonical849",owners["canonical849"],["--base-state",base,"--current-state",source,
  "--manifest-out",m849,"--report-out",r849],out,py,serial,logs)
 r849doc=loadj(r849)
 require_status(r849doc,"PASS_SOURCE_ONLY_AUTODISCOVERED_CELL_ANCESTRY_MANIFEST","849")
 m834=m849
 o834=out/"canonical834"/"volumes";r834=out/"canonical834"/"report.json"
 run_serial("canonical834",owners["canonical834"],["--manifest",m834,"--output-dir",o834,"--report",r834],out,py,serial,logs)
 d834=loadj(r834)
 if d834.get("status")!="PASS_SOURCE_ONLY_CANONICAL_CURRENT_VOLUME_PARENT_UNIONS":raise ValueError("834 failed source-only canonical-union gate")
 if d834.get("physical_admission") is not False or int(d834.get("native_steps",0))!=0 or d834.get("mass_mapping")!="NOT_RUN":raise ValueError("834 overclaims")
 vskin=get_output(d834,"Skin");vfat=get_output(d834,"Fat")
 # 938 pairs the fresh canonical volumes, exact shell, final state and 818c prefix.
 hdir=out/"handoff-938"
 run_serial("handoff-938",owners["bridge938"],["--skin-volume-npz",vskin["path"],"--fat-volume-npz",vfat["path"],
  "--skin-shell-npz",skin_shell,"--fat-shell-npz",fat_shell,"--inner-surface-npz",state["source_surfaces"]["I"]["path"],
  "--current-state",source,"--canonical-834-report",r834,"--canonical-849-report",r849,
  "--volume-prefix-818b-report",br["metadata_prefix_818c"]["report"]["path"],
  "--shell-assembly-receipt",shellreceipt,"--output",hdir],out,py,serial,logs)
 handreceipt=hdir/"adapter-receipt.json";cross=hdir/"skin-fat-combined-registered-surface-node-map.npz"
 hd=loadj(handreceipt)
 if hd.get("status")!="PREPARED_EXISTING_217_COMBINED_CHILD_VOLUME_HANDOFF_NO_PHYSICAL_ADMISSION":raise ValueError("938 handoff status mismatch")
 samepin(hd.get("inputs",{}).get("current_state"),br["source_state"],"938 state")
 for p in (hdir/"skin-fat-combined.node",hdir/"skin-fat-combined.ele",cross,hdir/"skin-fat-combined-node-map.csv.gz",hdir/"skin-fat-combined-element-map.csv.gz"):
  if not p.is_file():raise ValueError("938 missing output "+str(p))
 # Bind the shell/source proof trio to one exact state before generating 230 sidecars.
 side=out/"sidecars-230"
 run_serial("sidecars-230",owners["sidecar230"],["--outer",state["source_surfaces"]["O"]["path"],
  "--inner",state["source_surfaces"]["I"]["path"],"--deep",state["source_surfaces"]["D"]["path"],
  "--parent",pins["parent_mass_csv"],"--poses",pins["registered_parent_poses"],
  "--parent-full86-weights",pins["full86_weights"],"--current-state",source,
  "--source-proof",br["proof_808"]["path"],"--shell-receipt",shellreceipt,"--shell-binding",shbind,
  "--handoff-receipt",handreceipt,"--current-surface-node-map",cross,"--out",side],out,py,serial,logs)
 side_receipt=side/"receipt.json"
 # Fresh current P1/one-hop support on both canonical current volumes.
 p1={}
 for role,key in (("Skin","skin"),("Fat","fat")):
  dest=out/"p1-support-1027"/role
  role_ele=hdir/(key+".ele")
  run_serial("p1-1027-"+role,owners["p1support1027"],["--role",role,"--current-state",source,
   "--volume-npz",vskin["path"] if role=="Skin" else vfat["path"],"--canonical-834-report",r834,
   "--source330-parent-mass",pins["parent_mass_csv"],"--source230-owner",owners["sidecars230"],
   "--role-ele",role_ele,"--handoff-receipt",handreceipt,"--output-dir",dest],out,py,serial,logs)
  rec=dest/"receipt.json";rd=loadj(rec)
  support=rd.get("outputs",{}).get("p1_plus_authenticated_143")
  if not isinstance(support,dict):raise ValueError("1027 "+role+" receipt missing P1+143 output")
  checkpin(support,"1027 "+role+" support")
  p1[role]={"receipt":rec,"support":Path(support["path"])}
 # Regenerate the versioned active-domain config from the pinned template.
 labels=out/"current-domain-937"/"child.labels.csv"
 dcfg=copy_domain_template(pins["p1_domain_template"],out/"current-domain-937/current-domain-pins.json",
   source,{"Skin":Path(vskin["path"]),"Fat":Path(vfat["path"])},handreceipt,p1,labels,pins)
 dcfgsha=sha(dcfg)
 p1root=out/"p1-support-1027"
 run_serial("p1-export-937",owners["p1export937"],["--current-domain-pins",dcfg,
  "--current-domain-pins-sha256",dcfgsha,"--","--parent-mass-csv",pins["parent_mass_csv"],
  "--parent-full86-weights-csv",pins["full86_weights"],"--child-node",hdir/"skin-fat-combined.node",
  "--child-ele",hdir/"skin-fat-combined.ele","--output-labels-csv",labels,
  "--source-map-csv",str(labels)+".source.csv","--receipt-json",out/"current-domain-937/receipt.json",
  "--capacity-profile","full-reference-500k-nodes-2m-cells","--reference-registration-max-distance-m","0.02",
  "--reference-registration-ambiguity-tolerance-m","1e-8","--p1-supported-reference-anchor",
  "--p1-skin-support-npz",p1["Skin"]["support"],"--p1-skin-support-receipt-json",p1["Skin"]["receipt"],
  "--p1-fat-support-npz",p1["Fat"]["support"],"--p1-fat-support-receipt-json",p1["Fat"]["receipt"],
  "--combined-element-map-csv-gz",hdir/"skin-fat-combined-element-map.csv.gz",
  "--combined-node-map-csv-gz",hdir/"skin-fat-combined-node-map.csv.gz",
  "--combined-handoff-receipt-json",handreceipt,"--anchor-csr-npz",out/"current-domain-937/p1-anchor.npz",
  "--anchor-projection-csv",out/"current-domain-937/p1-anchor-projection.csv",
  "--projection-helper-path",pins["projection_helper"],"--projection-helper-sha256",sha(pins["projection_helper"]),
  "--collect-rejection-census","--rejection-witness-json",out/"current-domain-937/rejection-witness.json"],out,py,serial,logs)
 # 217 registration with fresh source proof, shell, sidecars, and typed volume state.
 reg=out/"registration-217"; reg.mkdir(exist_ok=True)
 regmap=reg/"candidate-node-map.csv";regrec=reg/"receipt.json"
 regargs=["--node-map",hdir/"skin-fat-combined-node-map.csv.gz","--node",hdir/"skin-fat-combined.node",
  "--ele",hdir/"skin-fat-combined.ele","--labels",labels,
  "--parent-base",pins["parent_mass_csv"],"--registered-dir",side,
  "--poses",pins["registered_parent_poses"],"--maximum-parent-distance-m","0.02",
  "--ambiguity-tolerance-m","1e-10","--output",regmap,"--receipt",regrec,
  "--capacity-profile","full-reference-500k-nodes-2m-cells","--current-typed-sidecars"]
 for r in ("O","I","D"):regargs += ["--current-typed-role",r,state["source_surfaces"][r]["path"],state["source_surfaces"][r]["sha256"]]
 regargs += ["--current-230-source-proof",br["proof_808"]["path"],br["proof_808"]["sha256"],
  "--current-230-receipt-sha256",sha(side_receipt),"--current-volume-state",str(source),br["source_state"]["sha256"]]
 run_serial("registration-217",owners["register217"],regargs,out,py,serial,logs)
 # Current 433 candidate from the exact 217 registration, labels and P1 anchors.
 canddir=out/"export-1037";canddir.mkdir()
 candnpz=canddir/"candidate-current-typed-volume.npz";candmap=canddir/"candidate-node-map.csv"
 ancestry=canddir/"inserted-node-ancestry.csv";candreceipt=canddir/"receipt.json"
 args=["--current-surface-node-map",cross,sha(cross),"--build-current-typed-volume-candidate",
  "--capture",capture_root,"--child-node",hdir/"skin-fat-combined.node",
  "--child-ele",hdir/"skin-fat-combined.ele","--child-labels",labels,
  "--child-label-source",str(labels)+".source.csv","--p1-anchor-csr",out/"current-domain-937/p1-anchor.npz",
  "--p1-anchor-receipt",out/"current-domain-937/receipt.json","--p1-anchor-projection-csv",out/"current-domain-937/p1-anchor-projection.csv",
  "--child-node-map",regmap]
 for r in ("O","I","D"):args += ["--current-typed-role",r,state["source_surfaces"][r]["path"],state["source_surfaces"][r]["sha256"]]
 args += ["--output-npz",candnpz,"--output-node-map",candmap,"--output-ancestry",ancestry,"--output-receipt",candreceipt]
 run_serial("export-1037",owners["export1037"],args,out,py,serial,logs)
 # Canonical F32 node write through the existing 1024 adapter, then 1043 and 1044.
 canndir=out/"canonical-node-1039";canndir.mkdir()
 cnode=canndir/"candidate-canonical-f32.node"
 run_serial("canonical-node-1039",owners["export1037"],["--current-surface-node-map",cross,sha(cross),
  "--candidate-npz",candnpz,"--candidate-node",cnode,"--candidate-node-map",candmap,
  "--inserted-node-ancestry",ancestry,"--capture",capture_root,
  "--out",canndir/"volume-export","--candidate-operational-reference","--current-typed-volume-candidate",
  *sum((["--current-typed-role",r,state["source_surfaces"][r]["path"],state["source_surfaces"][r]["sha256"]] for r in ("O","I","D")),[])],out,py,serial,logs)
 for p in (cnode,): 
  if not p.is_file():raise ValueError("canonical F32 node owner did not emit "+str(p))
 # Fresh 1057 signed face audit is an input to 1044; bind it to exact current state/shell.
 orient=out/"orientation-1057.json"
 run_serial("orientation-1057",owners["orientation1057"],["--state",source,"--receipt",shellreceipt,"--output",orient],out,py,serial,logs)
 orientdoc=loadj(orient)
 if orientdoc.get("physical_admission") is not False or int(orientdoc.get("native_steps",0))!=0:raise ValueError("1057 orientation audit overclaims")
 roleargs=[]
 for r in ("O","I","D"):roleargs += ["--current-typed-role",r,state["source_surfaces"][r]["path"],state["source_surfaces"][r]["sha256"]]
 for name,owner in (("1043",owners["export1043"]),("1044",owners["export1044"])):
  dest=out/("export-"+name)
  args=["--current-surface-node-map",cross,sha(cross)]
  if name=="1044":args += ["--signed-role-face-audit",orient,sha(orient)]
  args += ["--candidate-npz",candnpz,"--candidate-node",cnode,"--candidate-node-map",candmap,
   "--inserted-node-ancestry",ancestry,"--capture",capture_root,
   "--out",dest,"--candidate-operational-reference","--current-typed-volume-candidate",*roleargs]
  run_serial("export-"+name,owner,args,out,py,serial,logs)
  rec=dest/"receipt.json"
  if rec.is_file():
   d=loadj(rec)
   if d.get("physical_admission") is not False or int(d.get("native_steps",0))!=0:raise ValueError(name+" overclaims")
 # Final evidence inventory; mass and native state remain pending.
 final={"schema":"numi.h706.mass-aware-I-refresh-execution-final.v1",
  "status":"PASS_SOURCE_REFRESH_1044; MASS_MAPPING_NOT_RUN",
  "bind_receipt":pin(bindpath),"execution_inputs":cfgpin,"execution_journal":pin(out/"execution-journal.json"),
  "qualified_state":br["source_state"],"outputs":{},"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
 for key,p in (("canonical849_manifest",out/"canonical849/manifest.json"),("canonical849_report",out/"canonical849/report.json"),
  ("canonical834_report",out/"canonical834/report.json"),("handoff938",handreceipt),("sidecars230",side_receipt),
  ("registration217",regrec),("candidate1037",candnpz),("candidate1037_receipt",candreceipt),
  ("canonical_node",cnode),("orientation1057",orient),("volume_export1044",out/"export-1044/receipt.json")):
  if Path(p).is_file():final["outputs"][key]=pin(p)
 fp=out/"final-refresh-receipt.json";fp.write_text(json.dumps(final,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"status":final["status"],"receipt":pin(fp),"journal":pin(out/"execution-journal.json")},sort_keys=True))
def main():
 ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest="cmd",required=True)
 b=sp.add_parser("bind");b.add_argument("--prepare-receipt",required=True,type=Path);b.add_argument("--out-dir",required=True,type=Path)
 e=sp.add_parser("execute");e.add_argument("--bind-receipt",required=True,type=Path);e.add_argument("--inputs",required=True,type=Path);e.add_argument("--out-dir",required=True,type=Path)
 q=sp.add_parser("preflight");q.add_argument("--inputs",required=True,type=Path);q.add_argument("--report",required=True,type=Path)
 a=ap.parse_args()
 if a.cmd=="bind":bind(a)
 elif a.cmd=="execute":execute(a)
 else:preflight(a)
if __name__=="__main__":main()

