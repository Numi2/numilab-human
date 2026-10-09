#!/usr/bin/env python3
"""Fail-closed delta transfer for exact same-index lung geometry repairs."""
from __future__ import annotations
import argparse, collections, gzip, hashlib, importlib.util, itertools, json, struct, time
from fractions import Fraction
from pathlib import Path
import numpy as np

E=Path("/Users/n/numi-human-resting-evidence-20261005")
PACKAGE=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[4]
AUDIT=REPO/"tools/evidence/native-lung-transformed-pose-audit-1096/source/audit_lung_cycle_1096.py"
AUDIT_SHA="f55bd2646e579700472b70a4d1a498e631f023237725ec91a4b312e43eb15e59"
SUCCESSOR_DMAP_ADAPTER=PACKAGE/"successor_dmap_adapter_1120.py"
SUCCESSOR_DMAP_ADAPTER_SHA="f838af4533f423d36dd8ca38e87529e5da1a82a1204b20e74bb01a2fe8efff67"
BASE_REPORT=E/"native-lung-transformed-pose-audit-runner-1096/attempt-001-native1113-geometry-only/report.json"
BASE_REPORT_SHA="740a9e704c97616f401ce767ef7253503846766da53277e0cefb22d83274800f"
BASE_DECL=BASE_REPORT.parent/"declaration.json"
BASE_DECL_SHA="d1c8107e391d3dfd2112f806afd36a0d528c1ee80b92c7e30fafb97874a43753"
ROWS=(305,306,307,308,309,311)
PLEURA_NATIVE_CHECK=E/"native-lung-v8-integrity-review-1115/compare_native.py"
PLEURA_NATIVE_CHECK_SHA="32f1818241866b6d08d1b559d9f32ed250408ecf09232f86c5c0332abc67247c"
PLEURA_NATIVE_REVIEW=E/"native-lung-v8-integrity-review-1115/review_v8.py"
PLEURA_NATIVE_REVIEW_SHA="d0e74eb333eba379b65394dc4c7b448d6c55116e459d61600e5eb11b789b5431"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def loadmod(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def bits(a):return np.ascontiguousarray(a,dtype="<f4").view("<u4")
def fail(msg):raise ValueError(msg)

def geo_sha(base,row):
 h=hashlib.sha256()
 for name,array,dtype in (("vertices6",row["vertices6"],"<f4"),("faces",row["faces"],"<i8")):
  value=np.ascontiguousarray(array,dtype=dtype)
  h.update(name.encode("ascii")+b"\0")
  h.update(repr(tuple(value.shape)).encode("ascii")+b"\0")
  h.update(value.tobytes(order="C"))
 return h.hexdigest()

def merge_input_hashes(*input_maps):
 out={}
 for input_map in input_maps:
  for path,digest in input_map.items():
   if path in out and out[path]!=digest:
    fail("same input path has conflicting baseline/candidate hashes: "+path)
   out[path]=digest
 return out

def source_delta(old,new):
 if set(old)!=set(new):fail("NHA stable-ID rows changed")
 # This delta reader audits six lung/diaphragm owners. Derived row 310 is
 # excluded from their collision census and receives a separate strict copy proof.
 for sid in sorted(set(old)-set(ROWS)):
  a,b=old[sid],new[sid]
  if any(int(a[k])!=int(b[k]) for k in ("body_index","layer","flags")):
   fail("non-audited row metadata changed: "+str(sid))
  if sid==310:
   # Row 310 is a derived visualization duplicate, not a seventh collision
   # target. Its current exact-copy lineage is separately required below.
   continue
  if not np.array_equal(bits(np.asarray(a["vertices6"],dtype="<f4")),bits(np.asarray(b["vertices6"],dtype="<f4"))):
   fail("non-audited row geometry/attributes changed: "+str(sid))
  if not np.array_equal(np.asarray(a["faces"],dtype=np.int64),np.asarray(b["faces"],dtype=np.int64)):
   fail("non-audited row topology changed: "+str(sid))
 out={}
 for sid in ROWS:
  a,b=old[sid],new[sid]
  if any(int(a[k])!=int(b[k]) for k in ("body_index","layer","flags")):
   fail("audited row identity/semantics changed: "+str(sid))
  av=np.asarray(a["vertices6"],dtype="<f4");bv=np.asarray(b["vertices6"],dtype="<f4")
  af=np.asarray(a["faces"],dtype=np.int64);bf=np.asarray(b["faces"],dtype=np.int64)
  if av.shape!=bv.shape or af.shape!=bf.shape or af.ndim!=2 or af.shape[1]!=3 or av.ndim!=2 or av.shape[1]!=6:fail("row count/topology layout changed: "+str(sid))
  if not np.isfinite(av).all() or not np.isfinite(bv).all():fail("nonfinite audited source vertex data: "+str(sid))
  if np.any(af<0) or np.any(bf<0) or np.any(af>=len(av)) or np.any(bf>=len(bv)):fail("out-of-range face index")
  moved=set(np.flatnonzero(np.any(bits(av[:,:3])!=bits(bv[:,:3]),axis=1)).tolist())
  changed_normals=set(np.flatnonzero(np.any(bits(av[:,3:])!=bits(bv[:,3:]),axis=1)).tolist())
  changed=set(np.flatnonzero(np.any(af!=bf,axis=1)).tolist())
  for fi in range(len(af)):
   if moved.intersection(af[fi].tolist()+bf[fi].tolist()):changed.add(fi)
  out[sid]={"vertices":moved,"faces":changed,"normals":changed_normals}
 return out

def pose_delta(old,new,sd):
 out={}
 for sid in ROWS:
  a,b=old[sid],new[sid];av=np.asarray(a["v"],dtype="<f4");bv=np.asarray(b["v"],dtype="<f4")
  af=np.asarray(a["f"],dtype=np.int64);bf=np.asarray(b["f"],dtype=np.int64)
  if av.shape!=bv.shape or af.shape!=bf.shape:fail("captured row topology changed: "+str(sid))
  moved=set(np.flatnonzero(np.any(bits(av)!=bits(bv),axis=1)).tolist())
  if moved-sd[sid]["vertices"]:fail("captured vertex moved outside source edit support: "+str(sid))
  changed=set(np.flatnonzero(np.any(af!=bf,axis=1)).tolist())|sd[sid]["faces"]
  for fi in range(len(af)):
   if moved.intersection(af[fi].tolist()+bf[fi].tolist()):changed.add(fi)
   if fi not in changed and not np.array_equal(bits(av[af[fi]]),bits(bv[bf[fi]])):changed.add(fi)
  out[sid]=changed
 return out

def points(event):
 return [tuple(Fraction(int(n),int(d)) for n,d in p) for p in event["points_exact_lattice_rational"]]
def recmap(records):return {int(r[3]):r for r in records}

def load_candidate_context(runner,core,run_path,nha_path,nha_sha,d_map_report,lineage_report):
 # Candidate geometry must carry its own exact current payload/per-lobe area
 # and config binding. The predecessor's geometry-only mismatch is not inherited.
 # The child adapter proves exact mapped-face continuity from the immutable V8
 # parent; the pinned 1096 adapter remains unchanged.
 if not SUCCESSOR_DMAP_ADAPTER_SHA or sha(SUCCESSOR_DMAP_ADAPTER)!=SUCCESSOR_DMAP_ADAPTER_SHA:
  fail("successor D-map adapter hash is not frozen")
 previous=(runner.V8_DMAP_ADAPTER,runner.V8_DMAP_ADAPTER_SHA)
 runner.V8_DMAP_ADAPTER=SUCCESSOR_DMAP_ADAPTER
 runner.V8_DMAP_ADAPTER_SHA=SUCCESSOR_DMAP_ADAPTER_SHA
 try:
  return runner.load_owner_context(core,Path(run_path).resolve(),Path(nha_path).resolve(),nha_sha,
    [],Path(lineage_report).resolve(),regression_1078=False,
    d_map_composition_report=Path(d_map_report).resolve(),geometry_only_area_mismatch=False)
 finally:
  runner.V8_DMAP_ADAPTER,runner.V8_DMAP_ADAPTER_SHA=previous

def load_audit_runtime():
 # Load the 1096 runner and its separately pinned 1042 audit core.
 runner=loadmod(AUDIT,"audit_delta_runner_1096")
 core=runner.load_module(runner.require_hash(runner.BASE,runner.BASE_SHA),"native_delta_core_1042")
 runner.load_native_adapters(core)
 if not callable(getattr(core,"classify",None)):
  fail("pinned 1042 core did not receive the native classifiers")
 return runner,core

def derive_pleura_lineage(rows):
 def key(points):
  q=np.ascontiguousarray(points,dtype="<f4").view("<u4").reshape(-1,3)
  return tuple(sorted(tuple(map(int,p)) for p in q))
 lookup=collections.defaultdict(list)
 for sid in range(305,310):
  xyz=np.asarray(rows[sid]["vertices6"],dtype="<f4")[:,:3]
  faces=np.asarray(rows[sid]["faces"],dtype=np.int64)
  for i,face in enumerate(faces):
   lookup[key(xyz[face])].append((sid,i))
 result=[]
 xyz=np.asarray(rows[310]["vertices6"],dtype="<f4")[:,:3]
 for i,face in enumerate(np.asarray(rows[310]["faces"],dtype=np.int64)):
  found=lookup.get(key(xyz[face]),[])
  if len(found)!=1:fail("derived row310 face lacks unique exact lobe-face lineage: "+str(i))
  result.append(found[0])
 return np.asarray(result,dtype="<i4")

def validate_pleura_source_rows(rows,lineage):
 expected=derive_pleura_lineage(rows)
 lineage=np.asarray(lineage)
 if lineage.dtype.kind!="i" or lineage.dtype.itemsize!=4 or lineage.shape!=expected.shape:
  fail("row310 lineage has wrong integer dtype/shape")
 if not np.array_equal(lineage.astype("<i4",copy=False),expected):
  fail("row310 lineage does not equal unique exact current source-face map")
 xyz310=np.asarray(rows[310]["vertices6"],dtype="<f4")[:,:3]
 f310=np.asarray(rows[310]["faces"],dtype=np.int64)
 counts=collections.Counter()
 for sid in range(305,310):
  ids=np.flatnonzero(lineage[:,0]==sid)
  counts[str(sid)]=len(ids)
  if len(ids):
   parent_faces=lineage[ids,1].astype(np.int64)
   parent_xyz=np.asarray(rows[sid]["vertices6"],dtype="<f4")[:,:3]
   pf=np.asarray(rows[sid]["faces"],dtype=np.int64)
   if np.any(parent_faces<0) or np.any(parent_faces>=len(pf)):fail("row310 lineage parent face out of range")
   if not np.array_equal(bits(xyz310[f310[ids]]),bits(parent_xyz[pf[parent_faces]])):
    fail("row310 source face is not same-winding exact copied lobe geometry: "+str(sid))
 return {"face_count":int(len(lineage)),"copied_faces_by_lobe":dict(counts),
         "source_same_winding_exact":True,"unique_exact_face_lineage":True}

def pleura_source_binding(ctx,runner):
 argv=ctx["invocation"]["argv"]
 receipt_path=Path(runner.option_many(argv,"--resting-anatomy-receipt",1)[0]).resolve()
 receipt_sha=ctx["invocation"].get("asset_sha256",{}).get(str(receipt_path))
 if not receipt_sha or sha(receipt_path)!=receipt_sha:fail("candidate anatomy receipt is not invocation-hash bound")
 receipt=json.loads(receipt_path.read_text());payload=receipt.get("payload",{})
 if Path(payload.get("path","")).resolve()!=ctx["nha_path"] or payload.get("sha256")!=ctx["nha_sha"]:
  fail("candidate anatomy receipt does not bind current NHA path/hash")
 if receipt.get("functional_bindings",{}).get("anatomy_payload_sha256")!=ctx["nha_sha"]:
  fail("candidate functional anatomy binding does not match current NHA")
 provenance=receipt.get("provenance",{})
 derived=provenance.get("derived_visceral_pleura_proxy",{})
 if derived.get("output_payload_sha256")!=ctx["nha_sha"]:
  fail("derived pleura record does not bind current NHA")
 # Use the current shared-interface rebuild lineage. The older source-precision
 # record can remain historically bound to an ancestor payload.
 deriv=provenance.get("lung_final_shared_interface_rebuild",{})
 if deriv.get("output_payload_sha256")!=ctx["nha_sha"]:
  fail("current lung/pleura interface rebuild does not bind current NHA")
 lineage_path=Path(deriv.get("row310_face_lineage_path",""))
 if not lineage_path.is_absolute():lineage_path=receipt_path.parent/lineage_path
 lineage_path=lineage_path.resolve();lineage_sha=deriv.get("row310_face_lineage_sha256")
 if not lineage_sha or not lineage_path.is_file() or sha(lineage_path)!=lineage_sha:
  fail("candidate row310 lineage sidecar missing/hash mismatch")
 lineage=np.load(lineage_path,allow_pickle=False)
 proof=validate_pleura_source_rows(ctx["nha_rows"],lineage)
 declared_count=deriv.get("row310_face_lineage_face_count")
 if declared_count is not None and int(declared_count)!=proof["face_count"]:
  fail("candidate receipt row310 lineage face count mismatch")
 return {"receipt":{"path":str(receipt_path),"sha256":receipt_sha},
         "lineage":{"path":str(lineage_path),"sha256":lineage_sha},
         "output_payload_sha256":ctx["nha_sha"],**proof}

def pleura_native_copy_check(ctx,step,lineage,checker):
 pack_path=ctx["run_path"]/("accepted-geometry/step-%d.mrvpack"%step)
 receipt_path=pack_path.with_suffix(".receipt.json")
 if sha(PLEURA_NATIVE_REVIEW)!=PLEURA_NATIVE_REVIEW_SHA:
  fail("pinned native NHA topology reader changed")
 review=checker.load_review_module();pack=checker.Pack(pack_path,receipt_path,step);nha=review.NHA(ctx["nha_path"])
 try:
  mapped=checker.source_nha_topology_check(nha,pack)
  if len(mapped)!=524:fail("native pack does not map all current NHA rows")
  check=checker.check_pleura(nha,nha,lineage,pack)
  if (check.get("faces")!=len(lineage) or
      check.get("exact_xyz_same_winding_faces")!=len(lineage) or
      check.get("same_winding_mismatches")!=0 or
      check.get("maximum_abs_coordinate_delta_m")!=0.0):
   fail("captured row310 is not exact same-winding current lobe copy at step "+str(step))
  return {"accepted_step":int(step),"pack_sha256":pack.pack_sha,
          "lineage_face_count":int(len(lineage)),
          "exact_native_same_winding_copied_faces":int(check["exact_xyz_same_winding_faces"]),
          "same_winding_mismatches":int(check["same_winding_mismatches"]),
          "maximum_abs_coordinate_delta_m":float(check["maximum_abs_coordinate_delta_m"]),
          "native_all_524_row_mapping":True}
 finally:
  nha.close();pack.close()

def reclass_event(ev,changed,pose,recs,maps,lobe,base):
 if ev["type"]=="self":
  sid=int(ev["owner"]);i,j=map(int,ev["faces"])
  if i in changed[sid] or j in changed[sid]:return None
  allowed=base.adjacency(i,j,pose[sid]["v"],pose[sid]["f"],points(ev))
  cl="allowed_indexed_adjacency" if allowed else "unallowed_self_intersection"
 else:
  a,b=map(int,ev["owners"]);i,j=map(int,ev["faces"])
  if i in changed[a] or j in changed[b]:return None
  x,y=recs[a].get(i),recs[b].get(j)
  if x is None or y is None or tuple(ev["face_vertex_ids"][0])!=tuple(x[4]) or tuple(ev["face_vertex_ids"][1])!=tuple(y[4]):
   fail("transferred event face identity mismatch")
  cl=base.classify(a,b,x,y,points(ev),pose,maps,lobe)
 out=dict(ev)
 if out["class"]!=cl:out["transferred_class_from"]=out["class"];out["class"]=cl
 return out

def affected_pairs(pr,ra,rb,ca,cb,same):
 pairs={}
 if same:
  ca=set(ca);sub=[r for r in ra if int(r[3]) in ca]
  streams=(pr._aabb_candidate_pairs(sub,ra,same_surface=True),pr._aabb_candidate_pairs(ra,sub,same_surface=True))
 else:
  ca=set(ca);cb=set(cb);sa=[r for r in ra if int(r[3]) in ca];sb=[r for r in rb if int(r[3]) in cb]
  streams=(pr._aabb_candidate_pairs(sa,rb,same_surface=False),pr._aabb_candidate_pairs(ra,sb,same_surface=False))
 for stream in streams:
  for x,y in stream:
   i,j=int(x[3]),int(y[3])
   if same and i==j:continue
   if same and i>j:i,j,x,y=j,i,y,x
   pairs[(i,j)]=(x,y)
 return pairs

def scan_star(step,a,b,kind,records,changed,pose,maps,lobe,sink,base):
 pr=base.pred()
 pairs=affected_pairs(pr,records[a],records[b],changed[a],changed[b],kind=="self")
 classes=collections.Counter()
 for (i,j),(x,y) in sorted(pairs.items()):
  pts=pr.triangle_intersection_points(x[0],y[0])
  if not pts:continue
  if kind=="self":
   common=set(x[4])&set(y[4]);common_points={x[0][x[4].index(k)] for k in common}
   ok=len(common) in (1,2) and all(pr._allowed_shared_point(p,common_points) for p in pts)
   cl="allowed_indexed_adjacency" if ok else "unallowed_self_intersection"
   ev={"step":step,"type":"self","owner":a,"faces":[i,j],"class":cl,
       "shared_local_vertex_count":len(common),**base.event_points(pts)}
  else:
   cl=base.classify(a,b,x,y,pts,pose,maps,lobe)
   ev={"step":step,"type":"cross","owners":[a,b],"faces":[i,j],
       "face_vertex_ids":[list(x[4]),list(y[4])],"class":cl,**base.event_points(pts)}
  classes[cl]+=1;sink.write(json.dumps(ev,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
 return {"aabb_candidates":len(pairs),"classes":dict(classes)}

def check_base(report_path,decl_path):
 for p,h in ((AUDIT,AUDIT_SHA),(report_path,BASE_REPORT_SHA),(decl_path,BASE_DECL_SHA)):
  if sha(p)!=h:fail("pinned predecessor input changed: "+str(p))
 r=json.loads(report_path.read_text());d=json.loads(decl_path.read_text())
 if r.get("status")!="complete_geometry_scan_with_geometry_failures" or not all(r.get(k) for k in ("complete_pair_coverage","all_6_self_and_15_cross_per_pose","topology_verified_each_pose","input_hashes_unchanged")) or r.get("worker_failures"):
  fail("predecessor is not the pinned complete full scan")
 if r.get("input_hashes_before")!=r.get("input_hashes_after"):fail("predecessor changed inputs")
 for p,h in r["input_hashes_after"].items():
  if not Path(p).is_file() or sha(p)!=h:fail("predecessor evidence input changed: "+p)
 for pose in r["pose_results"]:
  if len(pose["self"])!=6 or len(pose["cross"])!=15:fail("predecessor coverage incomplete")
  for item in pose["witness_files"].values():
   if sha(item["path"])!=item["sha256"]:fail("predecessor witness changed")
 if not r["unclassified_cross_pair_total"]:fail("expected retained predecessor failures absent")
 return r,d

def support_contact_identity(inv):
 argv=inv["argv"]
 if argv.count("--support-contact-payload")!=1:
  fail("support-contact payload flag missing/duplicate")
 i=argv.index("--support-contact-payload")
 if i+1>=len(argv):fail("support-contact payload path missing")
 path=Path(argv[i+1]).resolve()
 assets={str(Path(k).resolve()):v for k,v in inv.get("asset_sha256",{}).items()}
 digest=assets.get(str(path))
 if not digest:fail("support-contact payload is not invocation-hash-bound")
 if not path.is_file() or sha(path)!=digest:
  fail("support-contact payload content/hash differs from its invocation identity")
 return {"path":str(path),"sha256":digest}

def normalized_owner_environment(inv):
 env=dict(inv.get("environment") or {})
 key="NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"
 if key not in env:fail("run-local common-failure receipt environment key missing")
 argv=inv["argv"]
 if len(argv)<=4:fail("native run output path missing from invocation")
 run_path=Path(argv[4]).resolve()
 expected=(run_path/"common-field-failure.json").resolve()
 actual=Path(env[key]).resolve()
 if actual!=expected:fail("common-failure receipt environment path is not bound to this run output")
 env[key]="<run>/common-field-failure.json"
 return env

def generated_scene_identity(inv,support):
 argv=inv["argv"];run=Path(argv[4]).resolve();scene_root=run.parent
 assets={str(Path(k).resolve()):v for k,v in inv.get("asset_sha256",{}).items()}
 paths={}
 for name in ("common-cardiac-domains-f32.bin","common-cardiac-map-f32.bin","common-cardiac-volumes-f32.bin"):
  path=(scene_root/"anatomy"/name).resolve();digest=assets.get(str(path))
  if not digest or not path.is_file() or sha(path)!=digest:
   fail("generated cardiac assembly input is missing or not invocation-hash-bound: "+name)
  paths[str(path)]={"path":str(path),"sha256":digest,"kind":"exact_same_bytes_required"}
 manifest_path=(scene_root/"resting-scene"/"resting-supine-scene.manifest.json").resolve()
 digest=assets.get(str(manifest_path))
 if not digest or not manifest_path.is_file() or sha(manifest_path)!=digest:
  fail("run-local scene manifest is missing or not invocation-hash-bound")
 manifest=json.loads(manifest_path.read_text())
 support_ref=manifest.get("outputs",{}).get("support_contact")
 if not isinstance(support_ref,dict):fail("run-local manifest lacks support-contact identity")
 if Path(support_ref.get("path","")).resolve()!=Path(support["path"]).resolve():
  fail("run-local manifest support path differs from native invocation")
 if support_ref.get("sha256")!=support["sha256"]:
  fail("run-local manifest support digest differs from native invocation")
 normalized=json.loads(json.dumps(manifest))
 normalized["outputs"]["support_contact"]["path"]="<support-contact-payload>"
 canonical=json.dumps(normalized,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
 paths[str(manifest_path)]={"path":str(manifest_path),"sha256":digest,
  "normalized_sha256":hashlib.sha256(canonical).hexdigest(),"kind":"manifest_only_support_path_normalized"}
 return {"assets":paths,"normalized_manifest":normalized,"normalized_manifest_sha256":hashlib.sha256(canonical).hexdigest()}

def compare_run_identity(old,new):
 old_support=support_contact_identity(old["invocation"])
 new_support=support_contact_identity(new["invocation"])
 if old_support["sha256"]!=new_support["sha256"]:
  fail("support-contact payload bytes changed between native runs")
 def args(inv):
  a=list(inv["argv"]);dyn={4:"<run>"}
  for flag,offset in (("--support-contact-payload",1),("--torso-anatomy-payload",1),
                      ("--resting-anatomy-receipt",1),("--resting-movie",1),("--resting-scene",2)):
   if a.count(flag)!=1:fail("argv flag missing/duplicate "+flag)
   i=a.index(flag);dyn[i+offset]="<"+flag+">"
  return [dyn.get(i,x) for i,x in enumerate(a)]
 def dynamic_paths(inv):
  a=inv["argv"];out={str(Path(a[4]).resolve())}
  for flag,offset in (("--support-contact-payload",1),("--torso-anatomy-payload",1),
                      ("--resting-anatomy-receipt",1),("--resting-movie",1),("--resting-scene",2)):
   i=a.index(flag);out.add(str(Path(a[i+offset]).resolve()))
  return out
 if args(old["invocation"])!=args(new["invocation"]):fail("native argv changed beyond output/anatomy/config and content-identical support-payload paths")
 old_env=normalized_owner_environment(old["invocation"])
 new_env=normalized_owner_environment(new["invocation"])
 if old_env!=new_env:fail("native environment changed beyond the run-local receipt path")
 old_generated=generated_scene_identity(old["invocation"],old_support)
 new_generated=generated_scene_identity(new["invocation"],new_support)
 for name in ("common-cardiac-domains-f32.bin","common-cardiac-map-f32.bin","common-cardiac-volumes-f32.bin"):
  old_item=next(v for v in old_generated["assets"].values() if Path(v["path"]).name==name)
  new_item=next(v for v in new_generated["assets"].values() if Path(v["path"]).name==name)
  if old_item["sha256"]!=new_item["sha256"]:
   fail("generated cardiac assembly bytes changed between native runs: "+name)
 if old_generated["normalized_manifest_sha256"]!=new_generated["normalized_manifest_sha256"]:
  fail("run-local scene manifests differ beyond the support-contact output path")
 for c in (old,new):
  if c["metadata"].get("exit_code")!=0 or c["metadata"].get("source_files_changed_during_run"):fail("native run did not exit cleanly with sources unchanged")
  if not c["metadata"].get("loaded_metal_runtime",{}).get("verified"):fail("unverified loaded runtime")
 if old["metadata"]["loaded_metal_runtime"]!=new["metadata"]["loaded_metal_runtime"]:fail("loaded runtime changed")
 if old["dt_seconds"]!=new["dt_seconds"] or old["requested_roots"]!=new["requested_roots"] or old["steps"]!=new["steps"]:fail("native horizon/dt/captures changed")
 def normalized_cfg(inv):
  a=inv["argv"];i=a.index("--resting-scene");path=Path(a[i+2]).resolve()
  expected=inv.get("asset_sha256",{}).get(str(path))
  if not expected or sha(path)!=expected:fail("respiration config not invocation-hash-bound")
  value=json.loads(path.read_text())
  if not isinstance(value,dict):fail("respiration configuration must be an object")
  # These nested fields are provenance, not consumed by the native loader.
  # Every other config field, including numerical settings, stays exact.
  provenance_keys=("parameter_scope","source_geometry_candidate")
  provenance={k:value.get(k) for k in provenance_keys}
  if not isinstance(provenance["parameter_scope"],dict):
   fail("respiration config lacks parameter-scope provenance")
  candidate=provenance["source_geometry_candidate"]
  if not isinstance(candidate,dict) or candidate.get("updated_parameter")!="diaphragm_area_m2":
   fail("respiration config lacks effective-area provenance")
  area=value.get("diaphragm_area_m2");derived=candidate.get("recomputed_area_m2")
  if not isinstance(area,(int,float)) or not isinstance(derived,(int,float)) or not np.isfinite(area) or not np.isfinite(derived):
   fail("respiration config effective area is not finite numeric data")
  area_bits=struct.pack("<f",float(area))
  if area_bits!=struct.pack("<f",float(derived)):
   fail("configured effective area differs from its Float32 provenance value")
  runtime={k:v for k,v in value.items() if k not in provenance_keys}
  return path,expected,runtime,provenance,area_bits.hex()
 oc,nc=normalized_cfg(old["invocation"]),normalized_cfg(new["invocation"])
 if oc[2]!=nc[2]:fail("normalized respiratory numerical configuration differs")
 provenance_changed=[k for k in ("parameter_scope","source_geometry_candidate") if oc[3][k]!=nc[3][k]]
 oa={str(Path(k).resolve()):v for k,v in old["invocation"].get("asset_sha256",{}).items()}
 na={str(Path(k).resolve()):v for k,v in new["invocation"].get("asset_sha256",{}).items()}
 for aset,inv,cfg,generated in ((oa,old["invocation"],oc,old_generated),(na,new["invocation"],nc,new_generated)):
  for path in dynamic_paths(inv)|{str(cfg[0])}|set(generated["assets"]):aset.pop(path,None)
 if oa!=na:fail("native source/runtime/asset identities differ outside declared scene inputs")
 return {"argv_equal_except_declared_paths":True,"environment_equal":True,"loaded_runtime_same_verified":True,
         "normalized_respiration_configuration_equal":True,"run_local_receipt_path_normalized":True,
         "support_contact_payload_identical":True,
         "support_contact_payloads":[old_support,new_support],"unchanged_asset_count":len(oa),
         "baseline_config_sha256":oc[1],"candidate_config_sha256":nc[1],
         "configuration_provenance_fields_excluded_from_runtime_comparison":["parameter_scope","source_geometry_candidate"],
         "configuration_provenance_changed_fields":provenance_changed,
         "baseline_configuration_provenance":oc[3],"candidate_configuration_provenance":nc[3],
         "configured_diaphragm_area_m2":{"baseline":float(oc[3]["source_geometry_candidate"]["recomputed_area_m2"]),
          "candidate":float(nc[3]["source_geometry_candidate"]["recomputed_area_m2"]),
          "float32_bits_baseline":oc[4],"float32_bits_candidate":nc[4],
          "configured_values_equal":oc[3]["source_geometry_candidate"]["recomputed_area_m2"]==nc[3]["source_geometry_candidate"]["recomputed_area_m2"]},
         "generated_scene_inputs_content_bound":True,
         "generated_scene_inputs":{"baseline":old_generated["assets"],"candidate":new_generated["assets"],
          "normalized_manifest_sha256_baseline":old_generated["normalized_manifest_sha256"],
          "normalized_manifest_sha256_candidate":new_generated["normalized_manifest_sha256"],
          "normalized_manifest_equal":old_generated["normalized_manifest_sha256"]==new_generated["normalized_manifest_sha256"]}}

def pose_state(old,new,step):
 def get(c):
  p=c["run_path"]/("accepted-geometry/step-%d.receipt.json"%step);d=json.loads(p.read_text())
  if int(d.get("accepted_step",-1))!=step:fail("receipt step mismatch")
  return d
 a,b=get(old),get(new)
 for k in ("accepted_body_state_sha256","accepted_respiration_state_sha256","accepted_time_s"):
  if a.get(k)!=b.get(k):fail("accepted-state mismatch at %d (%s)"%(step,k))
 return {k:a.get(k) for k in ("accepted_body_state_sha256","accepted_respiration_state_sha256","accepted_time_s")}

def run(args):
 report,decl=check_base(args.baseline_report.resolve(),args.baseline_declaration.resolve())
 runner,base=load_audit_runtime()
 old=runner.load_owner_context(base,Path(report["run_path"]),Path(decl["selected_NHA"]["path"]),decl["selected_NHA"]["sha256"],
   [],decl["lobe_lineage"]["report"],regression_1078=False,d_map_composition_report=decl["D_lobe_map"]["composition_report"],geometry_only_area_mismatch=True)
 if sha(args.candidate_nha)!=args.candidate_nha_sha256:fail("candidate NHA hash mismatch")
 new=load_candidate_context(runner,base,args.candidate_run,args.candidate_nha,args.candidate_nha_sha256,
   args.candidate_d_map_composition_report,args.candidate_lobe_lineage_report)
 ident=compare_run_identity(old,new);sd=source_delta(old["nha_rows"],new["nha_rows"])
 pleura=pleura_source_binding(new,runner)
 pleura_lineage=np.load(pleura["lineage"]["path"],allow_pickle=False)
 if sha(PLEURA_NATIVE_CHECK)!=PLEURA_NATIVE_CHECK_SHA:fail("pinned native pleura copy checker changed")
 pleura_checker=loadmod(PLEURA_NATIVE_CHECK,"native_pleura_copy_checker_1115")
 a310,b310=old["nha_rows"][310],new["nha_rows"][310]
 pleura_delta={"predecessor_vertex_count":len(a310["vertices6"]),"candidate_vertex_count":len(b310["vertices6"]),
   "predecessor_face_count":len(a310["faces"]),"candidate_face_count":len(b310["faces"]),
   "predecessor_geometry_and_attribute_sha256":geo_sha(base,a310),
   "candidate_geometry_and_attribute_sha256":geo_sha(base,b310),
   "geometry_and_attribute_bytes_identical":geo_sha(base,a310)==geo_sha(base,b310),
   "same_vertex_count":len(a310["vertices6"])==len(b310["vertices6"]),
   "same_face_count":len(a310["faces"])==len(b310["faces"]),
   "candidate_row310_copy_proof":pleura}
 if old["steps"]!=report["accepted_steps"] or old["steps"]!=new["steps"]:fail("accepted steps differ")
 out=args.out.resolve()
 if out.exists() or not out.is_relative_to(E):fail("output must be fresh under evidence root")
 out.mkdir(parents=True);started=time.monotonic();poses=[];failures=[]
 oldpr={int(p["accepted_step"]):p for p in report["pose_results"]}
 tracked=merge_input_hashes(old["input_hashes"],new["input_hashes"],
   {str(args.baseline_report.resolve()):sha(args.baseline_report),
    str(args.baseline_declaration.resolve()):sha(args.baseline_declaration),
    pleura["lineage"]["path"]:pleura["lineage"]["sha256"],
    str(PLEURA_NATIVE_CHECK.resolve()):PLEURA_NATIVE_CHECK_SHA,
    str(PLEURA_NATIVE_REVIEW.resolve()):PLEURA_NATIVE_REVIEW_SHA})
 for step in new["steps"]:
  try:
   state=pose_state(old,new,int(step));op,oldpack=base.row_pose(old["run_path"],int(step),old["nha_rows"]);np_,newpack=base.row_pose(new["run_path"],int(step),new["nha_rows"])
   source_support={s:{"vertices":sd[s]["vertices"],"faces":sd[s]["faces"]} for s in ROWS}
   changed=pose_delta(op,np_,source_support)
   orec,orej={},{};nrec,nrej={},{}
   for sid in ROWS:
    orec[sid],orej[sid]=base.exact_records(op[sid]["v"],op[sid]["f"]);nrec[sid],nrej[sid]=base.exact_records(np_[sid]["v"],np_[sid]["f"])
   top=new["owner"].topology(new["owner_module"],new["run_path"],int(step))
   if not top or top.get("all_524_nha_targets_match_current_source") is not True:fail("candidate target topology invalid")
   dm={s:base.native_map(s,new["maps_doc"],np_) for s in base.LOBES};lm=base.native_lobe_maps(new["lineage"],np_)
   if any(not x.get("valid") for x in dm.values()) or any(not x.get("valid") for x in lm.values()):fail("candidate interface map invalid")
   records={s:recmap(nrec[s]) for s in ROWS};eventpose=dict(np_);eventpose["_step"]=int(step)
   pleura_native=pleura_native_copy_check(new,int(step),pleura_lineage,pleura_checker)
   old_summary=oldpr[int(step)];selfp=out/("step-%06d-self.jsonl.gz"%step);crossp=out/("step-%06d-cross.jsonl.gz"%step)
   if selfp.exists() or crossp.exists():fail("refusing to overwrite")
   counts={"self":collections.defaultdict(collections.Counter),"cross":collections.defaultdict(collections.Counter)}
   scan={"self":{},"cross":{}}
   with gzip.open(selfp,"xt",compresslevel=1,encoding="utf-8") as ss,gzip.open(crossp,"xt",compresslevel=1,encoding="utf-8") as cs:
    for kind,sink in (("self",ss),("cross",cs)):
     oldfile=old_summary["witness_files"][kind]
     with gzip.open(oldfile["path"],"rt",encoding="utf-8") as f:
      for line in f:
       e=json.loads(line)
       if int(e.get("step",-1))!=step:fail("old event step mismatch")
       q=reclass_event(e,changed,eventpose,records,dm,lm,base)
       if q is not None:
        sink.write(json.dumps(q,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
        key=(int(e["owner"]),) if kind=="self" else tuple(map(int,e["owners"]));counts[kind][key][q["class"]]+=1
    for sid in ROWS:
     scan["self"][str(sid)]=scan_star(int(step),sid,sid,"self",nrec,changed,eventpose,dm,lm,ss,base)
    for a,b in itertools.combinations(ROWS,2):
     scan["cross"][str(a)+"-"+str(b)]=scan_star(int(step),a,b,"cross",nrec,changed,eventpose,dm,lm,cs,base)
   # Recount complete merged event ledgers, including freshly scanned changed-star hits.
   counts={"self":collections.defaultdict(collections.Counter),"cross":collections.defaultdict(collections.Counter)}
   for kind,path in (("self",selfp),("cross",crossp)):
    with gzip.open(path,"rt",encoding="utf-8") as f:
     for line in f:
      e=json.loads(line);key=(int(e["owner"]),) if kind=="self" else tuple(map(int,e["owners"]))
      counts[kind][key][e["class"]]+=1
   selfrows=[{"owner":s,"face_count":len(np_[s]["f"]),"exact_hit_pairs":sum(counts["self"][(s,)].values()),
    "class_counts":dict(counts["self"][(s,)]),"unallowed_self_pairs":counts["self"][(s,)].get("unallowed_self_intersection",0),
    "degenerate_face_rows":nrej[s]} for s in ROWS]
   crossrows=[]
   for a,b in itertools.combinations(ROWS,2):
    c=counts["cross"][(a,b)]
    crossrows.append({"owners":[a,b],"exact_hit_pairs":sum(c.values()),"class_counts":dict(c),
     "unclassified_cross_pairs":sum(v for k,v in c.items() if k.startswith("unclassified")),
     "degenerate_face_rows":{str(a):nrej[a],str(b):nrej[b]}})
   sbad=sum(x["unallowed_self_pairs"] for x in selfrows);cbad=sum(x["unclassified_cross_pairs"] for x in crossrows);deg=sum(len(nrej[s]) for s in ROWS)
   result={"accepted_step":int(step),"accepted_state_match":state,"pack_sha256":sha(newpack),
    "source_changed_faces":{str(s):sorted(sd[s]["faces"]) for s in ROWS},
    "source_changed_vertex_ids":{str(s):sorted(sd[s]["vertices"]) for s in ROWS},
    "native_changed_faces":{str(s):sorted(changed[s]) for s in ROWS},"topology":top,
    "self":selfrows,"cross":crossrows,"self_coverage":len(selfrows)==6,"cross_coverage":len(crossrows)==15,
    "derived_pleura_native_copy":pleura_native,
    "self_unallowed":sbad,"cross_unclassified":cbad,"degenerate_faces":deg,"changed_star_scan":scan,
    "witness_files":{"self":{"path":str(selfp),"sha256":sha(selfp)},"cross":{"path":str(crossp),"sha256":sha(crossp)}}}
   poses.append(result);write(out/"progress.json",{"completed_poses":len(poses),"poses":poses})
   print(json.dumps({"step":int(step),"changed_face_counts":{str(s):len(changed[s]) for s in ROWS},"unallowed_self":sbad,"unclassified_cross":cbad,"degenerate":deg}),flush=True)
  except Exception as e:
   failures.append({"step":int(step),"error":type(e).__name__+": "+str(e)})
   write(out/"failure.json",{"completed_poses":len(poses),"failure":failures[-1]});break
 unchanged=all(Path(p).is_file() and sha(p)==h for p,h in tracked.items())
 complete=not failures and unchanged and len(poses)==8 and [p["accepted_step"] for p in poses]==new["steps"]
 sbad=sum(p["self_unallowed"] for p in poses);cbad=sum(p["cross_unclassified"] for p in poses);deg=any(p["degenerate_faces"] for p in poses)
 ok=complete and not(sbad or cbad or deg)
 final={"schema":"numi.human.native-transformed-lung-cycle-delta-audit.summary.v1",
  "status":"complete_delta_scan_no_unclassified_hits" if ok else ("complete_delta_scan_with_geometry_failures" if complete else "incomplete_delta_scan"),
  "baseline_report":{"path":str(args.baseline_report.resolve()),"sha256":sha(args.baseline_report)},
  "baseline_run":str(old["run_path"]),"candidate_run":str(new["run_path"]),
  "baseline_nha":{"path":str(old["nha_path"]),"sha256":old["nha_sha"]},
  "candidate_nha":{"path":str(new["nha_path"]),"sha256":new["nha_sha"]},"accepted_steps":new["steps"],
  "requested_roots":new["requested_roots"],"dt_seconds":new["dt_seconds"],"terminal_step_included":new["requested_roots"] in new["steps"],
  "run_identity_comparison":ident,"source_geometry_delta":{str(s):{"changed_faces":sorted(sd[s]["faces"]),"changed_vertices":sorted(sd[s]["vertices"]),"changed_normal_vertex_ids":sorted(sd[s]["normals"])} for s in ROWS},
  "derived_pleura_row_310_delta":pleura_delta,
  "input_hashes_before":tracked,"input_hashes_unchanged":unchanged,"complete_pair_coverage":complete,
  "all_6_self_and_15_cross_per_pose":complete,"topology_verified_each_pose":complete,"pose_results":poses,"worker_failures":failures,
  "unallowed_self_pair_total":sbad,"unclassified_cross_pair_total":cbad,"degenerate_face_seen":deg,
  "respiration_area_binding":{"predecessor":old["area_binding"],"candidate":new["area_binding"]},
  "candidate_area_config_binding_status":new["area_binding"]["status"],
  "physiology_endurance_status":"NOT_ASSESSED_BY_GEOMETRY_DELTA_AUDIT",
  "qualification":"Discrete lung geometry delta audit only. Candidate payload, per-lobe geometry, and effective-area/config binding are checked strictly. Untouched exact events are reclassified from the full predecessor ledger; every current changed-face star is tested against all six target surfaces. Failures remain visible. Not continuous-time or physiological acceptance.",
  "elapsed_s":time.monotonic()-started}
 write(out/"report.json",final);print(json.dumps({"status":final["status"],"complete":complete,"self":sbad,"cross":cbad,"degenerate":deg,"unchanged":unchanged}))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--baseline-report",type=Path,default=BASE_REPORT);p.add_argument("--baseline-declaration",type=Path,default=BASE_DECL)
 p.add_argument("--candidate-run",type=Path,required=True);p.add_argument("--candidate-nha",type=Path,required=True);p.add_argument("--candidate-nha-sha256",required=True)
 p.add_argument("--candidate-d-map-composition-report",type=Path,required=True);p.add_argument("--candidate-lobe-lineage-report",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 run(p.parse_args())
if __name__=="__main__":main()
