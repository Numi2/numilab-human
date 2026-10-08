#!/usr/bin/env python3
"""Replay the pinned row306 overlay and reviewed two-face edge-star repair."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from numilab_human.lung_row306_recipe import filter_parallel_face_rows
E=Path("/Users/n/numi-human-resting-evidence-20261005")
S1=E/"native-lung-row306-chain-propagation-1048/derive_overlays.py"
OWNER=Path("/tmp/manifold_authoritative_patch_1020.py")
RAW=E/"native-lung-authoritative-patch-trial-1020/intermediate.npz"
NHA=E/"native-lung-seam-sidewall-candidate-947/resting-thorax.nhanatomy"
OS=Path("/Users/n/numi-human-lung-seam-correction-001/src/numilab_human")
DEP=E/"source-mesh-repair-dependency-955/python"
SURF=OS/"resting_anatomy_interface_patch.py"; PREC=OS/"surface_precision_retriangulation.py"
PRED=OS/"cardiac_cavity_intersections.py"; MAN=DEP/"manifold3d.cpython-313-darwin.so"
R1048=E/"native-lung-row306-chain-propagation-1048"
R1049=E/"native-lung-row306-chain-propagation-incidence-review-1049/report.json"
R1052=E/"native-lung-row306-local-two-face-star-repair-1052"
ALT=E/"native-lung-row306-local-two-face-star-repair-1051/candidate-row306.npz"
VENV_PYTHON=Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
PINS={
 S1:"13a0788886f58ca666ec76220a09e56c63e4428982569527080aac73176db834",
 OWNER:"06347dad8ce49696f71a9b02acf327beae11ff8ae9794e825325ba8bd8f5b7a7",
 RAW:"f09349e36478de64f8b387929ddb3d3f6dc94c946d9f97c991b3ef6229d93936",
 NHA:"c8bfdb3055835f53ec7354560323667fba9d01b6283f1d4c0cb24d1eafefd5a3",
 SURF:"e7692e4239c6a5ca4c12667b9bce94c06efa9bc3bec86cbb74e4625eeb416187",
 PREC:"df331a439414a14235b6a1f26b5d4be5d39bd60e5cc016e222eada50f4ae93e1",
 PRED:"11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb",
 MAN:"e76ec87bcff071b06890ae5c2e6ba4906d46e208153c0fddb9d896ae731b3255",
 R1048/"report.json":"94f8d2aac257d0cb6d13cf1f7b6480b2439bfbd444204992ed92c5930c859226",
 R1048/"debug-306-after-overlay-before-validation.npz":"7552f19dade0bb207d39400c17a76a1db97d339a902440062413cb7d8f42f5f6",
 R1048/"source-edge-chain-propagation.json":"653b8822b58c07b1dcddf6b3049059be0bf16307441b12ac82adc8016a4a49c1",
 R1048/"bounded-merge-receipt.json":"308b86681fc94a8c0e4dc566302c89c53435ed13768f770c3efd074c6f4f973e",
 R1048/"bounded-local-cleanup-record.json":"8d4b3721ee18e1c742669be249f1890eb2e6d824a989edd1d38f9f358fdcc0f8",
 R1049:"52bd273caece2f63b6dee1a59fa21c2e3cc61473a1260a1cded725b9fff2b845",
 R1052/"run_two_face_trial.py":"30c8912f0db80b1c15b904f6150d8a4503c081b4ae1ddcc21b7e8c8ab5c11527",
 R1052/"candidate-row306.npz":"0c2fcb224e3da417d52ccd43e8dbe9559def11345eecae38fe9a81152f473ab6",
 R1052/"report.json":"40ef93562ed061aaddf7610f08f64be6853f2d6eb5210b8fae5a4a9dad3dd94d",
}
STAGE={"vertices":"2afbc8efa677746f3e0a54ced86d0cd68c90ec623a3e473a47f5c6c97adc7052","faces":"3686e5007d132d5e051e655c40f1ad75080f9afc1ddf7209a6f34191dd51d5c5","origins":"b3d00058ad12239f2e17b486e75337daf0dc162834bf6d1af94a7b8c4e1b8cd9","kinds":"eae40a2828aed46f2c149217c4da75b67a34d39b897c63b001a5a1782ee6e78c"}
FINAL={"vertices":"2afbc8efa677746f3e0a54ced86d0cd68c90ec623a3e473a47f5c6c97adc7052","faces":"50f50e17a1cc3a05cf7cd2fdea6abdcf99aec15243e33c200ee07725ffc5c4d1","origins":"ca721c5c38f7b764378cf6740029812f2608bb1c06aea79d49b75222a0ba7829","kinds":"736d741e2f1f8196752d6bc71a88c0340fc2e046729466b0e51f8c71e9b03bd0"}
TARGETS=((57553,(40945,40940,40973),57527,1),(57558,(40945,40940,40974),57530,0))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(a):
 a=np.ascontiguousarray(a); h=hashlib.sha256(); h.update(a.dtype.str.encode()); h.update(repr(tuple(a.shape)).encode()); h.update(a.tobytes()); return h.hexdigest()
def hashes(d): return {k:ahash(d[k]) for k in ("vertices","faces","origins","kinds")}
def verify_pins():
 result={}
 for p,w in PINS.items():
  if not p.is_file() or sha(p)!=w: raise RuntimeError(f"missing or changed pinned input: {p}")
  result[str(p)]={"sha256":w,"bytes":p.stat().st_size}
 if not ALT.is_file(): raise RuntimeError(f"missing retained alternate artifact: {ALT}")
 result[str(ALT)]={"sha256":sha(ALT),"bytes":ALT.stat().st_size,"use":"retained but not an input"}
 sys.path.insert(0,str(DEP))
 if importlib.metadata.version("manifold3d")!="3.5.4": raise RuntimeError("requires Manifold3D 3.5.4")
 pybin=Path(sys.executable).resolve()
 if sha(pybin)!="d6923d821cb2fba4e8edf49679d4a3073d3013b60e25a84fe36351493bd7b09e":
  raise RuntimeError("Python executable differs from the pinned preparation runtime")
 import numpy._core._multiarray_umath as numpy_core
 numpy_core_path=Path(numpy_core.__file__).resolve()
 if np.__version__!="2.5.3" or sha(numpy_core_path)!="c3f325f8c20ffada58134c2bf0cfe157bd03bccdaac784168f1b97bd6ba12c23":
  raise RuntimeError("NumPy runtime differs from the pinned preparation runtime")
 result["runtime"]={
  "platform":__import__("platform").platform(),
  "python_version":sys.version,
  "python_executable":str(pybin),
  "python_executable_sha256":sha(pybin),
  "numpy_version":np.__version__,
  "numpy_core_extension":str(numpy_core_path),
  "numpy_core_extension_sha256":sha(numpy_core_path),
  "manifold3d_version":importlib.metadata.version("manifold3d"),
  "manifold3d_extension":str(MAN),
  "manifold3d_extension_sha256":sha(MAN),
 }
 return result
def incidence(faces):
 out={}
 for i,(a,b,c) in enumerate(np.asarray(faces,dtype=np.int64)):
  for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
   e=(min(u,v),max(u,v)); out.setdefault(e,[]).append((i,1 if (u,v)==e else -1))
 return out
def area(v,f):
 t=np.asarray(v,dtype=np.float64)[np.asarray(f,dtype=np.int64)]
 return float((np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)*.5).sum())
def volume(v,f):
 t=np.asarray(v,dtype=np.float64)[np.asarray(f,dtype=np.int64)]
 return float(np.einsum("ij,ij->i",t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)
def owners():
 sys.path.insert(0,str(OS.parent)); sys.path.insert(0,str(DEP))
 from numilab_human.resting_anatomy_interface_patch import topology_report
 from numilab_human.surface_precision_retriangulation import validate_closed_oriented_surface
 from numilab_human.cardiac_cavity_intersections import _audit_pair,_records,float32_point_lattice_key
 return topology_report,validate_closed_oriented_surface,_audit_pair,_records,float32_point_lattice_key
def rebuild(out,py=None):
 out=Path(out).expanduser().resolve()
 if out.exists(): raise RuntimeError(f"refusing to overwrite {out}")
 pins=verify_pins(); py=Path(VENV_PYTHON if py is None else py).expanduser()
 if not py.is_file() or py.resolve()!=Path(sys.executable).resolve():
  raise RuntimeError("stage replay must use the pinned preparation-venv launcher for this Python binary")
 out.mkdir(parents=True); stage=out/"overlay-stage-1048"
 cmd=[str(py),str(S1),"--out",str(stage)]; t=time.monotonic()
 run=subprocess.run(cmd,capture_output=True,text=True,check=False)
 (out/"stage1.stdout").write_text(run.stdout); (out/"stage1.stderr").write_text(run.stderr)
 if run.returncode:
  failure={
   "schema":"numi.human.row306-reconstruction-replay-failure.v1",
   "status":"failed_before_geometry_recipe",
   "reason":"1048 replay subprocess returned nonzero; source-level expected failure has not been assessed",
   "exit_code":run.returncode,"command":cmd,"python_launcher":str(py),
   "python_binary_resolved":str(py.resolve()),
   "stdout_sha256":sha(out/"stage1.stdout"),"stderr_sha256":sha(out/"stage1.stderr"),
   "stderr":run.stderr,"pinned_inputs_before":pins,
  }
  (out/"failure.json").write_text(json.dumps(failure,indent=2,sort_keys=True)+"\n")
  raise RuntimeError(f"1048 replay failed with exit {run.returncode}; failure.json and logs retained")
 sr=json.loads((stage/"report.json").read_text()); row=sr.get("rows",{}).get("306",{})
 expected="ValueError('surface edge (40940, 40945) has 4 incident faces, expected two')"
 if sr.get("complete") is not False or row.get("status")!="failed" or row.get("error")!=expected:
  raise RuntimeError("known 1048 four-use-edge failure did not reproduce")
 comp=sr.get("owner_runtime_compatibility_patch",{})
 if comp.get("geometry_changes") is not False or "out_kinds" not in comp.get("change",""):
  raise RuntimeError("duplicate compaction does not filter patch-kind metadata in lockstep")
 with np.load(stage/"debug-306-after-overlay-before-validation.npz",allow_pickle=False) as z:
  pre={n:np.asarray(z[n]).copy() for n in ("vertices","faces","origins","kinds")}
 sh=hashes(pre)
 if sh!=STAGE: raise RuntimeError("reconstructed 1048 arrays differ from retained stage")
 v=pre["vertices"]; f=pre["faces"].astype(np.int64,copy=False)
 o=pre["origins"].astype(np.int64,copy=False); k=pre["kinds"].astype(np.int8,copy=False)
 if f.shape!=(81946,3) or v.shape!=(40975,6): raise RuntimeError("unexpected 1048 row306 shape")
 e0=incidence(f); bad0={e:u for e,u in e0.items() if len(u)!=2 or sum(x[1] for x in u)!=0}
 if len(bad0)!=5 or len(e0.get((40940,40945),[]))!=4: raise RuntimeError("1048 incidence defect differs")
 keep=np.ones(len(f),dtype=bool); removed=[]
 for i,tri,origin,kind in TARGETS:
  if f[i].tolist()!=list(tri) or int(o[i])!=origin or int(k[i])!=kind: raise RuntimeError(f"target lineage differs at face {i}")
  x=v[f[i],:3].astype(np.float64)
  removed.append({"face_row":i,"vertices":list(tri),"origin":origin,"kind":kind,"xyz_f32_m":x.tolist(),"area_m2":float(np.linalg.norm(np.cross(x[1]-x[0],x[2]-x[0]))*.5)})
  keep[i]=False
 cf,co,ck=filter_parallel_face_rows(f,o,k,keep)
 if not(len(cf)==len(co)==len(ck)==len(f)-2): raise RuntimeError("parallel face arrays are misaligned")
 if not(np.array_equal(cf,f[keep]) and np.array_equal(co,o[keep]) and np.array_equal(ck,k[keep])): raise RuntimeError("aligned filter changed retained order or labels")
 e1=incidence(cf); bad1={e:(len(u),sum(x[1] for x in u)) for e,u in e1.items() if len(u)!=2 or sum(x[1] for x in u)!=0}
 if bad1: raise RuntimeError(f"invalid edges remain: {list(bad1.items())[:4]}")
 trep,validate,audit,records,key=owners(); topo=validate(v[:,:3],cf); detail=trep(cf)
 if topo.get("connected_component_count")!=1 or topo.get("euler_characteristic")!=2 or topo.get("zero_area_face_count")!=0: raise RuntimeError("closed oriented topology check failed")
 if detail.get("boundary_edge_count") or detail.get("nonmanifold_edge_count") or detail.get("orientation_error_edge_count"): raise RuntimeError("topology summary disagrees with validation")
 arr=np.linalg.norm(np.cross(v[cf[:,1],:3].astype(np.float64)-v[cf[:,0],:3].astype(np.float64),v[cf[:,2],:3].astype(np.float64)-v[cf[:,0],:3].astype(np.float64)),axis=1)*.5
 if not np.isfinite(arr).all() or np.any(arr<=0): raise RuntimeError("invalid face area")
 rec=records([key(x[:3]) for x in v],cf); t0=time.monotonic(); self_audit=audit(rec,rec,same_surface=True); self_elapsed=time.monotonic()-t0
 if self_audit.get("count")!=0: raise RuntimeError("exact self-intersection audit failed")
 output={"vertices":v.copy(),"faces":cf,"origins":co,"kinds":ck}; fh=hashes(output)
 if fh!=FINAL: raise RuntimeError("final arrays differ from frozen 1052 result")
 out_npz=out/"candidate-row306.npz"; np.savez_compressed(out_npz,**output)
 if sha(out_npz)!=PINS[R1052/"candidate-row306.npz"]:
  raise RuntimeError("serialized candidate NPZ is not byte-identical to retained 1052 output")
 mp=stage/"bounded-merge-receipt.json"; cp=stage/"source-edge-chain-propagation.json"; lp=stage/"bounded-local-cleanup-record.json"
 merge=json.loads(mp.read_text()); chain=json.loads(cp.read_text()); cleanup=json.loads(lp.read_text())
 if chain.get("source_edge_vertex_indices")!=[30295,30323] or chain.get("incident_original_source_face_ids")!=[57527,57530] or not chain.get("no_vertex_positions_changed"): raise RuntimeError("ordered source-edge chain differs")
 if cleanup.get("source_face_origin")!=57471 or cleanup.get("kind")!="complement" or cleanup.get("revised_local_bound_m")!=2e-7: raise RuntimeError("scoped 200nm cleanup differs")
 if cleanup.get("status")!="applied_to_single_identified_complement_face" or cleanup.get("global_gates_unchanged") is not True: raise RuntimeError("cleanup global gates changed")
 if len(merge.get("conflicts",[]))!=1 or merge["conflicts"][0].get("slot_source_face")!=61269:
  raise RuntimeError("one-ULP conflict source-face identity differs")
 if len(merge.get("resolutions",[]))!=1: raise RuntimeError("one-ULP merge count differs")
 mr=merge["resolutions"][0]; move=float(mr["max_move_from_original_request_m"])
 max_gap=max(float(x.get("independent_serialization_gap_m",0.0) or 0.0) for x in mr["all_requests"])
 max_reprojection=float(mr["max_reprojection_error_m"])
 if (mr.get("source_faces")!=[61268,61269] or move>1e-9
     or not mr.get("all_requests_within_recorded_bounds")):
  raise RuntimeError("merge lineage or recorded local bounds differ")
 if not (cleanup["previous_helper_bound_m"] < cleanup["max_altitude_m"] <= cleanup["revised_local_bound_m"]):
  raise RuntimeError("single complement cleanup is outside its explicitly revised local altitude eligibility interval")
 if "edge(30297,40962)" not in cleanup.get("admission_rule",""):
  raise RuntimeError("the single cleanup no longer names its authorized original indexed edge")
 transitions=cleanup["edge_use_transitions"]
 if sorted(x["uses_before"] for x in transitions)!=[1,1,3] or sorted(x["uses_after"] for x in transitions)!=[0,0,2]:
  raise RuntimeError("the scoped cleanup edge incidence transitions differ")
 va,vb=volume(v[:,:3],cf),volume(v[:,:3],f); aa,ab=area(v[:,:3],cf),area(v[:,:3],f)
 report={
 "schema":"numi.human.row306-reconstruction-recipe.v1","status":"passed_local_topology_and_exact_self_scan",
 "scope":"CPU-only row306 reconstruction from pinned 1020 raw-cut evidence and reviewed local face-star repair; no row311 composition, full all-lobe cross scan, captured pose, or GPU qualification.",
 "runtime_changed":False,"command":{"python":str(py),"argv":cmd,"exit_code":run.returncode,"elapsed_s":time.monotonic()-t},
 "implementation":{"tool_path":str(Path(__file__).resolve()),"tool_sha256":sha(Path(__file__).resolve()),"package_helper_path":str(ROOT/"src/numilab_human/lung_row306_recipe.py"),"package_helper_sha256":sha(ROOT/"src/numilab_human/lung_row306_recipe.py"),"git_revision":subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()},
 "inputs":pins,"1048_checkpoint":{"status":row["status"],"failure":row["error"],"four_use_edge":[40940,40945],"bad_edge_count":len(bad0),"array_hashes":sh,"duplicate_face_metadata_note":comp},
 "operations":{
  "one_ulp_conflict_merge":{"count":len(merge["resolutions"]),"trigger_source_face":merge["conflicts"][0]["slot_source_face"],"merged_source_faces":mr["source_faces"],"selected_authority":mr["selected_authority"],"maximum_independent_serialization_gap_m":max_gap,"max_authority_point_move_m":move,"max_reprojection_error_m":max_reprojection,"all_requests_within_recorded_bounds":True,"receipt_sha256":sha(mp)},
  "scoped_200nm_cleanup":{"origin":cleanup["source_face_origin"],"kind":cleanup["kind"],"max_altitude_m":cleanup["max_altitude_m"],"old_bound_m":cleanup["previous_helper_bound_m"],"revised_bound_m":cleanup["revised_local_bound_m"],"removed_area_m2":cleanup["area_m2"],"component_diameter_m_support_extent_not_displacement":cleanup["component_diameter_m"],"edge_transitions":cleanup["edge_use_transitions"],"receipt_sha256":sha(lp)},
  "ordered_source_edge_chain":{"edge":chain["source_edge_vertex_indices"],"incident_source_faces":chain["incident_original_source_face_ids"],"descendants_considered":chain["descendant_edge_rows_considered"],"existing_points":chain["ordered_existing_points"],"positions_changed":not chain["no_vertex_positions_changed"],"receipt_sha256":sha(cp)},
  "two_face_edge_star_deletion":removed},
 "candidate":{"path":str(out_npz),"sha256":sha(out_npz),"bytes":out_npz.stat().st_size,"reference_1052_sha256":PINS[R1052/"candidate-row306.npz"],"canonical_array_hashes":fh,"positions_unchanged":True,"faces_origins_kinds_filtered_by_one_mask":True},
 "geometry":{"before":{"faces":len(f),"area_m2":ab,"signed_volume_m3":vb,"bad_edges":len(bad0)},"after":{"faces":len(cf),"area_m2":aa,"signed_volume_m3":va,"bad_edges":len(bad1),"topology":topo,"summary":detail,"minimum_face_area_m2":float(arr.min())},"area_delta_m2":aa-ab,"signed_volume_delta_m3":va-vb},
 "exact_self_scan":{"predicate_path":str(PRED),"predicate_sha256":PINS[PRED],"same_surface":True,"result":self_audit,"elapsed_s":self_elapsed},
 "retained_trials":{"1048":{"report":str(R1048/"report.json"),"sha256":PINS[R1048/"report.json"],"status":"expected pre-repair failure reproduced"},"1051":{"path":str(ALT),"sha256":pins[str(ALT)]["sha256"],"status":"retained but not used; associated report unavailable"}}}
 pins_after=verify_pins()
 if pins_after!=pins: raise RuntimeError("pinned source or runtime input changed during replay")
 report["inputs_after"]=pins_after
 (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 (out/"README.md").write_text("# Row306 reconstruction recipe\n\n"
  "On the Mac mini, use a new output path:\n\n"
  f"    {py} {Path(__file__).resolve()} --out /Users/n/numi-human-resting-evidence-20261005/NEW-row306-reconstruction\n\n"
  "It reruns the hash-pinned 1048 overlay and requires its known four-use-edge failure, then applies only the two reviewed face deletions and local topology/exact-self checks. "
  "The 200 nm eligibility bound is limited to the single origin57471 complement microface; global gates are unchanged. The reported component diameter is a support extent, not displacement. Vertex positions are unchanged; faces, origins and patch kinds share one mask. "
  "This is not row311 composition, all-lobe cross-scan, captured-pose or native qualification.\n")
 return report
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--out",required=True,type=Path); p.add_argument("--python",type=Path,default=None); a=p.parse_args()
 try: r=rebuild(a.out,a.python)
 except (OSError,RuntimeError,ValueError,KeyError,TypeError) as exc: print(f"FAIL: {exc}",file=sys.stderr); return 1
 print(json.dumps({"status":r["status"],"candidate_sha256":r["candidate"]["sha256"],"output":str(a.out.expanduser().resolve()),"self_intersections":r["exact_self_scan"]["result"]["count"]},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
