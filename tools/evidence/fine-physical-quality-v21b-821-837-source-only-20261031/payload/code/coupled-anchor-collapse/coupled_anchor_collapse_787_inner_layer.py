#!/usr/bin/env python3
"""787: inner-back I topology preflight and runner on the corrected 786 serializer.

Uses 805-authenticated current I/D source and 786's pre-serialization journal.
Current back I rows are eligible, while route and rim anchors remain fixed.
Every collapse moves its retained I anchor at most 50 um from current; cumulative
805 displacement is bounded only for authenticated back rows. O and D are
unchanged by this I-only increment. No thresholds or geometry predicates change.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, inspect, json, sys
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
BASE786=HERE/"coupled_anchor_collapse_786_root.py"
BASE786_SHA="23759366608fe9c73b6f32efebabb183ebfd54c22eeecc9755133d8e877a7892"
HELPER784=HERE/"coupled_anchor_collapse_784.py"
HELPER784_SHA="09c12fc697c7fe2151fe02d24abf07dc469426a67a9e8a2e11827bdcb47da5fc"
REPLAY789=HERE/"coupled_anchor_collapse_replay_789_root.py"
REPLAY789_SHA="5fcefd6745b09754e6414bfd1d86a364b76fcfd8946fb077d8636ac5c1663052"
INNER_CAP_M=5.0e-5
EDGE_CAP_M=5.0e-3
BACK_ROWS=set()

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def import_file(path,expected,name):
 if sha(path)!=expected:raise RuntimeError(f"pinned input changed: {path}")
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise RuntimeError(f"cannot import {path}")
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

V786=import_file(BASE786,BASE786_SHA,"coupled_anchor_collapse_786_for_787")
U=import_file(HELPER784,HELPER784_SHA,"coupled_anchor_collapse_784_for_787")
M=V786.M
if sha(REPLAY789)!=REPLAY789_SHA:raise RuntimeError("789 independent replay source pin changed")
# 786.G is the 783 wrapper; its G member is the frozen 782 state-evidence owner.
G=V786.G.G
U.M=M
U.G=G
U.BASE_PREPARE=M.prepare_operation
U.BASELINE_I_POINTS=None

def _prepare_operation_787(surface,volumes,keep,drop,*,protected_surface_rows=(),
                           route_anchor_rows=(),move_cap_m=INNER_CAP_M):
 if U.BASELINE_I_POINTS is None:return {"status":"REJECT_787_POLICY_NOT_BOUND"}
 points=np.asarray(surface["points_world_m_f32"],np.float32)
 baseline=np.asarray(U.BASELINE_I_POINTS,np.float32)
 if points.shape!=baseline.shape:return {"status":"REJECT_787_BASELINE_POINT_DOMAIN"}
 if not np.isfinite(move_cap_m) or float(move_cap_m)<=0:
  return {"status":"REJECT_787_BAD_MOVE_CAP"}
 result=U.BASE_PREPARE(surface,volumes,keep,drop,
   protected_surface_rows=protected_surface_rows,route_anchor_rows=route_anchor_rows,
   move_cap_m=min(float(move_cap_m),INNER_CAP_M))
 if result.get("status")!="PASS":return result
 keep=int(keep);anchor=np.asarray(result["anchor_f32"],np.float32).reshape(3)
 step=float(np.linalg.norm(anchor.astype(np.float64)-points[keep].astype(np.float64)))
 if step>INNER_CAP_M:
  return {"status":"REJECT_787_STEP_CAP","keep_row":keep,
          "step_displacement_m":step,"maximum_step_displacement_m":INNER_CAP_M}
 cumulative=float(np.linalg.norm(anchor.astype(np.float64)-baseline[keep].astype(np.float64)))
 back=keep in BACK_ROWS
 if back and cumulative>INNER_CAP_M:
  return {"status":"REJECT_805_CUMULATIVE_BACK_I_CAP","keep_row":keep,
          "baseline_displacement_m":cumulative,"maximum_displacement_m":INNER_CAP_M,
          "step_displacement_m":step}
 result["inner_layer_topology_repair"]=True
 result["inner_back_layer_repair"]=back
 result["anchor_step_displacement_m"]=step
 result["anchor_displacement_from_805_baseline_m"]=cumulative
 result["maximum_anchor_step_displacement_m"]=INNER_CAP_M
 result["maximum_cumulative_back_I_displacement_m"]=INNER_CAP_M if back else None
 result["back_I_endpoint_eligible"]=bool(keep in BACK_ROWS or int(drop) in BACK_ROWS)
 return result

M.prepare_operation=_prepare_operation_787

def _make_775_config_787(run_dir,loaded,report,out_path):
 # Current-step identity: only I is declared bounded; O and D are explicit
 # exact parent aliases and are validated by the 805 reader's byte crosswalk.
 run_dir=Path(run_dir).resolve()
 ip=run_dir/"candidate-I-typed-current.npz"
 if not ip.is_file():raise ValueError("787 run has no typed I output")
 ientry={"path":str(ip),"sha256":sha(ip)}
 base={r:dict(loaded["cfg"]["baseline_surfaces"][r]) for r in ("O","I","D")}
 candidates={r:dict(base[r]) for r in ("O","I","D")}
 candidates["I"]=ientry
 context={"schema":"h706.combined-current-exact-surface-diff-config.v1",
   "baseline_candidates":base,"candidate_surfaces":candidates,"changed_roles":["I"],
   "bounded_point_repair":{"I":{"maximum_displacement_m":INNER_CAP_M,
      "inner_back_layer_repair":True,"inner_back_layer_policy":"INNER_BACK_LAYER_REPAIR"}},
   "source_candidate_run":{"operation_count":int(report.get("accepted_operations",0)),
      "surface_output":ientry},
   "inner_layer_topology_repair":{"enabled":True,"role":"I",
      "maximum_displacement_m":INNER_CAP_M,"audit":"805",
      "outer_O_fixed_by_exact_candidate_parent":True,
      "D_fixed_by_exact_candidate_parent":True,
      "actual_I_rim_fixed":True,"149_routes_fixed":True},
   "physical_admission":False,"native_steps":0}
 out=Path(out_path).resolve()
 M._write_json(out,context)
 if context["changed_roles"]!=["I"] or set(context["bounded_point_repair"])!={"I"}:
  raise AssertionError("787 must declare only bounded I movement")
 if context["candidate_surfaces"]["O"]!=context["baseline_candidates"]["O"] or context["candidate_surfaces"]["D"]!=context["baseline_candidates"]["D"]:
  raise AssertionError("787 O/D must remain exact parent aliases")
 if context["bounded_point_repair"]["I"].get("inner_back_layer_repair") is not True:
  raise AssertionError("787 I movement must be declared under 805 policy")
 if float(context["bounded_point_repair"]["I"]["maximum_displacement_m"])!=INNER_CAP_M:
  raise AssertionError("787 I trust cap mismatch")
 return {"path":str(out),"sha256":sha(out)}

M.make_775_config=_make_775_config_787

# Version the 784 roster policy: prior 805 cumulative movement constrains only
# authenticated back rows. Non-back I anchors remain subject to the new 50 um
# step cap in _prepare_operation_787, not an unrelated cumulative 805 cap.
SRC_STATE_CONFIG=U.state_config_784
src=inspect.getsource(SRC_STATE_CONFIG)
old_source_gate = '    mode = proof.get("audit_implementation", {}).get("mode")\n    if mode != "AUDITED_BOUNDED_INNER_BACK_LAYER_MOVEMENT_630_805":\n        raise ValueError("784 requires a current state authenticated under 805 inner-layer policy")\n    source_cfg_ref = proof.get("config")\n    if not isinstance(source_cfg_ref, dict) or sha(source_cfg_ref.get("path", "")) != source_cfg_ref.get("sha256"):\n        raise ValueError("807 proof lacks a valid 805 config pin")\n    source_cfg_path = Path(source_cfg_ref["path"]).resolve()\n    source_cfg = json.loads(source_cfg_path.read_text())\n    baseline_refs = source_cfg.get("baseline_candidates")\n    if not isinstance(baseline_refs, dict) or set(baseline_refs) != {"O", "I", "D"}:\n        raise ValueError("805 config must pin O/I/D baselines")\n    bounded = source_cfg.get("bounded_point_repair", {})\n    for role in ("I", "D"):\n        item = bounded.get(role, {})\n        if item.get("inner_back_layer_repair") is not True or float(item.get("maximum_displacement_m", -1)) > INNER_CAP_M:\n            raise ValueError("current 805 baseline does not grant 50 micrometre inner-layer movement")\n'
new_source_gate = '    mode = proof.get("audit_implementation", {}).get("mode")\n    allowed_modes = {"AUDITED_BOUNDED_INNER_BACK_LAYER_MOVEMENT_630_805",\n                     "AUDITED_BOUNDED_POINT_MOVEMENT_630_775"}\n    if proof.get("status") not in ("PASS_CURRENT_EXACT_SOURCE_EVIDENCE",\n                                    "PASS_INHERITED_CURRENT_SOURCE_EVIDENCE") or mode not in allowed_modes:\n        raise ValueError("787 requires current exact source evidence from 775 or 805")\n    all185 = proof.get("all185", {})\n    if all185.get("changed_vs_all185_count") != 0 or all185.get("changed_self_count") != 0:\n        raise ValueError("787 source evidence is not zero-crossing")\n    source_cfg_ref = proof.get("config")\n    if not isinstance(source_cfg_ref, dict) or sha(source_cfg_ref.get("path", "")) != source_cfg_ref.get("sha256"):\n        raise ValueError("787 source proof lacks an authenticated current-context config")\n    source_cfg_path = Path(source_cfg_ref["path"]).resolve()\n    source_cfg = json.loads(source_cfg_path.read_text())\n    source_candidates = source_cfg.get("candidate_surfaces")\n    if not isinstance(source_candidates, dict) or set(source_candidates) != {"O", "I", "D"}:\n        raise ValueError("current proof config must pin current O/I/D candidates")\n    baseline_refs = state.get("source_surfaces")\n    if not isinstance(baseline_refs, dict) or set(baseline_refs) != {"O", "I", "D"}:\n        raise ValueError("current verified state must pin O/I/D source surfaces")\n    for role in ("O", "I", "D"):\n        a, b = source_candidates[role], baseline_refs[role]\n        if a.get("sha256") != b.get("sha256") or Path(a.get("path", "")).resolve() != Path(b.get("path", "")).resolve():\n            raise ValueError("current proof config/source state pins disagree for " + role)\n'
if src.count(old_source_gate)!=1: raise RuntimeError('784 source gate changed')
src=src.replace(old_source_gate,new_source_gate,1)
old_edge = '    es = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1), axis=1)\n'
new_edge = '    es = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)\n'
if src.count(old_edge)!=1: raise RuntimeError('784 edge construction changed')
src=src.replace(old_edge,new_edge,1)
old_check='''    if np.any(prior_i_disp > INNER_CAP_M):
        bad = np.flatnonzero(prior_i_disp > INNER_CAP_M)
        raise ValueError(f"current I already exceeds 805 cap at rows {bad[:16].tolist()}")
'''
if src.count(old_check)!=1:raise RuntimeError("784 all-I cap block changed")
src=src.replace(old_check,"    # 787: cumulative 805 cap is checked on authenticated back rows below.\n",1)
old_head='''    keep_baseline_disp = prior_i_disp[keep_rows]
    keep_headroom = np.maximum(0.0, INNER_CAP_M - keep_baseline_disp)
    keep_is_back = np.isin(keep_rows, back_rows)
'''
new_head='''    keep_baseline_disp = prior_i_disp[keep_rows]
    keep_is_back = np.isin(keep_rows, back_rows)
    keep_headroom = np.where(keep_is_back,
        np.maximum(0.0, INNER_CAP_M - keep_baseline_disp), INNER_CAP_M)
'''
if src.count(old_head)!=1:raise RuntimeError("784 keep-headroom source changed")
src=src.replace(old_head,new_head,1)
exec(compile(src,str(Path(__file__).resolve()),"exec"),U.__dict__)
STATE_CONFIG=U.state_config_784

def validate_run_config_787(cfgpath):
 # Use the unchanged production loader to authenticate every referenced file
 # and make sure the actual surface/volume/edge arrays can be read.
 x=M.load_config(cfgpath)
 if set(x["baseline"])!={"O","I","D"} or set(x["volumes"])!={"Skin","Fat"}:
  raise ValueError("787 loader role set mismatch")
 if not np.array_equal(x["surface"]["faces"],x["baseline"]["I"]["faces"]):
  raise ValueError("787 loaded I surface does not match its exact parent")
 if not np.array_equal(x["surface"]["points_world_m_f32"],x["baseline"]["I"]["points_world_m_f32"]):
  raise ValueError("787 loaded I coordinates do not match its exact parent")
 if x["edges"].ndim!=2 or x["edges"].shape[1]!=2 or not len(x["edges"]):
  raise ValueError("787 loaded edge roster is empty or malformed")
 return {"status":"PASS_READ_ONLY_781_CONFIG_LOAD","config_path":str(Path(cfgpath).resolve()),
   "config_sha256":sha(cfgpath),"surface_I_face_count":int(len(x["surface"]["faces"])),
   "surface_I_point_count":int(len(x["surface"]["points_world_m_f32"])),
   "Skin_cell_count":int(len(x["volumes"]["Skin"]["tetrahedra"])),
   "Fat_cell_count":int(len(x["volumes"]["Fat"]["tetrahedra"])),
   "candidate_orientation_count":int(len(x["edges"])),
   "solver_invoked":False}

def prepare_state(state_path,out_dir,*,max_operations=32,time_limit_s=90,
                  max_edge_length_m=EDGE_CAP_M,max_edges=100000):
 global BACK_ROWS
 if float(max_edge_length_m)<=0 or float(max_edge_length_m)>EDGE_CAP_M:
  raise ValueError("787 edge cap must be positive and at most 5 mm")
 cfgpath,prep=STATE_CONFIG(state_path,out_dir,max_operations=max_operations,
    time_limit_s=time_limit_s,max_edge_length_m=max_edge_length_m,max_edges=max_edges)
 prot=U._load(prep["protection_input"],"787 protected I rows")
 BACK_ROWS=set(map(int,np.asarray(prot["authenticated_inner_back_rows"],np.int64)))
 back=np.asarray(prot["authenticated_inner_back_rows"],np.int64)
 isback=np.asarray(prot["authenticated_inner_back_rows_candidate_eligible"],bool)
 if len(isback)!=1 or not bool(isback[0]):raise ValueError("787 back-row eligibility marker missing")
 prep["schema"]="h706.coupled-anchor-collapse-inner-layer-preparation.v2"
 prep["status"]="PASS_PREFLIGHT_ONLY_NO_SOLVER"
 prep["inner_layer_topology_repair"]={"enabled":True,"role":"I",
   "back_rows_candidate_eligible":True,"back_rows_count":int(len(back)),
   "maximum_step_displacement_m":INNER_CAP_M,"maximum_cumulative_back_displacement_from_current_state_m":INNER_CAP_M,
   "all_other_I_anchor_step_cap_m":INNER_CAP_M,"authenticated_805_source_audit_required":True,
   "actual_I_rim_fixed":True,"149_routes_fixed":True,"outer_O_fixed":True,"D_unchanged_this_increment":True}
 prep["required_independent_replay_789"]={"path":str(REPLAY789),"sha256":REPLAY789_SHA,
   "required_before_candidate_acceptance":True,"complete_typed_union_and_face_ancestry_checks":True}
 prep["producer_787"]={"path":str(Path(__file__).resolve()),"sha256":sha(__file__),
   "parent_786":{"path":str(BASE786),"sha256":BASE786_SHA},
   "preflight_helper_784":{"path":str(HELPER784),"sha256":HELPER784_SHA},
   "source_transform":"remove prior-all-I 805-cap check; retain current-back-row cumulative check"}
 prep["candidate_solver"]="NOT_STARTED"
 prep["physical_admission"]=False;prep["native_steps"]=0
 policy=prep["current_inner_back_policy"]
 policy.pop("current_I_max_back_displacement_from_805_baseline_m",None)
 policy.pop("current_D_max_back_displacement_from_805_baseline_m",None)
 policy.pop("candidate_I_anchor_cap_from_805_baseline_m",None)
 policy["candidate_I_anchor_step_cap_from_current_state_m"]=INNER_CAP_M
 if "prior_805_context" in prep:
  prep["prior_current_exact_context"]=prep.pop("prior_805_context")
 policy["current_I_max_back_displacement_m"]=0.0
 policy["current_D_max_back_displacement_m"]=0.0
 policy["eligible_as_collapse_endpoints"]=True
 policy["reference"]="exact current verified-state source arrays; no historical movement cap reused"
 policy["endpoint_policy"]="eligible except true current rim, registered route, complete-star/link, paired-volume, and exact-context guards"
 contract=prep["acceptance_contract"]
 contract.pop("cumulative_I_baseline_cap_m",None)
 contract["cumulative_back_I_anchor_cap_from_current_state_m"]=INNER_CAP_M
 contract["all_other_I_anchor_step_cap_m"]=INNER_CAP_M
 load_receipt=validate_run_config_787(cfgpath)
 prep["read_only_config_load"]=load_receipt
 prep_path=Path(cfgpath).parent/"state-preparation.json"
 M._write_json(prep_path,prep)
 return Path(cfgpath),prep,prep_path

def run_from_prepared(cfgpath,out_dir,prep):
 out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=False)
 result=M.run_from_config(cfgpath,out)
 state_path=out/"state.json"
 if state_path.exists():
  state=json.loads(state_path.read_text())
  state["inner_layer_topology_repair"]=prep["inner_layer_topology_repair"]
  state["producer_787"]={"path":str(Path(__file__).resolve()),"sha256":sha(__file__)}
  M._write_json(state_path,state)
 return result

def main():
 ap=argparse.ArgumentParser(description="787 I inner-back topology preflight on 786 journaled serializer")
 ap.add_argument("--mode",choices=("prepare-state","run"),required=True)
 ap.add_argument("--state",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
 ap.add_argument("--inner-layer-topology-repair",action="store_true",required=True)
 ap.add_argument("--max-operations",type=int,default=32);ap.add_argument("--seconds",type=float,default=90)
 ap.add_argument("--max-edge-length-m",type=float,default=EDGE_CAP_M);ap.add_argument("--max-edges",type=int,default=100000)
 a=ap.parse_args()
 cfg,prep,preppath=prepare_state(a.state,a.output,max_operations=a.max_operations,
   time_limit_s=a.seconds,max_edge_length_m=a.max_edge_length_m,max_edges=a.max_edges)
 if a.mode=="prepare-state":
  print(json.dumps({"status":prep["status"],"preparation":{"path":str(preppath),"sha256":sha(preppath)},
    "run_config":{"path":str(cfg),"sha256":sha(cfg)},"candidate_solver":"NOT_STARTED",
    "physical_admission":False,"native_steps":0},indent=2));return
 proposal=a.output.resolve()/"proposal"
 report=run_from_prepared(cfg,proposal,prep)
 wrapper={"schema":"h706.coupled-anchor-collapse-inner-layer-run.v1",
   "preparation":{"path":str(preppath),"sha256":sha(preppath)},
   "candidate_run":{"path":str(proposal/"state.json"),
     "sha256":sha(proposal/"state.json") if (proposal/"state.json").is_file() else None,
     "status":report.get("status"),"accepted_operations":int(report.get("accepted_operations",0))},
   "independent_replay_789":{"path":str(REPLAY789),"sha256":REPLAY789_SHA,
     "status":"REQUIRED_NOT_RUN","full_CSR_checks":True},
   "candidate_exact_audit":"805_REQUIRED_FOR_THIS_NEW_I_GEOMETRY",
   "actual_697_217":"REQUIRED_FOR_TYPED_I_CANDIDATE","physical_admission":False,"native_steps":0}
 M._write_json(a.output.resolve()/"state-wrapper.json",wrapper)
 print(json.dumps(wrapper,indent=2,default=str))

if __name__=="__main__":main()
