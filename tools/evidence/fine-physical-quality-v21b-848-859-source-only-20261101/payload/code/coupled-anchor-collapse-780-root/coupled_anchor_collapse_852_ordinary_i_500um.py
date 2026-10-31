#!/usr/bin/env python3
"""Ordinary-I paired Skin/Fat topology collapse with actual back/rim/route rows fixed.

Construction-only adapter over frozen 790/781 paired-anchor owner. It reads current
source evidence through 807, applies complete-star protection to actual I back rows
plus source rims/routes, and preserves 789 independent replay. No source or native
quality gate is relaxed.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
BASE790=HERE/"coupled_anchor_collapse_790_inner_layer_root.py"
BASE790_SHA="097e4c7256d23ddee251e782fa8cc81b56275aea6f516adc4ea298fbd4c4070a"
REPLAY789=HERE/"coupled_anchor_collapse_replay_789_root.py"
REPLAY789_SHA="5fcefd6745b09754e6414bfd1d86a364b76fcfd8946fb077d8636ac5c1663052"
SOURCE807=H/"verify_current_source_evidence_807_root.py"
SOURCE807_SHA="9a539227c072936a95b3a9457405521998fb40e5df6139f08d7877fbb49c3f62"
MOVE_CAP_M=5.0e-4
MAX_EDGE_M=5.0e-3


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def import_pinned(path,expected,name):
    if sha(path)!=expected:raise RuntimeError(f"pinned dependency changed: {path}")
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError(f"cannot import {path}")
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

V790=import_pinned(BASE790,BASE790_SHA,"ordinary_i_v790")
R789=import_pinned(REPLAY789,REPLAY789_SHA,"ordinary_i_replay789")
E807=import_pinned(SOURCE807,SOURCE807_SHA,"ordinary_i_source807")
M=V790.M
U=V790.U
BASE_PREPARE=U.BASE_PREPARE
BASE_RUN=M.run_from_config
BACK_ROWS=set()


def _same_pin(a,b):
    return (isinstance(a,dict) and isinstance(b,dict) and
            Path(str(a.get("path",""))).resolve()==Path(str(b.get("path",""))).resolve() and
            str(a.get("sha256",""))==str(b.get("sha256","")))


_PROOF_CACHE={}

def _source_evidence_807(state_path,state):
    """Use current 807 proof directly; do not inherit 782's fixed-O gate."""
    state_path=Path(state_path).resolve()
    state_key=(str(state_path),sha(state_path))
    if state_key in _PROOF_CACHE:
        proof=_PROOF_CACHE[state_key]
    else:
        proof=E807.verify_current_source_evidence(state_path)
        _PROOF_CACHE[state_key]=proof
    if proof.get("status") not in ("PASS_CURRENT_EXACT_SOURCE_EVIDENCE","PASS_INHERITED_CURRENT_SOURCE_EVIDENCE"):
        raise ValueError("807 current source evidence did not pass")
    if proof.get("physical_admission") is not False or int(proof.get("native_steps",-1))!=0:
        raise ValueError("807 source proof must remain construction-only")
    pins=proof.get("source_surfaces");declared=state.get("source_surfaces")
    if not isinstance(pins,dict) or set(pins)!={"O","I","D"} or not isinstance(declared,dict) or set(declared)!={"O","I","D"}:
        raise ValueError("807 proof/state must bind all O/I/D source pins")
    for role in ("O","I","D"):
        if not _same_pin(pins[role],declared[role]):
            raise ValueError("807 source proof pin differs from current state for "+role)
    return proof

# 784 state preparation consults this hook. Direct 807 binding allows an already
# audited current O geometry while retaining exact O/I/D proof.
V790.G._source_evidence=_source_evidence_807


def _ordinary_i_prepare(surface,volumes,keep,drop,*,protected_surface_rows=(),route_anchor_rows=(),move_cap_m=MOVE_CAP_M):
    """Retain the 781 pair fit but allow 0.5 mm only on unprotected I stars."""
    if float(move_cap_m)<=0 or float(move_cap_m)>MOVE_CAP_M:
        return {"status":"REJECT_852_MOVE_CAP"}
    protected=set(map(int,protected_surface_rows))
    routes=set(map(int,route_anchor_rows))
    keep=int(keep);drop=int(drop)
    if keep in BACK_ROWS or drop in BACK_ROWS or {keep,drop}&routes:
        return {"status":"REJECT_852_PROTECTED_ENDPOINT"}
    faces=np.asarray(surface["faces"],np.int64)
    endpoint_rows=np.union1d(np.flatnonzero(np.any(faces==keep,axis=1)),np.flatnonzero(np.any(faces==drop,axis=1)))
    star_vertices=set(map(int,faces[endpoint_rows].reshape(-1)))
    if star_vertices & (protected|BACK_ROWS):
        return {"status":"REJECT_852_PROTECTED_COMPLETE_STAR"}
    result=BASE_PREPARE(surface,volumes,keep,drop,
        protected_surface_rows=protected|BACK_ROWS,route_anchor_rows=routes,move_cap_m=float(move_cap_m))
    if result.get("status")!="PASS":return result
    anchor=np.asarray(result["anchor_f32"],np.float32).reshape(3)
    old=np.asarray(surface["points_world_m_f32"],np.float32)[keep].astype(np.float64)
    step=float(np.linalg.norm(anchor.astype(np.float64)-old))
    if step>MOVE_CAP_M:return {"status":"REJECT_852_STEP_CAP","step_m":step}
    result["ordinary_inner_I_topology_repair"]=True
    result["actual_back_I_fixed"]=True
    result["anchor_step_displacement_m"]=step
    result["maximum_ordinary_I_step_displacement_m"]=MOVE_CAP_M
    return result

M.prepare_operation=_ordinary_i_prepare


def _make_775_config_852(run_dir,loaded,report,out_path):
    run_dir=Path(run_dir).resolve();ip=run_dir/"candidate-I-typed-current.npz"
    if not ip.is_file():raise ValueError("ordinary-I candidate output is missing")
    ientry={"path":str(ip),"sha256":sha(ip)}
    base={r:dict(loaded["cfg"]["baseline_surfaces"][r]) for r in ("O","I","D")}
    candidates={"O":base["O"],"I":ientry,"D":base["D"]}
    context={"schema":"h706.combined-current-exact-surface-diff-config.v1",
      "baseline_candidates":base,"candidate_surfaces":candidates,"changed_roles":["I"],
      "bounded_point_repair":{"O":{"maximum_displacement_m":0.0},
        "I":{"maximum_displacement_m":MOVE_CAP_M,"ordinary_inner_I_topology_repair":True},
        "D":{"maximum_displacement_m":0.0}},
      "source_candidate_run":{"operation_count":int(report.get("accepted_operations",0)),"surface_output":ientry},
      "ordinary_inner_I_topology_repair":{"enabled":True,"role":"I",
        "maximum_displacement_m":MOVE_CAP_M,"actual_back_I_rows_fixed":True,
        "I_source_rims_fixed":True,"I_149_route_anchors_fixed":True,
        "outer_O_and_D_current_candidates_exact":True,"audit":"775"},
      "physical_admission":False,"native_steps":0}
    out=Path(out_path).resolve();M._write_json(out,context)
    return {"path":str(out),"sha256":sha(out)}

M.make_775_config=_make_775_config_852


def _filter_star_roster(cfgpath,prep_path,out_dir):
    cfg=json.loads(Path(cfgpath).read_text());prep=json.loads(Path(prep_path).read_text())
    rawprot=U._load(prep["protection_input"],"852 existing protection input")
    back=np.asarray(rawprot["authenticated_inner_back_rows"],np.int64)
    if back.ndim!=1 or len(np.unique(back))!=len(back):raise ValueError("852 actual back I rows are malformed")
    oldprotected=np.asarray(rawprot["protected_surface_rows"],np.int64)
    routes=np.asarray(rawprot["route_anchor_rows"],np.int64)
    rims=np.asarray(rawprot.get("source_rim_rows",np.empty(0,np.int64)),np.int64)
    protected=np.unique(np.r_[oldprotected,back,rims,routes]).astype(np.int64)
    surface=M.load_pinned_npz(cfg["surface_I"],"852 current I surface")
    faces=np.asarray(surface["faces"],np.int64);npoint=len(surface["points_world_m_f32"])
    if np.any(protected<0) or np.any(protected>=npoint):raise ValueError("852 protected I row outside point domain")
    protectmask=np.zeros(npoint,bool);protectmask[protected]=True
    face_touch=np.any(protectmask[faces],axis=1)
    vertices_touch=np.zeros(npoint,bool)
    if np.any(face_touch):vertices_touch[np.unique(faces[face_touch])]=True
    edge_ref=cfg["candidate_edges"];old_edges=U._load(edge_ref,"852 candidate edges")
    und=np.asarray(old_edges["undirected_edges"],np.int64).reshape((-1,2))
    blocked=np.any(vertices_touch[und],axis=1) if len(und) else np.empty(0,bool)
    keep_u=~blocked;und_new=und[keep_u]
    edge_map={tuple(sorted(map(int,e))):bool(k) for e,k in zip(und,keep_u)}
    ce=np.asarray(old_edges["candidate_edges"],np.int64).reshape((-1,2))
    keep_e=np.asarray([edge_map.get(tuple(sorted(map(int,e))),False) for e in ce],bool)
    edge_new={}
    for key,value in old_edges.items():
        a=np.asarray(value)
        if key=="candidate_edges":edge_new[key]=a[keep_e]
        elif key=="undirected_edges":edge_new[key]=a[keep_u]
        elif a.ndim and len(a)==len(ce):edge_new[key]=a[keep_e]
        elif a.ndim and len(a)==len(und):edge_new[key]=a[keep_u]
        else:edge_new[key]=a
    root=Path(out_dir).resolve();inputs=root/"inputs"
    edge_path=inputs/"candidate-I-edges-ordinary-back-fixed.npz"
    prot_path=inputs/"protected-I-ordinary-back-fixed.npz"
    np.savez_compressed(edge_path,**edge_new)
    np.savez_compressed(prot_path,protected_surface_rows=protected,route_anchor_rows=routes,
        source_rim_rows=rims,actual_back_I_rows=back,actual_back_I_rows_fixed=np.asarray([True]))
    cfg["candidate_edges"]={"path":str(edge_path),"sha256":sha(edge_path),"key":"candidate_edges"}
    cfg["protected_surface_rows"]={"path":str(prot_path),"sha256":sha(prot_path),"key":"protected_surface_rows"}
    cfg["route_anchor_rows"]={"path":str(prot_path),"sha256":sha(prot_path),"key":"route_anchor_rows"}
    cfg["move_cap_m"]=MOVE_CAP_M
    cfg.pop("inner_layer_topology_repair",None)
    cfg.pop("inner_layer_audit_baseline_candidates",None)
    cfg["ordinary_inner_I_topology_repair"]={"enabled":True,"role":"I",
        "maximum_displacement_m":MOVE_CAP_M,"actual_back_I_rows_fixed":True,
        "protected_complete_star_rows":int(len(protected)),"source_rims_fixed":True,
        "149_I_routes_fixed":True,"outer_O_and_D_exact_current_parent":True,
        "source_proof":"807","candidate_exact_audit":"775_REQUIRED"}
    M._write_json(Path(cfgpath),cfg)
    prep["ordinary_inner_I_topology_repair"]={"enabled":True,"role":"I",
        "maximum_displacement_m":MOVE_CAP_M,"actual_back_I_rows_fixed":True,
        "actual_back_I_row_count":int(len(back)),"rim_row_count":int(len(rims)),
        "route_row_count":int(len(routes)),"protected_union_count":int(len(protected)),
        "excluded_edges_touching_protected_complete_stars":int(np.count_nonzero(blocked)),
        "candidate_edges_before_filter":int(len(und)),"candidate_edges_after_filter":int(len(und_new)),
        "source_proof":"807","independent_replay":"789_REQUIRED"}
    prep["edge_roster"]={"path":str(edge_path),"sha256":sha(edge_path)}
    prep["run_config"]={"path":str(Path(cfgpath).resolve()),"sha256":sha(cfgpath)}
    prep["acceptance_contract"].update({"anchor_step_cap_m":MOVE_CAP_M,
        "actual_back_I_fixed":True,"source_rims_fixed":True,"149_routes_fixed":True,
        "exact_paired_volume_delta_integer":"0","all_native_and_exact_positive":True})
    M._write_json(prep_path,prep)
    return cfg,prep,{"back_rows":back,"rims":rims,"routes":routes,"protected":protected,
        "candidate_edges_before":len(und),"candidate_edges_after":len(und_new),
        "excluded_edges":int(np.count_nonzero(blocked))}


def prepare_state(state_path,out_dir,*,max_operations=32,time_limit_s=90,max_edge_length_m=MAX_EDGE_M,max_edges=100000):
    global BACK_ROWS
    state_path=Path(state_path).resolve();out=Path(out_dir).resolve()
    proof=_source_evidence_807(state_path,json.loads(state_path.read_text()))
    cfgpath,prep=V790.STATE_CONFIG(state_path,out,max_operations=max_operations,
        time_limit_s=time_limit_s,max_edge_length_m=max_edge_length_m,max_edges=max_edges)
    prep_path=cfgpath.parent/"state-preparation.json"
    cfg,prep,mask=_filter_star_roster(cfgpath,prep_path,out)
    BACK_ROWS=set(map(int,mask["back_rows"]))
    if proof.get("status") not in ("PASS_CURRENT_EXACT_SOURCE_EVIDENCE","PASS_INHERITED_CURRENT_SOURCE_EVIDENCE"):
        raise ValueError("852 state preparation lacks 807 current source evidence")
    prep["source_807_verification"]={"status":proof.get("status"),
        "audit_mode":proof.get("audit_implementation",{}).get("mode"),
        "state":{"path":str(state_path),"sha256":sha(state_path)},
        "implementation":{"path":str(SOURCE807.resolve()),"sha256":SOURCE807_SHA}}
    M._write_json(prep_path,prep)
    return Path(cfgpath),prep,prep_path


def run_from_config(config_path,out_dir,prune_receipt=None):
    global BACK_ROWS
    config_path=Path(config_path).resolve();cfg=json.loads(config_path.read_text())
    prot=M.load_pinned_npz(cfg["protected_surface_rows"],"852 ordinary-I protected rows")
    BACK_ROWS=set(map(int,np.asarray(prot["actual_back_I_rows"],np.int64)))
    cap=float(cfg.get("move_cap_m",-1.0))
    if abs(cap-MOVE_CAP_M)>1e-15:raise ValueError("852 run config does not pin the ordinary-I 0.5 mm cap")
    report=BASE_RUN(config_path,Path(out_dir).resolve(),prune_receipt=prune_receipt)
    state_path=Path(out_dir).resolve()/"state.json"
    if state_path.is_file():
        st=json.loads(state_path.read_text())
        st["ordinary_inner_I_topology_repair"]=cfg["ordinary_inner_I_topology_repair"]
        st["producer_852"]={"path":str(Path(__file__).resolve()),"sha256":sha(__file__),
            "base_790":{"path":str(BASE790.resolve()),"sha256":BASE790_SHA},
            "source_proof":{"path":str(SOURCE807.resolve()),"sha256":SOURCE807_SHA},
            "independent_replay_789":{"path":str(REPLAY789.resolve()),"sha256":REPLAY789_SHA}}
        M._write_json(state_path,st)
        report["producer_852"]=st["producer_852"]
    return report


def independent_replay(config_path,run_dir,out_path=None):
    cfgpath=Path(config_path).resolve();rd=Path(run_dir).resolve()
    cfg=M.load_config(cfgpath);rep=json.loads((rd/"state.json").read_text())
    base=R789.independent_replay(cfgpath,rd,None)
    if base.get("status")!="PASS_INDEPENDENT_SOURCE_REPLAY":raise ValueError("789 replay did not pass")
    protected=set(map(int,cfg["protected"]))
    prot=M.load_pinned_npz(json.loads(cfgpath.read_text())["protected_surface_rows"],"852 protected rows")
    back=set(map(int,np.asarray(prot["actual_back_I_rows"],np.int64)))
    inputp=np.asarray(cfg["surface"]["points_world_m_f32"],np.float32)
    output=M.load_pinned_npz(rep["outputs"]["candidate-I-typed-current.npz"],"852 replayed candidate I")
    outputp=np.asarray(output["points_world_m_f32"],np.float32)
    if inputp.shape!=outputp.shape:raise ValueError("852 candidate point domain changed")
    if not np.array_equal(inputp[list(protected)].view(np.uint32),outputp[list(protected)].view(np.uint32)):
        raise ValueError("852 protected back/rim/route I coordinates changed")
    if not np.array_equal(np.asarray(output["source_input_vertex_quotient"],np.int64)[list(protected)],np.asarray(list(protected),np.int64)):
        raise ValueError("852 protected I point quotient is not identity")
    faces=np.asarray(cfg["surface"]["faces"],np.int64);cap=float(json.loads(cfgpath.read_text())["move_cap_m"])
    for op in rep.get("operation_records",[]):
        keep,drop=int(op["keep"]),int(op["drop"])
        if keep in protected or drop in protected or keep in back or drop in back:
            raise ValueError("852 operation uses a protected I endpoint")
        star=np.asarray(op["surface_parent_faces"],np.int64)
        if np.any(star<0) or np.any(star>=len(faces)):raise ValueError("852 operation face star out of range")
        if set(map(int,np.unique(faces[star].reshape(-1))))&protected:
            raise ValueError("852 operation complete surface star touches protected I rows")
        disp=float(np.linalg.norm(outputp[keep].astype(np.float64)-inputp[keep].astype(np.float64)))
        if disp>cap:raise ValueError("852 output anchor exceeds the 0.5 mm construction cap")
    result={**base,"ordinary_inner_I_topology_replay":"PASS",
        "actual_back_I_rows_fixed":True,"protected_rims_routes_fixed":True,
        "protected_row_count":len(protected),"back_row_count":len(back),
        "maximum_ordinary_I_anchor_step_m":max((float(op.get("displacement_m",0.0)) for op in rep.get("operation_records",[])),default=0.0),
        "maximum_ordinary_I_anchor_step_cap_m":cap,
        "source_807_input_proof":"PASS_REQUIRED_FOR_STATE_PREPARATION",
        "775_exact_all185":"REQUIRED_BEFORE_CANDIDATE_ACCEPTANCE",
        "physical_admission":False,"native_steps":0}
    if out_path:M._write_json(Path(out_path),result)
    return result


def _typed_parent(points,faces):
    a=M.A;n=len(faces)
    tuples=np.asarray([[1,1,0,1000+i,2000+i,0,0] for i in range(n)],np.int64).reshape((-1,7))
    ancestry=np.asarray([[i,2000+i,i] for i in range(n)],np.int64).reshape((-1,3))
    data={"points_world_m":np.asarray(points,np.float64),"points_world_m_f32":np.asarray(points,np.float32),
        "faces":np.asarray(faces,np.int64),"quality_star_face_output_to_input_row":np.arange(n,dtype=np.int64),
        "quality_star_patch_union_offsets":np.arange(n+1,dtype=np.int64),
        "face_ancestry_offsets":np.arange(n+1,dtype=np.int64)}
    for j,k in enumerate(a.T):data[k]=tuples[:,j]
    for j,k in enumerate(a.U):data[k]=tuples[:,j]
    for j,k in enumerate(a.A):data[k]=ancestry[:,j]
    return data


def fixture(out_dir):
    out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    inp=out/"input";inp.mkdir()
    # Two tets share face (1,2,3); collapse of edge (0,1) retires the thin
    # tet and maps the remaining tet to [0,2,3,4]. Row 4 is protected but outside
    # the complete I source-face star, so it remains an untouched anchor.
    points=np.asarray([[0,0,0],[1e-8,0,0],[0,1e-2,0],[0,0,1e-2],
        [.01,.005,.005],[.02,.01,0],[.02,0,.01]],np.float32)
    faces=np.asarray([[0,1,2],[1,0,3],[4,5,6]],np.int64)
    typed=_typed_parent(points,faces)
    i_path=inp/"surface-I.npz";np.savez_compressed(i_path,**typed)
    o_points=np.asarray([[.04,0,0],[.05,0,0],[.04,.01,0]],np.float32)
    o_path=inp/"surface-O.npz";np.savez_compressed(o_path,points_world_m=o_points.astype(np.float64),points_world_m_f32=o_points,faces=np.asarray([[0,1,2]],np.int64))
    d_points=np.asarray([[.06,0,0],[.07,0,0],[.06,.01,0]],np.float32)
    d_path=inp/"surface-D.npz";np.savez_compressed(d_path,points_world_m=d_points.astype(np.float64),points_world_m_f32=d_points,faces=np.asarray([[0,1,2]],np.int64))
    volume_refs={}
    for role in ("Skin","Fat"):
        vp=np.vstack((points[:4],points[4])).astype(np.float32)
        tt=M.T.m.orient(vp,np.asarray([[0,1,2,3],[1,2,3,4]],np.int64))
        vpath=inp/(role.lower()+".npz")
        np.savez_compressed(vpath,points_world_m=vp,tetrahedra=tt,material_indices=np.zeros(len(tt),np.int32),desired_boundary_faces=M.T.boundary(tt))
        volume_refs[role]={"path":str(vpath),"sha256":sha(vpath)}
    refs={r:{"path":str(p),"sha256":sha(p)} for r,p in (("O",o_path),("I",i_path),("D",d_path))}
    back=np.asarray([4],np.int64);protected_path=inp/"protected.npz"
    np.savez_compressed(protected_path,protected_surface_rows=back,route_anchor_rows=np.empty(0,np.int64),actual_back_I_rows=back)
    edge_path=inp/"edges.npz";np.savez_compressed(edge_path,candidate_edges=np.asarray([[0,1]],np.int64),undirected_edges=np.asarray([[0,1]],np.int64))
    cfg={"schema":"h706.coupled-anchor-collapse-run-config.v1","surface_I":refs["I"],"baseline_surfaces":refs,
        "volumes":volume_refs,"candidate_edges":{"path":str(edge_path),"sha256":sha(edge_path),"key":"candidate_edges"},
        "protected_surface_rows":{"path":str(protected_path),"sha256":sha(protected_path),"key":"protected_surface_rows"},
        "route_anchor_rows":{"path":str(protected_path),"sha256":sha(protected_path),"key":"route_anchor_rows"},
        "move_cap_m":MOVE_CAP_M,"max_operations":1,"time_limit_s":5.0,
        "ordinary_inner_I_topology_repair":{"enabled":True,"actual_back_I_rows_fixed":True,"maximum_displacement_m":MOVE_CAP_M}}
    cfgpath=inp/"run-config.json";M._write_json(cfgpath,cfg)
    state={"schema":"h706.ordinary-I-synthetic-fixture-state.v1","source_surfaces":refs,
        "Skin_current":volume_refs["Skin"],"Fat_current":volume_refs["Fat"],
        "physical_admission":False,"native_steps":0,"fixture_only":True,
        "exact_context":{"status":"NOT_RUN_SYNTHETIC_ONLY"}}
    statepath=inp/"state.json";M._write_json(statepath,state)
    loaded=M.load_config(cfgpath);cache={r:M.cache_volume(loaded["volumes"][r]) for r in ("Skin","Fat")}
    blocked=M.prepare_operation(loaded["surface"],cache,0,1,protected_surface_rows={2},move_cap_m=MOVE_CAP_M)
    if blocked.get("status")!="REJECT_852_PROTECTED_COMPLETE_STAR":raise AssertionError("protected-star control did not reject")
    run_dir=out/"candidate-run"
    report=run_from_config(cfgpath,run_dir)
    if report.get("accepted_operations")!=1:raise AssertionError("synthetic paired collapse was not accepted: "+str(report.get("rejections")))
    replay_path=out/"independent-replay.json";replay=independent_replay(cfgpath,run_dir,replay_path)
    if replay.get("status")!="PASS_INDEPENDENT_SOURCE_REPLAY" or replay.get("ordinary_inner_I_topology_replay")!="PASS":
        raise AssertionError("full 789 replay failed")
    if report.get("775_context_config",{}).get("path") is None:raise AssertionError("775 context config path missing")
    if report.get("serialization_fix_785",{}).get("journal",{}).get("path") is None:raise AssertionError("pre-serialization journal path missing")
    result={"schema":"h706.ordinary-I-852-end-to-end-control.v1","status":"PASS_SYNTHETIC_PRODUCER_JOURNAL_789_REPLAY",
        "fixture_state":{"path":str(statepath),"sha256":sha(statepath)},
        "fixture_run_config":{"path":str(cfgpath),"sha256":sha(cfgpath)},
        "candidate_state":{"path":str(run_dir/"state.json"),"sha256":sha(run_dir/"state.json")},
        "pre_serialization_journal":report["serialization_fix_785"]["journal"],
        "775_context_config":report["775_context_config"],
        "independent_replay":{"path":str(replay_path),"sha256":sha(replay_path),"status":replay["status"]},
        "protected_star_negative_control":blocked.get("status"),
        "accepted_operations":report["accepted_operations"],
        "skin_native_bad_before_after":[replay["quality"]["Skin"]["native_bad_before"],replay["quality"]["Skin"]["native_bad_after"]],
        "fat_native_bad_before_after":[replay["quality"]["Fat"]["native_bad_before"],replay["quality"]["Fat"]["native_bad_after"]],
        "exact_paired_volume_delta_integer":"0","all185_775":"NOT_RUN_SYNTHETIC_ONLY",
        "actual_697_217":"NOT_RUN_SYNTHETIC_ONLY","physical_admission":False,"native_steps":0}
    reportpath=out/"fixture-report.json";M._write_json(reportpath,result)
    result["fixture_report"]={"path":str(reportpath),"sha256":sha(reportpath)}
    return result


def main():
    ap=argparse.ArgumentParser(description="852 ordinary-I paired topology adapter; actual back/rim/routes fixed")
    ap.add_argument("--mode",choices=("fixture","prepare-state","run","replay","state-run"),required=True)
    ap.add_argument("--state",type=Path);ap.add_argument("--config",type=Path);ap.add_argument("--run-dir",type=Path)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--max-operations",type=int,default=32)
    ap.add_argument("--seconds",type=float,default=90);ap.add_argument("--max-edge-length-m",type=float,default=MAX_EDGE_M)
    ap.add_argument("--max-edges",type=int,default=100000);a=ap.parse_args()
    if a.mode=="fixture":
        print(json.dumps(fixture(a.output),indent=2));return
    if a.mode in ("prepare-state","state-run"):
        if a.state is None:ap.error("prepare-state/state-run requires --state")
        cfg,prep,preppath=prepare_state(a.state,a.output,max_operations=a.max_operations,
            time_limit_s=a.seconds,max_edge_length_m=a.max_edge_length_m,max_edges=a.max_edges)
        if a.mode=="prepare-state":
            print(json.dumps({"status":"PASS_PREFLIGHT_ONLY_NO_SOLVER","preparation":{"path":str(preppath),"sha256":sha(preppath)},
                "run_config":{"path":str(cfg),"sha256":sha(cfg)},"ordinary_inner_I_topology_repair":prep["ordinary_inner_I_topology_repair"],
                "geometry_solver":"NOT_RUN","physical_admission":False,"native_steps":0},indent=2));return
        run_dir=a.output.resolve()/"proposal"
        report=run_from_config(cfg,run_dir)
        wrapper={"schema":"h706.ordinary-I-852-state-run.v1","preparation":{"path":str(preppath),"sha256":sha(preppath)},
            "candidate_run":{"path":str(run_dir/"state.json"),"sha256":sha(run_dir/"state.json"),
                "accepted_operations":int(report.get("accepted_operations",0)),"status":report.get("status")},
            "independent_replay":"REQUIRED_RUN_789_BEFORE_ACCEPTANCE","candidate_exact_audit":"775_REQUIRED",
            "actual_697_217":"REQUIRED","physical_admission":False,"native_steps":0}
        M._write_json(a.output.resolve()/"state-wrapper.json",wrapper);print(json.dumps(wrapper,indent=2));return
    if a.mode=="run":
        if a.config is None:ap.error("run requires --config")
        r=run_from_config(a.config,a.output);print(json.dumps({k:v for k,v in r.items() if k!="operation_records"},indent=2,default=str));return
    if a.mode=="replay":
        if a.config is None or a.run_dir is None:ap.error("replay requires --config and --run-dir")
        r=independent_replay(a.config,a.run_dir,a.output);print(json.dumps(r,indent=2,default=str));return

if __name__=="__main__":main()
