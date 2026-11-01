#!/usr/bin/env python3
"""Direct-O current-433 adapter; legacy NHSKIN detail bridge stays disabled."""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from pathlib import Path
from typing import Any

HERE=Path(__file__).resolve().parent
BASE_PATH=HERE/"prepare_current_native_1025_launch_003.py"
DEFAULT_MANIFEST=HERE/"current-native-launch-1025-direct-o-005.template.json"
MODE="direct_o_without_legacy_detail_bridge_v1"
BRIDGE="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE"
CANONICAL="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE_CANONICAL_FACE_EXPORT"
TERMINAL=re.compile(r"resting_FEM_terminal_audit=passed[^\n]*expected_steps=(\d+)[^\n]*accepted_steps=(\d+)(?:\s|$)")

spec=importlib.util.spec_from_file_location("native1025_launch_003_pinned",BASE_PATH)
if spec is None or spec.loader is None: raise RuntimeError("cannot import pinned 1025 build adapter")
base=importlib.util.module_from_spec(spec); sys.modules[spec.name]=base; spec.loader.exec_module(base)


def detail_fields(route: Any)->set[str]:
    return {k for k in route.REQUIRED_ENV if k.startswith("DETAIL_") or k.startswith("COARSE_BOUNDARY_FACES_")}|{"DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV"}


def direct_env(route: Any,cfg: dict[str,Any],pins: dict[str,str])->dict[str,str]:
    env=cfg.get("native_preflight_env")
    if not isinstance(env,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in env.items()):
        raise ValueError("native_preflight_env must be a string-to-string map")
    forbidden=sorted(k for k in env if k.startswith("DETAIL_") or k.startswith("COARSE_BOUNDARY_FACES_"))
    if forbidden: raise ValueError(f"direct-O mode forbids legacy detail-bridge inputs: {forbidden}")
    required=route.REQUIRED_ENV-detail_fields(route)
    missing=required-set(env)
    if missing: raise ValueError(f"direct-O environment missing {sorted(missing)}")
    if env["MAXIMUM_MASS_TRANSPORT_DISTANCE_M"]!=route.MAX_MASS_TRANSPORT_M: raise ValueError("mass transport limit differs from native contract")
    if env["MAX_PARENT_REGISTRATION_DISTANCE_M"]!=route.MAX_NATIVE_PARENT_REGISTRATION_M: raise ValueError("parent registration limit differs from native contract")
    for key in ("OPERATIONAL_CONFORMING_REFINEMENT_ACTUAL_PROOF_RECEIPT","OPERATIONAL_CONFORMING_REFINEMENT_SELF_AUDIT_RECEIPT"):
        if env.get(key): raise ValueError(f"direct-O mode rejects stale refinement proof {key}")
    if env["PARENT_PACKET_CSV"]!=str(route.PARENT) or env["PARENT_LABELS_CSV"]!=str(route.PARENT)+".labels.csv":
        raise ValueError("parent packet or labels differ from pinned source")
    for field,path in (("PARENT_PACKET_SHA256",str(route.PARENT)),("PARENT_LABELS_SHA256",str(route.PARENT)+".labels.csv"),
                       ("PARENT_NODES_SHA256",str(route.PARENT)+".nodes.csv"),("PARENT_SOURCE_SHA256",str(route.PARENT)+".source.csv"),
                       ("PARENT_TETS_SHA256",str(route.PARENT)+".tets.csv")):
        if env[field]!=route.PINNED_PARENT[path]: raise ValueError(f"{field} differs from source-parent pin")
    pose=str(route.SOURCE330/"registered-skin-poses-100.json")
    if env["COORDINATE_FRAME_SHA256"]!=route.PINNED_PARENT[pose]: raise ValueError("coordinate frame differs from pinned source pose")
    pairs={k:v for k,v in base.path_pair_map(route).items()
           if k in required and not k.startswith(("DETAIL_","COARSE_BOUNDARY_FACES_"))}
    for key in required:
        if not env.get(key): raise ValueError(f"direct-O environment field {key} is empty")
    for path_key,hash_key in pairs.items():
        path,digest=env.get(path_key),env.get(hash_key)
        if not path or not digest or pins.get(path)!=digest:
            raise ValueError(f"direct-O input path/hash missing or mismatched: {path_key}")
    for key,value in env.items():
        if key.endswith(("_CSV","_NODE","_ELEMENTS","_ASSET","_RECEIPT")) and value not in pins:
            raise ValueError(f"direct-O input {key} is not pinned")
    return dict(env)


def apply_direct_env(route: Any,scene: dict[str,str],suffix: dict[str,str],source_only: bool,split: bool)->dict[str,str]:
    env=route.build_stage_env(scene,suffix,source_only=source_only,canonical_export=False,operational_split=split)
    for key in list(env):
        if key.startswith(route.PREFIX+"DETAIL_") or key.startswith(route.PREFIX+"COARSE_BOUNDARY_FACES_"): env.pop(key)
    env[BRIDGE]="0"; env[CANONICAL]="0"
    # The frozen 016 diagnostic baseline enables this multi-hundred-megabyte dump.
    # Direct-O source and bounded motion stages explicitly disable it.
    env["NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE"]="0"
    return env


def ctx(cfg: dict[str,Any]):
    if cfg.get("registered_surface_mode")!=MODE: raise ValueError("manifest must explicitly select direct-O/no-legacy-bridge mode")
    static=base.verify_static(cfg); route=base.route_module(cfg); pins=base.merged_pins(cfg)
    route.verify_input_pins(pins); suffix=direct_env(route,cfg,pins)
    geo,mass=base.validate_claims(route,cfg)
    return static,route,pins,suffix,geo,mass


def root_path(cfg:dict[str,Any],allow:bool=False)->Path:
    root=Path(cfg.get("output_root",""))
    if not root.is_absolute() or (root.exists() and not allow): raise ValueError("output_root must be absolute and new")
    return root


def readiness(path:Path)->dict[str,Any]:
    cfg=base.load_config(path); blockers=[]; static=None; route=None
    try:
        if cfg.get("registered_surface_mode")!=MODE: raise ValueError("direct-O mode missing")
        static=base.verify_static(cfg)
    except Exception as e: blockers.append(f"build/mode verification: {type(e).__name__}: {e}")
    try: route=base.route_module(cfg)
    except Exception as e: blockers.append(f"route 711 import: {type(e).__name__}: {e}")
    if route is not None:
        required=route.REQUIRED_ENV-detail_fields(route); env=cfg.get("native_preflight_env",{})
        missing=required-set(env)
        if missing: blockers.append(f"current 433 direct-O environment missing {len(missing)} fields")
        if not cfg.get("input_pins"): blockers.append("current 433 input pins are absent")
        else:
            try: direct_env(route,cfg,base.merged_pins(cfg)); route.verify_input_pins(base.merged_pins(cfg))
            except Exception as e: blockers.append(f"current 433 direct-O inputs: {type(e).__name__}: {e}")
        for name,claims in (("geometry_admission",route.REQUIRED_GEOMETRY_CLAIMS),("mass_admission",route.REQUIRED_MASS_CLAIMS)):
            receipt=cfg.get(name)
            if not isinstance(receipt,dict) or not receipt.get("path") or not receipt.get("sha256"):
                blockers.append(f"candidate-specific {name.replace('_',' ')} receipt is absent")
            else:
                try: route.check_claim_receipt(receipt,claims,name.replace("_"," "))
                except Exception as e: blockers.append(f"{name.replace('_',' ')} receipt: {type(e).__name__}: {e}")
    root=Path(cfg.get("output_root",""))
    if not root.is_absolute() or root.exists(): blockers.append("output_root must be absolute and new")
    return {"schema":"numi.human.current-native-direct-o-readiness.v1","registered_surface_mode":MODE,
            "status":"READY_FOR_DIRECT_O_SOURCE_PREFLIGHT" if not blockers else "WAITING_FOR_CURRENT_433_SOURCE_AND_ADMISSION_PINS",
            "blockers":blockers,"build_manifest_sha256":static["manifest_sha256"] if static else None,
            "binary_sha256":static["native_binary_sha256"] if static else None,"legacy_detail_bridge_required":False,
            "native_launch_performed":False,"native_runtime_executed":False,"gpu_executed":False,
            "physical_admission_claimed":False,
            "route_contract":{"source_preflight":"typed O/I/D import and direct-O one-hot render; no detail bridge",
                              "motion":"existing install-plus-motion path, bounded to one or two accepted steps",
                              "accepted_geometry_steps":"initial, step 1, and terminal when distinct; maximum eight",
                              "geometry_and_mass_gates_required":True}}


def plan_source(cfg_path:Path,out_path:Path)->dict[str,Any]:
    cfg=base.load_config(cfg_path); static,route,pins,suffix,geo,mass=ctx(cfg)
    for flag,candidate in cfg.get("argv_overrides",{}).items():
        if candidate not in pins: raise ValueError(f"argv override {flag} lacks exact input pin")
    root=root_path(cfg); steps=cfg.get("measurement_steps")
    if not isinstance(steps,int) or steps not in (1,2): raise ValueError("first measurement must be one or two steps")
    argv,physics,stale=base.safe_scene(cfg,static,route,root/"source-preflight",steps)
    env=apply_direct_env(route,physics,suffix,True,True)
    stage={"name":"source_preflight_direct_O_without_legacy_detail_bridge","argv":argv,"environment":env,
           "expected_runtime":False,"expected_accepted_steps":0,"expected_gpu":False,"cwd":static["baseline"].get("cwd")}
    plan={"schema":"numi.human.current-native-launch-1025-direct-o.v1","status":"DIRECT_O_SOURCE_PREFLIGHT_READY",
          "registered_surface_mode":MODE,"execution_performed":False,"native_runtime_executed":False,"GPU_executed":False,
          "manifest_path":str(cfg_path.resolve()),"manifest_sha256":base.json_digest(cfg),
          "adapter_sha256":base.sha_file(Path(__file__).resolve()),"base_adapter_sha256":base.sha_file(BASE_PATH),
          "build_manifest_sha256":static["manifest_sha256"],"route_711_sha256":static["route_sha256"],
          "native_binary":{"path":static["build"]["build"]["binary"],"sha256":static["native_binary_sha256"],
                           "repo_head":static["build"]["source"]["head"],
                           "dirty_fingerprint_sha256":static["build"]["source"]["dirty_entries_fingerprint_sha256"],
                           "dirty_entry_count":static["build"]["source"]["dirty_entry_count"]},
          "geometry_admission":geo,"mass_admission":mass,"input_pins":pins,
          "direct_O_source":{"surface_stable_identifier":suffix["SURFACE_O_STABLE_IDENTIFIER"],
                             "source_domain":suffix["ROLE_O_SOURCE_DOMAIN"],
                             "source_asset_path":suffix["ROLE_O_SOURCE_ASSET"],
                             "source_asset_sha256":suffix["ROLE_O_SOURCE_ASSET_SHA256"],
                             "surface_nodes_path":suffix["SURFACE_O_NODES_CSV"],
                             "surface_nodes_sha256":suffix["SURFACE_O_NODES_SHA256"],
                             "legacy_detail_bridge_required":False},
          "physics_environment_sha256":base.json_digest(physics),"removed_stale_diagnostic_environment_keys":stale,
          "baseline_argv_without_executable":argv[1:],"cwd":stage["cwd"],"stages":[stage],
          "motion":{"requested_accepted_step_count":steps,"first_measurement":True,
                    "measurement_scope":"release-motion integration check; not settled-rest qualification",
                    "accepted_geometry_steps":sorted(set((0,1,steps))),"maximum_mrv_captures":8,
                    "install_and_motion_are_same_probe_invocation":True,"postrun_admission_required":True,
                    "trace_files_required":list(base.TRACE_FILES),"terminal_fem_audit_required":True,
                    "full_fem_endpoint_state_dump_requested":False},
          "execution_policy":{"launch_performed":False,"native_runtime_executed":False,"gpu_executed":False}}
    base.write_once(out_path,plan); shell=out_path.with_suffix(".stage-01.sh")
    if shell.exists(): raise FileExistsError(f"refusing to overwrite source command: {shell}")
    shell.write_text(base.stage_shell(route,stage,stage["cwd"])); shell.chmod(0o755)
    return plan


def read_plan(cfg_path:Path,path:Path):
    cfg=base.load_config(cfg_path); static,route,pins,suffix,geo,mass=ctx(cfg); plan=base.read_json(path)
    if plan.get("registered_surface_mode")!=MODE: raise ValueError("saved plan is not direct-O")
    if plan.get("manifest_path")!=str(cfg_path.resolve()) or plan.get("manifest_sha256")!=base.json_digest(cfg):
        raise ValueError("saved plan does not bind exact manifest bytes")
    if plan.get("build_manifest_sha256")!=static["manifest_sha256"] or plan.get("native_binary",{}).get("sha256")!=static["native_binary_sha256"]:
        raise ValueError("saved plan has stale build identity")
    if plan.get("input_pins")!=pins or plan.get("geometry_admission")!=geo or plan.get("mass_admission")!=mass:
        raise ValueError("saved plan input/geometry/mass pins changed")
    if any(s.get("environment",{}).get(BRIDGE)!="0" or s.get("environment",{}).get(CANONICAL)!="0" for s in plan.get("stages",[])):
        raise ValueError("saved plan enables legacy detail bridge")
    if any(k.startswith(route.PREFIX+"DETAIL_") for s in plan.get("stages",[]) for k in s.get("environment",{})):
        raise ValueError("saved plan carries legacy detail-bridge inputs")
    return plan,cfg,route,{"static":static,"pins":pins,"env":suffix,"geo":geo,"mass":mass}


def parse_o_pass(stdout:str)->dict[str,int]:
    if "registered_volume_native_preflight=FAIL" in stdout: raise ValueError("native preflight has FAIL marker")
    line=next((x for x in stdout.splitlines() if "registered_volume_native_preflight=PASS" in x and "O_direct_render=typed_one_hot_prepared" in x),None)
    if line is None: raise ValueError("native output lacks typed direct-O PASS marker")
    m=re.search(r"\bO_direct_render_rows=(\d+)\b.*?\bO_direct_render_nodes=(\d+)\b",line)
    if not m: raise ValueError("direct-O PASS marker lacks row/node readback counts")
    rows,nodes=map(int,m.groups())
    if rows<=0 or nodes<=0: raise ValueError("direct-O readback counts must be positive")
    for token in ("physical_admission=0","scene_installation=0","simulation_steps=0"):
        if token not in line: raise ValueError(f"direct-O source pass lacks {token}")
    if "registered_volume_typed_O_canonical_face_export_source_only=PASS" in stdout:
        raise ValueError("native output unexpectedly executed detail bridge")
    return {"direct_O_rows":rows,"direct_O_volume_nodes":nodes}


def record_source(plan_path:Path,cfg_path:Path,stdout_path:Path,stderr_path:Path,code:int,out:Path)->dict[str,Any]:
    plan,cfg,route,ctxdata=read_plan(cfg_path,plan_path)
    if plan.get("status")!="DIRECT_O_SOURCE_PREFLIGHT_READY" or len(plan.get("stages",[]))!=1: raise ValueError("plan is not source-preflight ready")
    if code!=0: raise ValueError(f"source-only native preflight exited {code}")
    stdout=Path(stdout_path).read_text(errors="replace"); stderr=Path(stderr_path).read_text(errors="replace")
    direct=parse_o_pass(stdout)
    if "physical_admission=0" not in stdout or "accepted_steps=0" not in stdout: raise ValueError("source-only output must attest no admission and zero steps")
    if "registered_volume_native_preflight=FAIL" in stdout+stderr: raise ValueError("native source preflight failed")
    receipt={"schema":"numi.human.current-native-direct-o-source-preflight-receipt.v1",
             "status":"PASS_SOURCE_ONLY_DIRECT_O_NO_LEGACY_DETAIL_BRIDGE","plan_path":str(plan_path.resolve()),
             "plan_sha256":base.sha_file(plan_path),"input_manifest_sha256":plan["manifest_sha256"],
             "native_binary_sha256":plan["native_binary"]["sha256"],
             "geometry_admission_receipt_sha256":ctxdata["geo"]["receipt_sha256"],
             "mass_admission_receipt_sha256":ctxdata["mass"]["receipt_sha256"],
             "stdout_sha256":base.sha_file(Path(stdout_path)),"stderr_sha256":base.sha_file(Path(stderr_path)),
             "direct_O":{**plan["direct_O_source"],**direct,"native_typed_one_hot_readback_passed":True},
             "legacy_detail_bridge_enabled":False,"legacy_detail_map_required":False,
             "physical_admission":False,"native_runtime_executed":False,"gpu_executed":False,
             "accepted_steps":0,"source_preflight_executed":True}
    base.write_once(out,receipt); return receipt


def plan_motion(cfg_path:Path,source_plan:Path,source_receipt:Path,out:Path,steps:int)->dict[str,Any]:
    if steps not in (1,2): raise ValueError("first measurement is limited to one or two accepted steps")
    plan,cfg,route,ctxdata=read_plan(cfg_path,source_plan)
    if plan.get("status")!="DIRECT_O_SOURCE_PREFLIGHT_READY" or len(plan.get("stages",[]))!=1: raise ValueError("source plan not ready")
    sr=Path(source_receipt); receipt=base.read_json(sr)
    if receipt.get("status")!="PASS_SOURCE_ONLY_DIRECT_O_NO_LEGACY_DETAIL_BRIDGE" or receipt.get("input_manifest_sha256")!=plan["manifest_sha256"] or receipt.get("plan_sha256")!=base.sha_file(source_plan):
        raise ValueError("direct-O source receipt is stale or belongs to another plan")
    if receipt.get("legacy_detail_bridge_enabled") is not False or receipt.get("legacy_detail_map_required") is not False:
        raise ValueError("source receipt retains legacy detail bridge dependency")
    root=root_path(cfg,True)
    if not (root/"source-preflight").is_dir(): raise ValueError("source-only output directory missing")
    install=root/"install-motion"
    if install.exists(): raise FileExistsError(f"refusing to reuse motion output: {install}")
    argv,_,stale=base.safe_scene(cfg,ctxdata["static"],route,install,steps)
    scene=dict(base.read_json(Path(cfg["static_pins"]["baseline_launch_snapshot_path"]))["environment"])
    for key in stale: scene.pop(key,None)
    for key,name in base.BASELINE_OUTPUT_ENV.items():
        if key in scene: scene[key]=str(install/"diagnostics"/name)
    env=apply_direct_env(route,scene,ctxdata["env"],False,False)
    env.update({"NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT":"1",
                "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS":"1",
                "NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT":"1",
                "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT":"1",
                "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP":"1",
                "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP":str(steps),
                "NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE":"0",
                "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS":",".join(map(str,sorted(set((0,1,steps)))))})
    route.validate_full_scene_mode(argv[1:],env)
    stage={"name":"direct_O_install_and_bounded_release_motion","argv":argv,"environment":env,
           "expected_runtime":True,"expected_accepted_steps":steps,
           "accepted_geometry_steps":sorted(set((0,1,steps))),"cwd":plan["cwd"]}
    plan["status"]="DIRECT_O_MOTION_PLAN_READY_AFTER_SOURCE_PREFLIGHT"; plan["stages"].append(stage)
    plan["source_preflight_receipt_path"]=str(sr.resolve()); plan["source_preflight_receipt_sha256"]=base.sha_file(sr)
    plan["source_preflight_receipt"]=receipt; plan["execution_performed"]=False
    plan["native_runtime_executed"]=False; plan["GPU_executed"]=False
    plan["motion"]={"requested_accepted_step_count":steps,
                    "measurement_scope":"release-motion integration check; not settled-rest qualification",
                    "accepted_geometry_steps":sorted(set((0,1,steps))),"maximum_mrv_captures":8,
                    "install_and_motion_are_same_probe_invocation":True,"postrun_admission_required":True,
                    "trace_files_required":list(base.TRACE_FILES),"terminal_fem_audit_required":True,
                    "full_fem_endpoint_state_dump_requested":False,"viewer_path_required":True,
                    "persistent_stand_trace_forbidden_with_continuum":True,"legacy_detail_bridge_enabled":False}
    base.write_once(out,plan); sh=out.with_suffix(".stage-02.sh")
    if sh.exists(): raise FileExistsError(f"refusing to overwrite motion command: {sh}")
    sh.write_text(base.stage_shell(route,stage,stage["cwd"])); sh.chmod(0o755)
    return plan


def record_motion(plan_path:Path,cfg_path:Path,stdout_path:Path,stderr_path:Path,code:int,out:Path):
    plan,cfg,route,ctxdata=read_plan(cfg_path,plan_path)
    if plan.get("status")!="DIRECT_O_MOTION_PLAN_READY_AFTER_SOURCE_PREFLIGHT" or len(plan.get("stages",[]))!=2: raise ValueError("direct-O motion plan is not ready")
    sr=Path(plan["source_preflight_receipt_path"])
    if base.sha_file(sr)!=plan.get("source_preflight_receipt_sha256"): raise ValueError("direct-O source receipt changed")
    source=base.read_json(sr)
    if source!=plan.get("source_preflight_receipt") or source.get("status")!="PASS_SOURCE_ONLY_DIRECT_O_NO_LEGACY_DETAIL_BRIDGE": raise ValueError("embedded direct-O source receipt stale")
    if code!=0: raise ValueError(f"native direct-O motion exited {code}")
    stdout=Path(stdout_path).read_text(errors="replace"); stderr=Path(stderr_path).read_text(errors="replace")
    if "whole_body_skin_fat_registered_volume_install=PASS" not in stdout: raise ValueError("registered-volume install PASS missing")
    if "whole_body_skin_fat_registered_volume_install=FAIL" in stdout+stderr: raise ValueError("registered-volume install FAIL present")
    direct=parse_o_pass(stdout); requested=int(plan["motion"]["requested_accepted_step_count"])
    if (requested,requested) not in [tuple(map(int,x)) for x in TERMINAL.findall(stdout)]: raise ValueError("terminal FEM audit lacks requested accepted steps")
    if "resting_FEM_terminal_audit=failed" in stdout+stderr: raise ValueError("terminal FEM audit failed")
    outdir=Path(plan["stages"][1]["argv"][4]); traces={}
    for name in base.TRACE_FILES:
        path=outdir/name
        if not path.is_file() or path.stat().st_size==0: raise ValueError(f"accepted-state trace missing: {path}")
        traces[str(path)]={"sha256":base.sha_file(path),"bytes":path.stat().st_size}
    frames={}; accepted=outdir/"accepted-geometry"
    for step in sorted(set((0,1,requested))):
        pack=accepted/f"step-{step}.mrvpack"; rp=accepted/f"step-{step}.receipt.json"
        if not pack.is_file() or not rp.is_file(): raise ValueError(f"accepted MRV frame/receipt missing at {step}")
        mr=base.read_json(rp); digest=base.sha_file(pack)
        if mr.get("accepted_step")!=step or mr.get("physical_endpoint")!="accepted" or mr.get("surface_audit_endpoint")!="passed" or mr.get("pack_file_sha256")!=digest:
            raise ValueError(f"accepted MRV receipt invalid at {step}")
        frames[str(step)]={"pack_path":str(pack),"pack_sha256":digest,"receipt_path":str(rp),"receipt_sha256":base.sha_file(rp)}
    route_path=out.with_name(out.stem+".route.json")
    if route_path.exists(): raise FileExistsError(f"refusing to overwrite motion receipt: {route_path}")
    route_receipt={"schema":"numi.human.current-native-direct-o-motion-receipt.v1",
                   "status":"PASS_DIRECT_O_INSTALL_AND_ACCEPTED_STATE_MOTION",
                   "plan_sha256":base.sha_file(plan_path),"stdout_sha256":base.sha_file(Path(stdout_path)),
                   "stderr_sha256":base.sha_file(Path(stderr_path)),"native_exit_code":code,
                   "native_binary_sha256":plan["native_binary"]["sha256"],
                   "geometry_admission_receipt_sha256":ctxdata["geo"]["receipt_sha256"],
                   "mass_admission_receipt_sha256":ctxdata["mass"]["receipt_sha256"],
                   "source_preflight_receipt_sha256":plan["source_preflight_receipt_sha256"],
                   "registered_volume_install_pass":True,"direct_O":direct,"legacy_detail_bridge_enabled":False,
                   "requested_accepted_steps":requested,"observed_accepted_steps":requested,
                   "native_runtime_executed":True,"gpu_executed":True,
                   "physical_admission_claimed":False,"long_horizon_qualification_claimed":False}
    base.write_once(route_path,route_receipt)
    observer={"schema":"numi.human.current-native-direct-o-measurement-observer-receipt.v1",
              "status":"PASS_BOUNDED_DIRECT_O_TRACES_AND_MRV","plan_sha256":base.sha_file(plan_path),
              "stdout_sha256":base.sha_file(Path(stdout_path)),"stderr_sha256":base.sha_file(Path(stderr_path)),
              "requested_accepted_steps":requested,"measurement_scope":"release-motion integration check; not settled-rest qualification",
              "direct_O":direct,"accepted_trace_files":traces,"accepted_geometry_frames":frames,
              "terminal_fem_audit_verified":True,"full_fem_endpoint_state_dump_requested":False,
              "native_runtime_executed":True,"gpu_executed":True,"physical_admission_claimed":False,
              "long_horizon_qualification_claimed":False,
              "install_route_receipt":{"path":str(route_path),"sha256":base.sha_file(route_path),"status":route_receipt["status"]}}
    base.write_once(out,observer); return observer,route_receipt


def main()->int:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    for name in ("readiness","plan-source"):
        q=sub.add_parser(name); q.add_argument("--manifest",default=str(DEFAULT_MANIFEST)); q.add_argument("--out",required=True)
    q=sub.add_parser("record-source"); q.add_argument("--manifest",default=str(DEFAULT_MANIFEST)); q.add_argument("--plan",required=True); q.add_argument("--stdout",required=True); q.add_argument("--stderr",required=True); q.add_argument("--exit-code",type=int,required=True); q.add_argument("--out",required=True)
    q=sub.add_parser("plan-measurement"); q.add_argument("--manifest",default=str(DEFAULT_MANIFEST)); q.add_argument("--source-plan",required=True); q.add_argument("--source-receipt",required=True); q.add_argument("--steps",type=int,choices=(1,2),default=2); q.add_argument("--out",required=True)
    q=sub.add_parser("record-measurement"); q.add_argument("--manifest",default=str(DEFAULT_MANIFEST)); q.add_argument("--plan",required=True); q.add_argument("--stdout",required=True); q.add_argument("--stderr",required=True); q.add_argument("--exit-code",type=int,required=True); q.add_argument("--out",required=True)
    a=p.parse_args()
    try:
        if a.cmd=="readiness":
            r=readiness(Path(a.manifest)); base.write_once(Path(a.out),r); print(json.dumps({"status":r["status"],"report":a.out,"blockers":r["blockers"],"native_launch_performed":False},sort_keys=True))
        elif a.cmd=="plan-source":
            r=plan_source(Path(a.manifest),Path(a.out)); print(json.dumps({"status":r["status"],"plan":a.out,"command":str(Path(a.out).with_suffix(".stage-01.sh")),"native_launch_performed":False},sort_keys=True))
        elif a.cmd=="record-source":
            r=record_source(Path(a.plan),Path(a.manifest),Path(a.stdout),Path(a.stderr),a.exit_code,Path(a.out)); print(json.dumps({"status":r["status"],"receipt":a.out},sort_keys=True))
        elif a.cmd=="plan-measurement":
            r=plan_motion(Path(a.manifest),Path(a.source_plan),Path(a.source_receipt),Path(a.out),a.steps); print(json.dumps({"status":r["status"],"plan":a.out,"command":str(Path(a.out).with_suffix(".stage-02.sh")),"native_launch_performed":False},sort_keys=True))
        else:
            r,_=record_motion(Path(a.plan),Path(a.manifest),Path(a.stdout),Path(a.stderr),a.exit_code,Path(a.out)); print(json.dumps({"status":r["status"],"receipt":a.out},sort_keys=True))
        return 0
    except Exception as e:
        print(f"BLOCKED: {type(e).__name__}: {e}",file=sys.stderr); return 2


if __name__=="__main__": raise SystemExit(main())

