#!/usr/bin/env python3
"""Versioned direct-O adapter with source-geometry-scoped receipt validation."""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

HERE=Path(__file__).resolve().parent
BASE_PATH=HERE.parent/"prepare_current_native_1025_launch_003.py"
DEFAULT_MANIFEST=HERE/"current-native-launch-1025-direct-o-006.template.json"
MODE="direct_o_without_legacy_detail_bridge_v1"
BRIDGE="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE"
CANONICAL="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE_CANONICAL_FACE_EXPORT"
TERMINAL=re.compile(r"resting_FEM_terminal_audit=passed[^\n]*expected_steps=(\d+)[^\n]*accepted_steps=(\d+)(?:\s|$)")

spec=importlib.util.spec_from_file_location("native1025_launch_003_pinned",BASE_PATH)
if spec is None or spec.loader is None: raise RuntimeError("cannot import pinned 1025 build adapter")
base=importlib.util.module_from_spec(spec); sys.modules[spec.name]=base; spec.loader.exec_module(base)

ROUTE_PATH=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-launch-route-preparation-712-source-geometry-scope/prepare_h706_native_route_712.py")
ROUTE_SHA256="991eab82488543f71d3eaf615df79d26fcb4eca1e4f6b87a2d89c4ea5d121c4e"


def route_module(cfg: dict[str,Any]):
    pin=cfg.get("scoped_geometry_route")
    if not isinstance(pin,dict) or pin.get("path")!=str(ROUTE_PATH) or pin.get("sha256")!=ROUTE_SHA256:
        raise ValueError("manifest must pin the exact versioned source-geometry receipt reader")
    if base.sha_file(ROUTE_PATH)!=ROUTE_SHA256:
        raise ValueError("versioned source-geometry receipt reader hash changed")
    spec=importlib.util.spec_from_file_location("pinned_route_712_source_geometry",ROUTE_PATH)
    if spec is None or spec.loader is None: raise RuntimeError("cannot import versioned source-geometry receipt reader")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_legacy_current_mass_binding(route: Any,receipt: dict[str,Any],cfg: dict[str,Any])->None:
    """Bind the passing mass result to this exact 433 child and frozen 711 cap."""
    env=cfg.get("native_preflight_env")
    if not isinstance(env,dict): raise ValueError("current mass binding requires the registered-volume environment")
    plan=receipt.get("plan")
    if not isinstance(plan,dict): raise ValueError("mass receipt lacks the current conservative transport plan")
    if plan.get("candidate_sha256")!=env.get("OPERATIONAL_CANDIDATE_VOLUME_SHA256"):
        raise ValueError("mass result candidate SHA differs from the current 433 volume")
    if plan.get("source_parent_packet_sha256")!=env.get("PARENT_PACKET_SHA256"):
        raise ValueError("mass result source-parent SHA differs from the pinned source330 packet")
    child_hashes=plan.get("child_volume_sha256")
    expected_child={
        "mesh.node":env.get("TETGEN_NODE_SHA256"),
        "mesh.ele":env.get("TETGEN_ELEMENTS_SHA256"),
        "child.labels.csv":env.get("CHILD_LABELS_SHA256"),
    }
    if not isinstance(child_hashes,dict) or child_hashes!=expected_child or any(
            not isinstance(v,str) or not re.fullmatch(r"[0-9a-f]{64}",v) for v in expected_child.values()):
        raise ValueError("mass result child node/element/label hashes differ from the current native import")
    if not isinstance(plan.get("binary_sha256"),str) or not re.fullmatch(r"[0-9a-f]{64}",plan["binary_sha256"]):
        raise ValueError("mass result lacks its exact transport executable hash")
    if plan.get("density_bounds_unchanged") is not True or plan.get("physical_admission") is not False:
        raise ValueError("mass result must preserve density bounds and remain nonphysical")
    try:
        cap=Decimal(str(plan.get("predeclared_transport_envelope_m")))
        max_distance=Decimal(str(plan.get("maximum_declared_source_centroid_distance_m")))
        allowed=Decimal(route.MAX_MASS_TRANSPORT_M)
    except (InvalidOperation,ValueError):
        raise ValueError("mass result transport envelope is not a finite decimal") from None
    if not cap.is_finite() or not max_distance.is_finite() or cap<=0 or max_distance<0:
        raise ValueError("mass result transport distances must be finite and nonnegative")
    if cap>allowed:
        raise ValueError("mass result exceeds the unchanged 0.019353 m transport bound")
    if max_distance>cap:
        raise ValueError("mass result transport envelope is smaller than its declared source distance")
    factors=plan.get("sensitivity_factors")
    if not isinstance(factors,list) or not any(type(f) in (int,float) and Decimal(str(f))==Decimal("1.0") for f in factors):
        raise ValueError("mass result lacks the predeclared factor-1.0 trial")
    trials=receipt.get("trials")
    passing=[t for t in trials if isinstance(t,dict) and type(t.get("factor")) in (int,float)
             and Decimal(str(t["factor"]))==Decimal("1.0")] if isinstance(trials,list) else []
    if len(passing)!=1 or type(passing[0].get("exit_code")) is not int or passing[0]["exit_code"]!=0:
        raise ValueError("current mass result must contain exactly one successful factor-1.0 trial")
    try: trial_bound=Decimal(str(passing[0].get("bound_m")))
    except (InvalidOperation,ValueError):
        raise ValueError("factor-1.0 mass trial lacks a finite bound") from None
    if not trial_bound.is_finite() or trial_bound!=cap:
        raise ValueError("factor-1.0 trial bound differs from its predeclared envelope")



FRACTIONAL_MASS_SCHEMA="numi.human.current-fractional-semantic-mass-mapping.v1"
FRACTIONAL_MASS_TOP_KEYS={"schema","status","physical_admission","native_steps","allocation","claims","transport_distance_limit_m","closure_tolerances","input_sha256","output_sha256","candidate","source330","solver","replay","semantic_group_rows","source_donor_rows","density"}
FRACTIONAL_MASS_OUTPUT_KEYS={"flow","semantic_groups","source_donors","carrier_trace"}
FRACTIONAL_MASS_INPUT_KEYS={"current_state","candidate_skin_volume","candidate_fat_volume","child_skin_node","child_skin_elements","child_skin_labels","child_fat_node","child_fat_elements","child_fat_labels","source330_mass","source330_labels","source330_nodes","source330_tets","p1_skin_support","p1_fat_support","carrier_skin_packet","carrier_fat_packet","projection_evidence","projection_report","source_centroid_slots","carrier_witness_skin","carrier_witness_fat"}
FRACTIONAL_CLOSURE_TOLERANCES={"mass_absolute_kg":1e-13,"mass_relative":1e-10,"measure_absolute_m3":1e-15,"carrier_barycentric_sum_absolute":2e-12,"carrier_first_moment_m":1e-11}

H706_FRACTIONAL_PREFIX="NUMI_H706_FRACTIONAL_"
H706_FRACTIONAL_PATH_HASH_PAIRS={
    "P1_SKIN_NPZ":"P1_SKIN_SHA256","P1_FAT_NPZ":"P1_FAT_SHA256",
    "SOURCE330_NODES":"SOURCE330_NODES_SHA256","SOURCE330_TETS":"SOURCE330_TETS_SHA256",
    "CARRIERS_SKIN_NPZ":"CARRIERS_SKIN_SHA256","CARRIERS_FAT_NPZ":"CARRIERS_FAT_SHA256",
    "SOURCE_CENTROID_CSV":"SOURCE_CENTROID_SHA256","EXACT_PROJECTION_CSV":"EXACT_PROJECTION_SHA256",
    "CARRIER_WITNESS_SKIN_CSV":"CARRIER_WITNESS_SKIN_SHA256",
    "CARRIER_WITNESS_FAT_CSV":"CARRIER_WITNESS_FAT_SHA256",
}
H706_FRACTIONAL_IDENTITY_HASHES={
    "STATE_SHA256","GEOMETRY_SKIN_SHA256","VOLUME_SKIN_SHA256",
    "GEOMETRY_FAT_SHA256","VOLUME_FAT_SHA256","SOURCE330_MASS_SHA256",
}
H706_FRACTIONAL_ENV_KEYS={"ENABLE",*H706_FRACTIONAL_PATH_HASH_PAIRS,
    *H706_FRACTIONAL_PATH_HASH_PAIRS.values(),*H706_FRACTIONAL_IDENTITY_HASHES}
H706_FRACTIONAL_PIN_KEYS=set(H706_FRACTIONAL_PATH_HASH_PAIRS)|H706_FRACTIONAL_IDENTITY_HASHES

def verify_h706_fractional_env(cfg: dict[str,Any])->dict[str,str]:
    """Authenticate every opt-in app path/hash; never drop or rewrite a field."""
    env=cfg.get("h706_fractional_env")
    if not isinstance(env,dict) or set(env)!=H706_FRACTIONAL_ENV_KEYS:
        missing=sorted(H706_FRACTIONAL_ENV_KEYS-set(env or {}))
        extra=sorted(set(env or {})-H706_FRACTIONAL_ENV_KEYS)
        raise ValueError(f"H706 fractional environment schema mismatch: missing={missing}, extra={extra}")
    if any(not isinstance(k,str) or not isinstance(v,str) for k,v in env.items()) or env["ENABLE"]!="1":
        raise ValueError("H706 fractional environment must be complete string values with ENABLE=1")
    source=cfg.get("h706_fractional_environment_source")
    if not isinstance(source,dict) or set(source)!={"path","resolved_path","sha256","argv_sha256"}:
        raise ValueError("H706 fractional environment must bind its exact argv source manifest")
    source_path=Path(source["path"]) if isinstance(source.get("path"),str) else Path()
    if not source_path.is_absolute() or source["resolved_path"]!=str(source_path.resolve()):
        raise ValueError("H706 fractional argv source path binding changed")
    if not source_path.is_file() or base.sha_file(source_path)!=source["sha256"]:
        raise ValueError("H706 fractional argv source manifest hash mismatch")
    source_doc=base.read_json(source_path); source_argv=source_doc.get("argv")
    if not isinstance(source_argv,list) or base.json_digest(source_argv)!=source["argv_sha256"]:
        raise ValueError("H706 fractional argv source digest mismatch")
    source_env={}
    source_cli_only={}
    for arg in source_argv:
        if isinstance(arg,str) and arg.startswith(H706_FRACTIONAL_PREFIX):
            if "=" not in arg: raise ValueError("H706 fractional argv item is malformed")
            key,value=arg[len(H706_FRACTIONAL_PREFIX):].split("=",1)
            if key=="OUTPUT_DIR":
                if key in source_cli_only: raise ValueError("duplicate H706 fractional CLI-only OUTPUT_DIR")
                if not Path(value).is_absolute(): raise ValueError("H706 fractional CLI-only OUTPUT_DIR must be absolute")
                source_cli_only[key]=value
                continue
            if key in source_env: raise ValueError(f"duplicate H706 fractional argv key: {key}")
            source_env[key]=value
    if source_env!=env:
        raise ValueError("H706 fractional env differs from its actual source argv")
    path_pins=cfg.get("h706_fractional_path_pins")
    if not isinstance(path_pins,dict) or set(path_pins)!=H706_FRACTIONAL_PIN_KEYS:
        raise ValueError("H706 fractional path-pins must cover all ten input files and six identity hashes")
    for path_key,hash_key in H706_FRACTIONAL_PATH_HASH_PAIRS.items():
        raw=env[path_key]; digest=env[hash_key]; pin=path_pins[path_key]
        if not Path(raw).is_absolute() or not isinstance(pin,dict) or set(pin)!={"path","resolved_path","sha256"}:
            raise ValueError(f"H706 fractional input {path_key} lacks its exact path binding")
        path=Path(raw)
        resolved=str(path.resolve())
        if pin["path"]!=raw or pin["resolved_path"]!=resolved or pin["sha256"]!=digest:
            raise ValueError(f"H706 fractional input {path_key} path/SHA binding changed")
        if not path.is_file() or base.sha_file(path)!=digest:
            raise ValueError(f"H706 fractional input {path_key} file hash mismatch")
    for hash_key in H706_FRACTIONAL_IDENTITY_HASHES:
        pin=path_pins[hash_key]; digest=env[hash_key]
        if not isinstance(pin,dict) or set(pin)!={"path","resolved_path","sha256"} or not isinstance(pin["path"],str) or not Path(pin["path"]).is_absolute():
            raise ValueError(f"H706 fractional identity {hash_key} lacks its source path")
        path=Path(pin["path"]); resolved=str(path.resolve())
        if pin["resolved_path"]!=resolved or pin["sha256"]!=digest or not path.is_file() or base.sha_file(path)!=digest:
            raise ValueError(f"H706 fractional identity {hash_key} source path/SHA mismatch")
    native=cfg.get("native_preflight_env")
    if not isinstance(native,dict): raise ValueError("H706 fractional env requires the existing native preflight env")
    for frac_key,native_key in (("SOURCE330_MASS_SHA256","PARENT_PACKET_SHA256"),
                                ("SOURCE330_NODES_SHA256","PARENT_NODES_SHA256"),
                                ("SOURCE330_TETS_SHA256","PARENT_TETS_SHA256")):
        if env[frac_key]!=native.get(native_key):
            raise ValueError(f"H706 fractional {frac_key} differs from exact native parent pin")
    return dict(env)

def _fractional_number(value: Any,label: str)->Decimal:
    if isinstance(value,bool) or not isinstance(value,(int,float,str,Decimal)):
        raise ValueError(f"fractional mass {label} must be finite")
    try: number=Decimal(str(value))
    except (InvalidOperation,ValueError): raise ValueError(f"fractional mass {label} must be finite") from None
    if not number.is_finite(): raise ValueError(f"fractional mass {label} must be finite")
    return number

def _fractional_path_hash(entry: Any,label: str)->tuple[str,str]:
    if not isinstance(entry,dict) or set(entry)!={"path","sha256"}:
        raise ValueError(f"fractional mass {label} must be exact path/SHA object")
    raw,digest=entry["path"],entry["sha256"]
    if not isinstance(raw,str) or not Path(raw).is_absolute() or not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest):
        raise ValueError(f"fractional mass {label} has malformed path/SHA")
    path=Path(raw)
    if not path.is_file() or base.sha_file(path)!=digest:
        raise ValueError(f"fractional mass {label} path/hash mismatch")
    return str(path.resolve()),digest

def verify_fractional_mass_contract(route: Any,receipt: dict[str,Any])->None:
    """Strictly verify the frozen 1047 source-only fractional mapping contract."""
    if set(receipt)!=FRACTIONAL_MASS_TOP_KEYS:
        raise ValueError(f"fractional mass top-level schema mismatch: missing={sorted(FRACTIONAL_MASS_TOP_KEYS-set(receipt))}, extra={sorted(set(receipt)-FRACTIONAL_MASS_TOP_KEYS)}")
    if receipt["schema"]!=FRACTIONAL_MASS_SCHEMA or receipt["status"]!="PASS_SOURCE_ONLY_REPLAY":
        raise ValueError("fractional mass receipt is not a source-only replay")
    if receipt["physical_admission"] is not False or receipt["native_steps"]!=0 or receipt["allocation"]!="NOT_NATIVE_EXECUTED":
        raise ValueError("fractional mass receipt must remain nonphysical and unexecuted")
    if set(receipt["claims"])!=set(route.REQUIRED_MASS_CLAIMS):
        raise ValueError("fractional mass receipt must carry exactly the five required claims")
    inputs=receipt["input_sha256"]
    if not isinstance(inputs,dict) or set(inputs)!=FRACTIONAL_MASS_INPUT_KEYS:
        raise ValueError("fractional mass input map differs from frozen 22-file contract")
    input_refs={key:_fractional_path_hash(value,f"input_sha256.{key}") for key,value in inputs.items()}
    outputs=receipt["output_sha256"]
    if not isinstance(outputs,dict) or set(outputs)!=FRACTIONAL_MASS_OUTPUT_KEYS:
        raise ValueError("fractional mass output hash map differs from frozen four-output contract")
    output_refs={}
    for key,value in outputs.items():
        if not isinstance(key,str) or not key: raise ValueError("fractional mass output name is malformed")
        output_refs[key]=_fractional_path_hash(value,f"output_sha256.{key}")
    candidate=receipt["candidate"]
    candidate_keys={"current_state","skin_volume","fat_volume","child_mesh_sha256","region_label_sha256","coordinate_frame_sha256","child_node_count","child_tetrahedron_count"}
    if not isinstance(candidate,dict) or set(candidate)!=candidate_keys:
        raise ValueError("fractional mass candidate schema mismatch")
    state_ref=_fractional_path_hash(candidate["current_state"],"candidate.current_state")
    if state_ref!=input_refs["current_state"] or _fractional_path_hash(candidate["skin_volume"],"candidate.skin_volume")!=input_refs["candidate_skin_volume"] or _fractional_path_hash(candidate["fat_volume"],"candidate.fat_volume")!=input_refs["candidate_fat_volume"]:
        raise ValueError("fractional mass candidate identity differs from input map")
    for key in ("child_mesh_sha256","region_label_sha256","coordinate_frame_sha256"):
        if not isinstance(candidate[key],str) or not re.fullmatch(r"[0-9a-f]{64}",candidate[key]): raise ValueError(f"fractional mass candidate {key} malformed")
    for key in ("child_node_count","child_tetrahedron_count"):
        if type(candidate[key]) is not int or candidate[key]<=0: raise ValueError(f"fractional mass candidate {key} must be positive integer")
    source=receipt["source330"]
    source_keys={"mass_packet","labels","nodes","tets","parent_source_contributions","source_mass_kg","parent_count","source_donor_contribution_count"}
    if not isinstance(source,dict) or set(source)!=source_keys: raise ValueError("fractional mass source330 schema mismatch")
    for field,key in (("mass_packet","source330_mass"),("labels","source330_labels"),("nodes","source330_nodes"),("tets","source330_tets")):
        if _fractional_path_hash(source[field],f"source330.{field}")!=input_refs[key]: raise ValueError(f"fractional mass source330 {field} differs from input map")
    source_mass=_fractional_number(source["source_mass_kg"],"source330.source_mass_kg")
    if source_mass<=0 or type(source["parent_count"]) is not int or source["parent_count"]<=0 or type(source["source_donor_contribution_count"]) is not int or source["source_donor_contribution_count"]<=0: raise ValueError("fractional mass source330 totals/counts must be positive")
    solver=receipt["solver"]
    solver_keys={"method","status","optimality_certified","transport_receipt_sha256","mapping_path_sha256","elapsed_s","transport_edge_count","augmentation_count","scale_factor","postsolve_rescaling"}
    if not isinstance(solver,dict) or set(solver)!=solver_keys: raise ValueError("fractional mass solver schema mismatch")
    if solver["method"]!="deterministic_dinic_feasibility" or solver["status"]!="PASS_FEASIBLE_NOT_OPTIMAL" or solver["optimality_certified"] is not False or solver["postsolve_rescaling"] is not False or _fractional_number(solver["scale_factor"],"solver.scale_factor")!=Decimal("1.0"): raise ValueError("fractional mass solver method/status/scaling invalid")
    for key in ("transport_receipt_sha256","mapping_path_sha256"):
        if not isinstance(solver[key],str) or not re.fullmatch(r"[0-9a-f]{64}",solver[key]): raise ValueError(f"fractional mass solver {key} malformed")
    if _fractional_number(solver["elapsed_s"],"solver.elapsed_s")<0 or type(solver["transport_edge_count"]) is not int or solver["transport_edge_count"]<=0 or type(solver["augmentation_count"]) is not int or solver["augmentation_count"]<0: raise ValueError("fractional mass solver counters invalid")
    limit=_fractional_number(receipt["transport_distance_limit_m"],"transport_distance_limit_m")
    if limit!=Decimal("0.019353") or limit!=Decimal(route.MAX_MASS_TRANSPORT_M): raise ValueError("fractional mass transport bound changed")
    tolerances=receipt["closure_tolerances"]
    if not isinstance(tolerances,dict) or set(tolerances)!=set(FRACTIONAL_CLOSURE_TOLERANCES): raise ValueError("fractional mass closure tolerance schema mismatch")
    for key,expected in FRACTIONAL_CLOSURE_TOLERANCES.items():
        if _fractional_number(tolerances[key],f"closure_tolerances.{key}")!=Decimal(str(expected)): raise ValueError(f"fractional mass closure tolerance {key} changed")
    replay=receipt["replay"]
    replay_keys={"status","result_sha256","max_parent_mass_closure_abs_kg","max_parent_measure_closure_abs_m3","max_source_donor_closure_abs_kg","max_child_fraction_sum_error","max_transport_distance_used_m","flow_array_equality"}
    if not isinstance(replay,dict) or set(replay)!=replay_keys or replay["status"]!="PASS_DETERMINISTIC_REPLAY" or replay["flow_array_equality"] is not True: raise ValueError("fractional mass deterministic replay failed")
    if not isinstance(replay["result_sha256"],str) or not re.fullmatch(r"[0-9a-f]{64}",replay["result_sha256"]): raise ValueError("fractional mass replay result SHA malformed")
    mass_limit=Decimal(str(FRACTIONAL_CLOSURE_TOLERANCES["mass_absolute_kg"]))+Decimal(str(FRACTIONAL_CLOSURE_TOLERANCES["mass_relative"]))*source_mass
    if _fractional_number(replay["max_parent_mass_closure_abs_kg"],"replay.parent_mass")>mass_limit or _fractional_number(replay["max_source_donor_closure_abs_kg"],"replay.source_donor")>mass_limit: raise ValueError("fractional mass closure exceeds tolerance")
    if _fractional_number(replay["max_parent_measure_closure_abs_m3"],"replay.parent_measure")>Decimal(str(FRACTIONAL_CLOSURE_TOLERANCES["measure_absolute_m3"])): raise ValueError("fractional measure closure exceeds tolerance")
    if _fractional_number(replay["max_child_fraction_sum_error"],"replay.child_fraction")>Decimal(str(FRACTIONAL_CLOSURE_TOLERANCES["carrier_barycentric_sum_absolute"])): raise ValueError("fractional child fraction sum exceeds tolerance")
    distance=_fractional_number(replay["max_transport_distance_used_m"],"replay.max_transport_distance")
    if distance<0 or distance>limit: raise ValueError("fractional mass replay exceeds predeclared transport bound")
    for name in ("semantic_group_rows","source_donor_rows"):
        rows=receipt[name]
        if not isinstance(rows,dict) or type(rows.get("count")) is not int or rows["count"]<=0: raise ValueError(f"fractional mass {name} count invalid")
        if _fractional_path_hash(rows.get("csv"),f"{name}.csv") not in output_refs.values(): raise ValueError(f"fractional mass {name} CSV not bound as output")
    groups=receipt["semantic_group_rows"]
    if groups.get("exact_tuple_membership_checked") is not True: raise ValueError("fractional semantic tuples not checked exactly")
    if _fractional_number(groups.get("max_group_mass_closure_abs_kg"),"semantic group closure")>mass_limit or _fractional_number(groups.get("max_shared_child_capacity_violation_kg"),"semantic capacity")>mass_limit: raise ValueError("fractional semantic group closure/capacity failed")
    donors=receipt["source_donor_rows"]
    if type(donors.get("source_donor_closure_group_count")) is not int or donors["source_donor_closure_group_count"]<=0 or _fractional_number(donors.get("max_closure_abs_kg"),"source donor closure")>mass_limit: raise ValueError("fractional source donor closure failed")
    density=receipt["density"]
    if not isinstance(density,dict) or density.get("violation_count")!=0: raise ValueError("fractional material density gate failed")
    bounds=density.get("bounds_by_material"); observed=density.get("by_material")
    if not isinstance(bounds,dict) or set(bounds)!={"15","16"} or not isinstance(observed,dict) or set(observed)!={"15","16"}: raise ValueError("fractional density materials must be exactly 15 and 16")
    expected={"15":(900,1100),"16":(850,950)}
    for material,(lower,upper) in expected.items():
        if bounds[material]!={"lower_kgm3":lower,"upper_kgm3":upper}: raise ValueError(f"fractional material {material} density bounds changed")
        item=observed[material]
        if not isinstance(item,dict) or set(item)!={"min_kgm3","max_kgm3","within_bounds"}: raise ValueError(f"fractional material {material} observed density schema mismatch")
        lo=_fractional_number(item["min_kgm3"],f"density.{material}.min"); hi=_fractional_number(item["max_kgm3"],f"density.{material}.max")
        if item["within_bounds"] is not True or lo<lower or hi>upper or lo>hi: raise ValueError(f"fractional material {material} outside unchanged density bounds")
    return input_refs

def verify_current_mass_binding(route: Any,receipt: dict[str,Any],cfg: dict[str,Any])->None:
    schema=receipt.get("schema")
    if schema==FRACTIONAL_MASS_SCHEMA:
        input_refs=verify_fractional_mass_contract(route,receipt)
        env=cfg.get("native_preflight_env")
        if not isinstance(env,dict): raise ValueError("fractional mass binding requires native environment")
        fractional=verify_h706_fractional_env(cfg)
        identity_pins=cfg["h706_fractional_path_pins"]
        def h706_ref(hash_key:str)->tuple[str,str]:
            pin=identity_pins[hash_key]
            return str(Path(pin["path"]).resolve()),fractional[hash_key]
        def h706_file(path_key:str,hash_key:str)->tuple[str,str]:
            return str(Path(fractional[path_key]).resolve()),fractional[hash_key]
        def native_file(path_key:str,hash_key:str)->tuple[str,str]:
            return str(Path(env[path_key]).resolve()),env[hash_key]
        auxiliary=cfg.get("h706_fractional_auxiliary_path_pins")
        if not isinstance(auxiliary,dict) or set(auxiliary)!={"projection_report"}:
            raise ValueError("fractional mass binding lacks exact projection-report source pin")
        report_pin=auxiliary["projection_report"]
        if not isinstance(report_pin,dict) or set(report_pin)!={"path","resolved_path","sha256"}:
            raise ValueError("fractional projection-report pin schema mismatch")
        report_path=Path(report_pin["path"])
        if not report_path.is_absolute() or report_pin["resolved_path"]!=str(report_path.resolve()) or not report_path.is_file() or base.sha_file(report_path)!=report_pin["sha256"]:
            raise ValueError("fractional projection-report pin changed")
        expected_inputs={
            "current_state":h706_ref("STATE_SHA256"),
            "candidate_skin_volume":h706_ref("VOLUME_SKIN_SHA256"),
            "candidate_fat_volume":h706_ref("VOLUME_FAT_SHA256"),
            "child_skin_node":native_file("TETGEN_NODE","TETGEN_NODE_SHA256"),
            "child_skin_elements":native_file("TETGEN_ELEMENTS","TETGEN_ELEMENTS_SHA256"),
            "child_skin_labels":native_file("CHILD_LABELS","CHILD_LABELS_SHA256"),
            "child_fat_node":native_file("TETGEN_NODE","TETGEN_NODE_SHA256"),
            "child_fat_elements":native_file("TETGEN_ELEMENTS","TETGEN_ELEMENTS_SHA256"),
            "child_fat_labels":native_file("CHILD_LABELS","CHILD_LABELS_SHA256"),
            "source330_mass":h706_ref("SOURCE330_MASS_SHA256"),
            "source330_labels":native_file("PARENT_LABELS_CSV","PARENT_LABELS_SHA256"),
            "source330_nodes":h706_file("SOURCE330_NODES","SOURCE330_NODES_SHA256"),
            "source330_tets":h706_file("SOURCE330_TETS","SOURCE330_TETS_SHA256"),
            "p1_skin_support":h706_file("P1_SKIN_NPZ","P1_SKIN_SHA256"),
            "p1_fat_support":h706_file("P1_FAT_NPZ","P1_FAT_SHA256"),
            "carrier_skin_packet":h706_file("CARRIERS_SKIN_NPZ","CARRIERS_SKIN_SHA256"),
            "carrier_fat_packet":h706_file("CARRIERS_FAT_NPZ","CARRIERS_FAT_SHA256"),
            "projection_evidence":h706_file("EXACT_PROJECTION_CSV","EXACT_PROJECTION_SHA256"),
            "projection_report":_fractional_path_hash({"path":report_pin["path"],"sha256":report_pin["sha256"]},"projection_report pin"),
            "source_centroid_slots":h706_file("SOURCE_CENTROID_CSV","SOURCE_CENTROID_SHA256"),
            "carrier_witness_skin":h706_file("CARRIER_WITNESS_SKIN_CSV","CARRIER_WITNESS_SKIN_SHA256"),
            "carrier_witness_fat":h706_file("CARRIER_WITNESS_FAT_CSV","CARRIER_WITNESS_FAT_SHA256"),
        }
        for key,expected in expected_inputs.items():
            if input_refs[key]!=expected:
                raise ValueError(f"fractional mass input {key} differs from authenticated source")
        state_ref=_fractional_path_hash(receipt["candidate"]["current_state"],"candidate.current_state")
        if state_ref!=h706_ref("STATE_SHA256"):
            raise ValueError("fractional mass candidate state differs from authenticated 818c H706 state")
        if _fractional_path_hash(receipt["candidate"]["skin_volume"],"candidate.skin_volume")!=h706_ref("VOLUME_SKIN_SHA256") or _fractional_path_hash(receipt["candidate"]["fat_volume"],"candidate.fat_volume")!=h706_ref("VOLUME_FAT_SHA256"):
            raise ValueError("fractional mass candidate Skin/Fat volumes differ from authenticated canonical834 sources")
        if receipt["candidate"]["coordinate_frame_sha256"]!=env.get("COORDINATE_FRAME_SHA256"): raise ValueError("fractional mass coordinate frame differs from native route")
        if receipt["candidate"]["region_label_sha256"]!=env.get("CHILD_LABELS_SHA256"): raise ValueError("fractional mass region labels differ from current 433 child labels")
        source=receipt["source330"]
        for field,pathkey,hashkey in (("mass_packet","PARENT_PACKET_CSV","PARENT_PACKET_SHA256"),("labels","PARENT_LABELS_CSV","PARENT_LABELS_SHA256")):
            if _fractional_path_hash(source[field],f"source330.{field}")!=native_file(pathkey,hashkey): raise ValueError(f"fractional source330 {field} differs from current native parent")
        contribution_ref=_fractional_path_hash(source["parent_source_contributions"],"source330.parent_source_contributions")
        mass_path=Path(identity_pins["SOURCE330_MASS_SHA256"]["path"])
        expected_contribution_path=Path(str(mass_path)+".source.csv")
        if not expected_contribution_path.is_file(): raise ValueError("fractional source330 parent-source contribution file is absent")
        expected_contribution=(str(expected_contribution_path.resolve()),base.sha_file(expected_contribution_path))
        if contribution_ref!=expected_contribution:
            raise ValueError("fractional source330 parent-source contribution differs from exact source packet sibling")
        source_manifest_ref=cfg["h706_fractional_environment_source"]
        source_manifest_path=Path(source_manifest_ref["path"])
        if not source_manifest_path.is_file() or base.sha_file(source_manifest_path)!=source_manifest_ref["sha256"]:
            raise ValueError("fractional source run-manifest hash changed before file-pin binding")
        source_manifest=base.read_json(source_manifest_path)
        run_file_pins=source_manifest.get("file_pins")
        if not isinstance(run_file_pins,list): raise ValueError("fractional source run-manifest lacks file_pins list")
        matching_pins=[]
        for pin in run_file_pins:
            if not isinstance(pin,dict) or not isinstance(pin.get("path"),str) or not Path(pin["path"]).is_absolute(): continue
            if str(Path(pin["path"]).resolve())==expected_contribution[0]: matching_pins.append(pin)
        if len(matching_pins)!=1:
            raise ValueError("fractional parent-source contribution must have exactly one authenticated run file pin")
        run_pin=matching_pins[0]
        if run_pin.get("path")!=str(expected_contribution_path) or run_pin.get("sha256")!=expected_contribution[1]:
            raise ValueError("fractional parent-source contribution run file pin differs from exact sibling bytes")
        if "bytes" in run_pin and (type(run_pin["bytes"]) is not int or run_pin["bytes"]!=expected_contribution_path.stat().st_size):
            raise ValueError("fractional parent-source contribution run file pin byte count changed")
        return
    if schema is not None: raise ValueError(f"unsupported mass admission receipt schema: {schema!r}")
    verify_legacy_current_mass_binding(route,receipt,cfg)

def validate_claims(route: Any,cfg: dict[str,Any],candidate_pins: dict[str,str]):
    if not isinstance(cfg.get("geometry_admission"),dict):
        raise ValueError("candidate-specific geometry qualification receipt is absent")
    if not isinstance(cfg.get("mass_admission"),dict):
        raise ValueError("candidate-specific conservative mass/density receipt is absent")
    geometry,geo_sha=route.check_claim_receipt(
        cfg["geometry_admission"],route.REQUIRED_GEOMETRY_CLAIMS,"geometry admission",
        input_pins=candidate_pins,expected_scope=route.SOURCE_GEOMETRY_SCOPE)
    mass,mass_sha=route.check_claim_receipt(
        cfg["mass_admission"],route.REQUIRED_MASS_CLAIMS,"mass admission")
    if geometry.get("qualification_scope")!="source_geometry_only" or geometry.get("physical_admission") is not False:
        raise ValueError("geometry receipt must remain source_geometry_only with physical_admission=false")
    verify_current_mass_binding(route,mass,cfg)
    return ({"receipt_path":cfg["geometry_admission"]["path"],"receipt_sha256":geo_sha,
             "qualification_scope":"source_geometry_only","physical_admission":False,
             "claims":cfg["geometry_admission"]["claims"]},
            {"receipt_path":cfg["mass_admission"]["path"],"receipt_sha256":mass_sha,
             "claims":cfg["mass_admission"]["claims"]})


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
    # Route 712 names operational volume digest fields *_VOLUME_SHA256;
    # generic 711 mapping incorrectly appends _ASSET_SHA256 here.
    pairs["OPERATIONAL_SOURCE_VOLUME_ASSET"]="OPERATIONAL_SOURCE_VOLUME_SHA256"
    pairs["OPERATIONAL_CANDIDATE_VOLUME_ASSET"]="OPERATIONAL_CANDIDATE_VOLUME_SHA256"
    pairs["CHILD_LABELS"]="CHILD_LABELS_SHA256"
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


def apply_direct_env(route: Any,scene: dict[str,str],suffix: dict[str,str],source_only: bool,split: bool,
                     h706_fractional: dict[str,str]|None=None)->dict[str,str]:
    env=route.build_stage_env(scene,suffix,source_only=source_only,canonical_export=False,operational_split=split)
    for key in list(env):
        if key.startswith(route.PREFIX+"DETAIL_") or key.startswith(route.PREFIX+"COARSE_BOUNDARY_FACES_"): env.pop(key)
    env[BRIDGE]="0"; env[CANONICAL]="0"
    # The frozen 016 diagnostic baseline enables this multi-hundred-megabyte dump.
    # Direct-O source and bounded motion stages explicitly disable it.
    env["NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE"]="0"
    if h706_fractional is not None:
        if set(h706_fractional)!=H706_FRACTIONAL_ENV_KEYS:
            raise ValueError("H706 fractional stage environment lost or added a field")
        for key,value in h706_fractional.items():
            env[H706_FRACTIONAL_PREFIX+key]=value
    return env


def ctx(cfg: dict[str,Any]):
    if cfg.get("registered_surface_mode")!=MODE: raise ValueError("manifest must explicitly select direct-O/no-legacy-bridge mode")
    static=base.verify_static(cfg); route=route_module(cfg); pins=base.merged_pins(cfg)
    route.verify_input_pins(pins); suffix=direct_env(route,cfg,pins)
    fractional=verify_h706_fractional_env(cfg)
    candidate_pins=cfg.get("input_pins",{})
    if not isinstance(candidate_pins,dict) or not candidate_pins: raise ValueError("candidate input pins are required")
    geo,mass=validate_claims(route,cfg,candidate_pins)
    return static,route,pins,suffix,geo,mass,fractional


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
    try: route=route_module(cfg)
    except Exception as e: blockers.append(f"route 711 import: {type(e).__name__}: {e}")
    if route is not None:
        required=route.REQUIRED_ENV-detail_fields(route); env=cfg.get("native_preflight_env",{})
        missing=required-set(env)
        if missing: blockers.append(f"current 433 direct-O environment missing {len(missing)} fields")
        if not cfg.get("input_pins"): blockers.append("current 433 input pins are absent")
        else:
            try: direct_env(route,cfg,base.merged_pins(cfg)); route.verify_input_pins(base.merged_pins(cfg))
            except Exception as e: blockers.append(f"current 433 direct-O inputs: {type(e).__name__}: {e}")
        try: verify_h706_fractional_env(cfg)
        except Exception as e: blockers.append(f"current H706 fractional app inputs: {type(e).__name__}: {e}")
        for name,claims in (("geometry_admission",route.REQUIRED_GEOMETRY_CLAIMS),("mass_admission",route.REQUIRED_MASS_CLAIMS)):
            receipt=cfg.get(name)
            if not isinstance(receipt,dict) or not receipt.get("path") or not receipt.get("sha256"):
                blockers.append(f"candidate-specific {name.replace('_',' ')} receipt is absent")
            else:
                try:
                    kwargs=({"input_pins":cfg.get("input_pins",{}),
                             "expected_scope":route.SOURCE_GEOMETRY_SCOPE}
                            if name=="geometry_admission" else {})
                    route.check_claim_receipt(receipt,claims,name.replace("_"," "),**kwargs)
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
    cfg=base.load_config(cfg_path); static,route,pins,suffix,geo,mass,fractional=ctx(cfg)
    for flag,candidate in cfg.get("argv_overrides",{}).items():
        if candidate not in pins: raise ValueError(f"argv override {flag} lacks exact input pin")
    root=root_path(cfg); steps=cfg.get("measurement_steps")
    if not isinstance(steps,int) or steps not in (1,2): raise ValueError("first measurement must be one or two steps")
    argv,physics,stale=base.safe_scene(cfg,static,route,root/"source-preflight",steps)
    # The H706 fractional owner consumes the current candidate directly; its
    # native contract forbids the separated operational-reference split.
    operational_split=fractional["ENABLE"]!="1"
    env=apply_direct_env(route,physics,suffix,True,operational_split,fractional)
    stage={"name":"source_preflight_direct_O_without_legacy_detail_bridge","argv":argv,"environment":env,
           "expected_runtime":False,"expected_accepted_steps":0,"expected_gpu":False,"cwd":static["baseline"].get("cwd")}
    plan={"schema":"numi.human.current-native-launch-1025-direct-o.v1","status":"DIRECT_O_SOURCE_PREFLIGHT_READY",
          "registered_surface_mode":MODE,"execution_performed":False,"native_runtime_executed":False,"GPU_executed":False,
          "manifest_path":str(cfg_path.resolve()),"manifest_sha256":base.json_digest(cfg),
          "adapter_sha256":base.sha_file(Path(__file__).resolve()),"base_adapter_sha256":base.sha_file(BASE_PATH),
          "build_manifest_sha256":static["manifest_sha256"],"route_711_sha256":static["route_sha256"],
          "scoped_geometry_route_712_sha256":ROUTE_SHA256,
          "geometry_qualification_scope":"source_geometry_only","physical_admission":False,
          "native_binary":{"path":static["build"]["build"]["binary"],"sha256":static["native_binary_sha256"],
                           "repo_head":static["build"]["source"]["head"],
                           "dirty_fingerprint_sha256":static["build"]["source"]["dirty_entries_fingerprint_sha256"],
                           "dirty_entry_count":static["build"]["source"]["dirty_entry_count"]},
          "geometry_admission":geo,"mass_admission":mass,"input_pins":pins,
          "h706_fractional_env_sha256":base.json_digest(fractional),
          "h706_fractional_path_pins":cfg["h706_fractional_path_pins"],
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
    cfg=base.load_config(cfg_path); static,route,pins,suffix,geo,mass,fractional=ctx(cfg); plan=base.read_json(path)
    if plan.get("registered_surface_mode")!=MODE: raise ValueError("saved plan is not direct-O")
    if plan.get("manifest_path")!=str(cfg_path.resolve()) or plan.get("manifest_sha256")!=base.json_digest(cfg):
        raise ValueError("saved plan does not bind exact manifest bytes")
    if plan.get("build_manifest_sha256")!=static["manifest_sha256"] or plan.get("native_binary",{}).get("sha256")!=static["native_binary_sha256"]:
        raise ValueError("saved plan has stale build identity")
    if plan.get("scoped_geometry_route_712_sha256")!=ROUTE_SHA256:
        raise ValueError("saved plan is bound to a different scoped geometry reader")
    if plan.get("geometry_qualification_scope")!="source_geometry_only" or plan.get("physical_admission") is not False:
        raise ValueError("saved plan improperly upgrades source geometry to physical admission")
    if plan.get("input_pins")!=pins or plan.get("geometry_admission")!=geo or plan.get("mass_admission")!=mass:
        raise ValueError("saved plan input/geometry/mass pins changed")
    if plan.get("h706_fractional_env_sha256")!=base.json_digest(fractional) or plan.get("h706_fractional_path_pins")!=cfg.get("h706_fractional_path_pins"):
        raise ValueError("saved plan H706 fractional env or path pins changed")
    for stage in plan.get("stages",[]):
        stage_env=stage.get("environment",{})
        observed={k:v for k,v in stage_env.items() if k.startswith(H706_FRACTIONAL_PREFIX)}
        if observed!={H706_FRACTIONAL_PREFIX+key:value for key,value in fractional.items()}:
            raise ValueError("saved plan dropped, changed, or added an H706 fractional environment field")
    if any(s.get("environment",{}).get(BRIDGE)!="0" or s.get("environment",{}).get(CANONICAL)!="0" for s in plan.get("stages",[])):
        raise ValueError("saved plan enables legacy detail bridge")
    if any(k.startswith(route.PREFIX+"DETAIL_") for s in plan.get("stages",[]) for k in s.get("environment",{})):
        raise ValueError("saved plan carries legacy detail-bridge inputs")
    return plan,cfg,route,{"static":static,"pins":pins,"env":suffix,"geo":geo,"mass":mass,"fractional":fractional}


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
             "geometry_qualification_scope":"source_geometry_only","physical_admission":False,
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
    env=apply_direct_env(route,scene,ctxdata["env"],False,False,ctxdata["fractional"])
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
             "geometry_qualification_scope":"source_geometry_only","physical_admission":False,
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

