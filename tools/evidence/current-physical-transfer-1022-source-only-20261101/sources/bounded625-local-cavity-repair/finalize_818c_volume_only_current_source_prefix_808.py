#!/usr/bin/env python3
"""818 source-authenticated volume-only metadata prefix.

The source surface pins are inherited unchanged from the supplied state. For
Skin and Fat, rebuild current typed target aliases with the existing 677 owner,
then use the existing 709 volume rebase helper to reset only immediate maps.
No source file, geometry array, mass ledger, or physical status is changed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OWNER_DIR=HERE
OWNER709=OWNER_DIR/"finalize_709_current_source_volume_metadata.py"
OWNER709_SHA="02cba640738c0f67bf08fc711c16ddf2f21ecd32a3ed1777507de670eb87005f"
OWNER677=OWNER_DIR/"finalize_677_target_metadata_625_v2.py"
OWNER677_SHA="af0333a7832cc136d15c5a24a169d69d041bc1e4bbf6813fd6b31f7740189756"
SOURCE807=ROOT/"verify_current_source_evidence_808_v26a.py"
SOURCE807_SHA="8c7f6e78218cb7f3db62d657ea23dd074d77cc58bd5514362678047549ac74a9"
SOURCE807_BASE=ROOT/"verify_current_source_evidence_808_validation_v26a.py"
SOURCE807_BASE_SHA="6daa77251509436219b9ad4cb710ccaf4e4cf527a55358ad400c0b71e8881ef0"
SCHEMA="numi.quality818b.current-source-volume-metadata-prefix.v1"
STATE_WRAPPER="current_verified_state_20261031"
ROLE_NAMES=("Skin","Fat")
SOURCE_NAMES=("O","I","D")
GEOMETRY_KEYS=("points_world_m","points_world_m_f32","tetrahedra","material_indices","boundary_faces")
STEP_CSR_KEYS=("replacement_parent_offsets","replacement_parent_cell_rows","replacement_tet_offsets",
               "replacement_tetrahedra","output_point_parent_offsets","output_point_parent_input_cell_rows")
IDENTITY_KEYS=("output_cell_to_input_cell","input_cell_to_output_cell","retained_input_cell_rows",
               "input_node_to_output_node","output_node_to_input_node","output_point_to_input_point_row",
               "source_interior_vertex_quotient")
CELL_MAP_ALIAS_KEYS=("output_cell_to_input_cell","output_cell_to_input_cell_rows",
 "input_cell_to_output_cell","input_cell_to_output_cell_rows","retained_input_cell_rows")
POINT_MAP_ALIAS_KEYS=("input_node_to_output_node","input_node_to_output_node_rows",
 "output_node_to_input_node","output_node_to_input_node_rows",
 "output_point_to_input_point_row","output_point_to_input_point_rows",
 "source_interior_vertex_quotient")

TARGET_ALIAS_KEYS=("desired_boundary_faces","desired_boundary_source_rows","desired_boundary_source_group",
 "desired_boundary_source_face_rows","desired_boundary_degenerate_source_rows","mapped_current_boundary_faces",
 "proposed_current_desired_boundary_faces","proposed_current_desired_boundary_source_group",
 "proposed_current_desired_boundary_source_rows","proposed_current_desired_boundary_source_face_rows")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def pin(path):
    p=Path(path).resolve()
    return {"path":str(p),"sha256":sha(p)}

def ah(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def payload_equal(a,b):
    x=np.asarray(a);y=np.asarray(b)
    return x.shape==y.shape and x.dtype==y.dtype and ah(x)==ah(y)

def load_json(path):
    return json.loads(Path(path).read_text())

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def owners():
    for path,digest,label in ((OWNER709,OWNER709_SHA,"709 owner"),
                              (OWNER677,OWNER677_SHA,"677 target owner"),
                              (SOURCE807,SOURCE807_SHA,"808 source verifier"),
                              (SOURCE807_BASE,SOURCE807_BASE_SHA,"808 verifier base")):
        if not path.is_file() or sha(path)!=digest:raise ValueError(label+" pin mismatch")
    v=import_file("quality818_709_volume_owner",OWNER709)
    m=v.helper()
    if Path(m.__file__).resolve()!=OWNER677.resolve():raise ValueError("709 loaded an unexpected 677 target owner")
    source=import_file("quality818_808_source_verifier",SOURCE807)
    return v,m,source

def unwrap(raw):
    if STATE_WRAPPER in raw and isinstance(raw[STATE_WRAPPER],dict):
        return STATE_WRAPPER,raw[STATE_WRAPPER]
    return None,raw

def normalized_source_pins(state):
    rows=state.get("source_surfaces")
    if not isinstance(rows,dict) or set(rows)!=set(SOURCE_NAMES):
        raise ValueError("input state must contain exactly current O/I/D source surface pins")
    out={}
    for role in SOURCE_NAMES:
        row=rows[role]
        path=Path(row.get("path","")).resolve()
        digest=str(row.get("sha256",""))
        if not path.is_file() or sha(path)!=digest:raise ValueError("current source surface pin mismatch: "+role)
        out[role]={"path":str(path),"sha256":digest}
    return out

def authorize_source(state_path,state,source_verifier):
    proof=source_verifier.verify_current_source_evidence(state_path,state)
    if proof.get("status") not in ("PASS_CURRENT_EXACT_SOURCE_EVIDENCE","PASS_INHERITED_CURRENT_SOURCE_EVIDENCE"):
        raise ValueError("808 source evidence did not pass")
    if proof.get("physical_admission") is not False or int(proof.get("native_steps",-1))!=0:
        raise ValueError("808 source evidence overstates admission")
    pins=normalized_source_pins(state)
    proof_pins=proof.get("source_surfaces")
    if not isinstance(proof_pins,dict):raise ValueError("808 proof does not expose current source surface pins")
    normalized_proof={r:{"path":str(Path(proof_pins[r]["path"]).resolve()),"sha256":proof_pins[r]["sha256"]}
                      for r in SOURCE_NAMES}
    if normalized_proof!=pins:raise ValueError("808 proof source pins differ from the supplied state")
    return proof,pins

def reset_present_identity_aliases(parent,rebased):
    ncell=len(np.asarray(parent["tetrahedra"]));npoint=len(np.asarray(parent["points_world_m_f32"]))
    prior={}
    for key in CELL_MAP_ALIAS_KEYS:
        if key in parent:
            a=np.asarray(parent[key])
            prior[key]={"shape":list(a.shape),"dtype":str(a.dtype),"sha256":ah(a)}
            rebased[key]=np.arange(ncell,dtype=np.int64)
    for key in POINT_MAP_ALIAS_KEYS:
        if key in parent:
            a=np.asarray(parent[key])
            prior[key]={"shape":list(a.shape),"dtype":str(a.dtype),"sha256":ah(a)}
            rebased[key]=np.arange(npoint,dtype=np.int64)
    rebased["metadata_finalizer_818b_previous_immediate_aliases_json"]=np.asarray(json.dumps(prior,sort_keys=True))
    rebased["metadata_finalizer_818b_alias_reset_schema"]=np.asarray(SCHEMA)
    return prior

def validate_identity_aliases(parent,rebased,role):
    ncell=len(np.asarray(parent["tetrahedra"]));npoint=len(np.asarray(parent["points_world_m_f32"]))
    checked=0
    for key in CELL_MAP_ALIAS_KEYS:
        if key in parent:
            expected=np.arange(ncell,dtype=np.int64)
            if key not in rebased or np.asarray(rebased[key]).shape!=(ncell,) or not np.array_equal(np.asarray(rebased[key]),expected):
                raise ValueError(role+" immediate cell alias is not current identity: "+key)
            checked+=1
    for key in POINT_MAP_ALIAS_KEYS:
        if key in parent:
            expected=np.arange(npoint,dtype=np.int64)
            if key not in rebased or np.asarray(rebased[key]).shape!=(npoint,) or not np.array_equal(np.asarray(rebased[key]),expected):
                raise ValueError(role+" immediate point alias is not current identity: "+key)
            checked+=1
    return checked

def check_volume_contract(parent,rebased,role,target,groups,rows,pmap,wall_local,wall_xyz,source_pins,
                          parent_path,parent_sha,owner709):
    npoint=len(np.asarray(parent["points_world_m_f32"],np.float32))
    ncell=len(parent["tetrahedra"])
    owner709.equal_geometry(parent,rebased,owner709.VGEOM,role+" volume")
    for key in GEOMETRY_KEYS:
        if key not in parent or key not in rebased or not payload_equal(parent[key],rebased[key]):
            raise ValueError(role+" geometry bytes changed: "+key)
    # Every field outside the explicit metadata/identity-map rebind allowlist
    # must be byte-preserved. This includes canonical absolute ancestry arrays.
    def permitted(k):
        if k in set(TARGET_ALIAS_KEYS)|set(STEP_CSR_KEYS)|set(IDENTITY_KEYS)|set(CELL_MAP_ALIAS_KEYS)|set(POINT_MAP_ALIAS_KEYS):return True
        if k in ("source_interior_vertex_quotient","canonical_parent_union_operation_order",
                 "current_geometry_parent_path","current_geometry_parent_sha256"):
            return True
        if k.startswith("typed_") and k.endswith("_candidate_path"):return True
        if k.startswith("typed_") and k.endswith("_candidate_sha256"):return True
        if k.startswith("source_") and k.endswith("_operation_rows"):return True
        if k.startswith("metadata_finalizer") or k.startswith("target_metadata_"):return True
        prefix="skin" if role=="Skin" else "fat"
        if k.startswith(prefix+"_current_target_rebind_"):return True
        return False
    missing=set(parent)-set(rebased)
    if missing:raise ValueError(role+" metadata rebase dropped input arrays: "+",".join(sorted(missing)[:8]))
    unexpected=[]
    preserved=0
    for key in parent:
        if permitted(key):continue
        if not payload_equal(parent[key],rebased[key]):unexpected.append(key)
        else:preserved+=1
    if unexpected:raise ValueError(role+" changed protected volume ancestry/metadata arrays: "+",".join(unexpected[:8]))
    # The absolute parent-union payload is immutable; only its operation-order
    # diagnostic may be rewritten by the 709 owner.
    for key in parent:
        if key.startswith("canonical_parent_union_") and key!="canonical_parent_union_operation_order":
            if key not in rebased or not payload_equal(parent[key],rebased[key]):
                raise ValueError(role+" canonical absolute parent union changed: "+key)
    if not np.array_equal(rebased["desired_boundary_faces"],target):
        raise ValueError(role+" current target face rows were not regenerated")
    for key,value in (("desired_boundary_source_group",groups),("desired_boundary_source_rows",rows),
                      ("desired_boundary_source_face_rows",rows),
                      ("proposed_current_desired_boundary_faces",target),
                      ("proposed_current_desired_boundary_source_group",groups),
                      ("proposed_current_desired_boundary_source_rows",rows),
                      ("proposed_current_desired_boundary_source_face_rows",rows)):
        if key not in rebased or not np.array_equal(rebased[key],value):
            raise ValueError(role+" target alias mismatch: "+key)
    if len(rows)!=len(target) or len(groups)!=len(target):raise ValueError(role+" target row metadata length mismatch")
    if not np.array_equal(rebased["mapped_current_boundary_faces"],parent["boundary_faces"]):
        raise ValueError(role+" actual boundary alias changed")
    if len(rebased["desired_boundary_degenerate_source_rows"])!=0:
        raise ValueError(role+" target unexpectedly declares degenerate source rows")
    if owner709.helper().directed_chain(target)!=owner709.helper().directed_chain(parent["boundary_faces"]):
        raise ValueError(role+" target does not equal current actual oriented boundary chain")
    if len(target)!=len(parent["boundary_faces"]):
        raise ValueError(role+" target face count differs from actual boundary")
    for key in IDENTITY_KEYS:
        n=ncell if key in ("output_cell_to_input_cell","input_cell_to_output_cell","retained_input_cell_rows") else npoint
        expected=np.arange(n,dtype=np.int64)
        if key not in rebased or not np.array_equal(np.asarray(rebased[key]),expected):
            raise ValueError(role+" immediate map is not identity: "+key)
    alias_count=validate_identity_aliases(parent,rebased,role)
    if not np.array_equal(rebased["replacement_parent_offsets"],np.asarray([0],np.int64)):
        raise ValueError(role+" replacement-parent CSR is not empty")
    if np.asarray(rebased["replacement_parent_cell_rows"]).size:
        raise ValueError(role+" replacement-parent rows are not empty")
    if not np.array_equal(rebased["replacement_tet_offsets"],np.asarray([0,0],np.int64)):
        raise ValueError(role+" replacement-tet CSR is not empty")
    if np.asarray(rebased["replacement_tetrahedra"]).shape!=(0,4):
        raise ValueError(role+" replacement tetrahedra are not empty")
    if not np.array_equal(rebased["output_point_parent_offsets"],np.zeros(npoint+1,np.int64)):
        raise ValueError(role+" immediate point-parent CSR offsets are not empty")
    if np.asarray(rebased["output_point_parent_input_cell_rows"]).size:
        raise ValueError(role+" immediate point-parent rows are not empty")
    for key,row in source_pins.items():
        for field,value in ((f"typed_{key}_candidate_path",row["path"]),
                            (f"typed_{key}_candidate_sha256",row["sha256"])):
            if field not in rebased or owner709.val(rebased[field])!=value:
                raise ValueError(role+" typed source pin alias mismatch: "+field)
    prefix="skin" if role=="Skin" else "fat"
    role_sources=("O","I") if role=="Skin" else ("I","D")
    for sr in role_sources:
        field=f"{prefix}_current_target_rebind_{sr}_point_to_volume_rows"
        if field not in rebased or not np.array_equal(rebased[field],pmap[sr]):
            raise ValueError(role+" target source-point crosswalk mismatch: "+sr)
    if not np.array_equal(rebased[f"{prefix}_current_target_rebind_wall_faces_volume_rows"],target[-90:]):
        raise ValueError(role+" wall face row alias mismatch")
    if not np.array_equal(rebased[f"{prefix}_current_target_rebind_wall_faces_collar_point_rows"],wall_local):
        raise ValueError(role+" wall collar row alias mismatch")
    if not payload_equal(rebased[f"{prefix}_current_target_rebind_wall_face_points_world_m_f32"],wall_xyz):
        raise ValueError(role+" wall point alias mismatch")
    opmaps=[k for k in parent if k.startswith("source_") and k.endswith("_operation_rows")]
    for key in opmaps:
        if not np.all(np.asarray(rebased[key],np.int64)==-1):
            raise ValueError(role+" stale immediate operation map remains: "+key)
    parent_pin={"path":str(Path(parent_path).resolve()),"sha256":parent_sha}
    if owner709.val(rebased["current_geometry_parent_path"])!=parent_pin["path"] or owner709.val(rebased["current_geometry_parent_sha256"])!=parent_sha:
        raise ValueError(role+" immediate geometry parent pin mismatch")
    return {"role":role,"point_count":npoint,"cell_count":ncell,"target_face_count":len(target),
      "target_group_counts":{str(int(g)):int(np.count_nonzero(groups==g)) for g in np.unique(groups)},
      "geometry_hashes":{k:ah(parent[k]) for k in GEOMETRY_KEYS},
      "source_operation_maps_cleared":len(opmaps),"identity_map_lanes":alias_count,
      "singular_and_plural_aliases_checked":alias_count,
      "absolute_ancestry_fields_byte_preserved":preserved,
      "current_target_equals_actual_oriented_boundary_chain":True,"all_typed_source_pins_current":True,
      "immediate_point_and_cell_csrs_empty":True,"physical_admission":False,"native_steps":0}

def prepare(state_path,wall_config_path,output_dir=None,preflight_only=False):
    state_path=Path(state_path).resolve();wall_config_path=Path(wall_config_path).resolve()
    if output_dir is not None:
        output_dir=Path(output_dir).resolve()
        if output_dir.exists():raise FileExistsError(output_dir)
    v,helper,source_verifier=owners()
    raw=load_json(state_path);wrapper,state=unwrap(raw)
    proof,source_pins=authorize_source(state_path,state,source_verifier)
    cfg=load_json(wall_config_path);v.check_walls(cfg)
    wall_csv=Path(cfg["wall_csv"]["path"]).resolve()
    collar_path=Path(cfg["collar"]["path"]).resolve()
    collar=v.load(collar_path)
    if set(state.get("source_surfaces",{}))!=set(SOURCE_NAMES):raise ValueError("current source surface roles incomplete")
    surface_data={}
    for r in SOURCE_NAMES:
        p=Path(source_pins[r]["path"]);surface_data[r]=v.load(p)
    source_out={r:(Path(source_pins[r]["path"]),source_pins[r]["sha256"]) for r in SOURCE_NAMES}
    input_state_pin={"path":str(state_path),"sha256":sha(state_path)}
    wall_assets={"wall_csv":pin(wall_csv),"collar":pin(collar_path),
                 "wall_receipt":{"path":str(Path(cfg["wall_receipt"]["path"]).resolve()),
                                 "sha256":cfg["wall_receipt"]["sha256"]}}
    producer_pin=pin(Path(__file__))
    target_pin=pin(OWNER677)
    stage=None
    if not preflight_only:
        if output_dir is None:raise ValueError("output directory required unless --preflight-only")
        output_dir.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix=".quality818-stage-",dir=output_dir.parent))
    role_reports={};output_entries={}
    try:
        for role in ROLE_NAMES:
            entry=state.get(role+"_current")
            if not isinstance(entry,dict):raise ValueError("input state missing "+role+" volume")
            parent_path=Path(entry.get("path","")).resolve();parent_sha=str(entry.get("sha256",""))
            if not parent_path.is_file() or sha(parent_path)!=parent_sha:
                raise ValueError(role+" current volume pin mismatch")
            parent=v.load(parent_path)
            if not np.array_equal(np.asarray(parent["points_world_m"],np.float32).view(np.uint32),
                                  np.asarray(parent["points_world_m_f32"],np.float32).view(np.uint32)):
                raise ValueError(role+" current emitted and native F32 coordinate lanes differ")
            points=np.asarray(parent["points_world_m_f32"],np.float32)
            target,groups,rows,pmap,wall_ids,wall_local,wall_xyz=helper.build_target(role,points,surface_data,wall_csv,collar)
            if helper.directed_chain(target)!=helper.directed_chain(np.asarray(parent["boundary_faces"],np.int64)):
                raise ValueError(role+" typed current target differs from actual boundary chain")
            if len(target)!=len(parent["boundary_faces"]):
                raise ValueError(role+" target and actual boundary face counts differ")
            assets={**wall_assets,"input_state":input_state_pin,"volume_parent":{"path":str(parent_path),"sha256":parent_sha},
                    "producer":producer_pin,"target_helper":target_pin,
                    "source_evidence_verifier":pin(SOURCE807)}
            updated,detail=v.rebase_volume(parent,role,parent_path,parent_sha,target,groups,rows,pmap,
                                            wall_local,wall_xyz,source_out,assets)
            reset_present_identity_aliases(parent,updated)
            check=check_volume_contract(parent,updated,role,target,groups,rows,pmap,wall_local,wall_xyz,
                                        source_pins,parent_path,parent_sha,v)
            outname=role.lower()+"-volume-current-source-prefix.npz"
            final_path=output_dir/outname if output_dir is not None else None
            if stage is not None:
                stage_path=stage/outname
                np.savez_compressed(stage_path,**updated)
                reread=v.load(stage_path)
                check_volume_contract(parent,reread,role,target,groups,rows,pmap,wall_local,wall_xyz,
                                      source_pins,parent_path,parent_sha,v)
                output_entries[role]={"path":str(final_path),"sha256":sha(stage_path),
                  "metadata_parent_path":str(parent_path),"metadata_parent_sha256":parent_sha,
                  "target_face_count":len(target),"target_boundary_match":True,
                  "metadata_only_geometry_identical":True}
                check["output"]={"path":str(final_path),"sha256":sha(stage_path)}
            role_reports[role]={**check,
              "input":{"path":str(parent_path),"sha256":parent_sha},
              "target_source_face_rows":int(len(rows)),
              "unmapped_unreferenced_source_points":{r:int(np.count_nonzero(np.asarray(pmap[r])<0)) for r in pmap},
              "owner_709_rebase_detail":detail}
        if preflight_only:
            return {"schema":SCHEMA,"status":"PASS_READ_ONLY_CURRENT_VOLUME_METADATA_PREFLIGHT",
              "input_state":input_state_pin,"source_proof":{"status":proof["status"],"mode":proof.get("mode"),
                "source_surfaces":proof["source_surfaces"],"all185":proof.get("all185")},
              "wall_config":pin(wall_config_path),"wall_assets":wall_assets,"roles":role_reports,
              "geometry_changed":False,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
        core=dict(state)
        core["source_state"]=input_state_pin
        # The volume prefix keeps the exact authenticated source_surface entries.
        core["source_surfaces"]=dict(state["source_surfaces"])
        for role in ROLE_NAMES:core[role+"_current"]={**state[role+"_current"],**output_entries[role]}
        core["metadata_finalizer_818b"]={"schema":SCHEMA,"status":"PASS_CURRENT_SOURCE_VOLUME_METADATA_PREFIX",
          "input_state":input_state_pin,"producer":producer_pin,"owner_709":{"path":str(OWNER709),"sha256":OWNER709_SHA},
          "owner_677":{"path":str(OWNER677),"sha256":OWNER677_SHA},
          "source_evidence_verifier":{"path":str(SOURCE807),"sha256":SOURCE807_SHA,
            "base_path":str(SOURCE807_BASE),"base_sha256":SOURCE807_BASE_SHA},
          "source_proof":{"status":proof["status"],"mode":proof.get("mode"),
            "source_surfaces":proof["source_surfaces"],"all185":proof.get("all185")},
          "wall_config":pin(wall_config_path),"wall_assets":wall_assets,
          "roles":role_reports,"geometry_changed":False,"physical_admission":False,
          "native_steps":0,"mass_mapping":"NOT_RUN"}
        if wrapper:
            final_state=dict(raw);final_state[wrapper]=core
        else:final_state=core
        state_stage=stage/"verified-state.json"
        state_stage.write_text(json.dumps(final_state,indent=2,sort_keys=True)+"\n")
        if core["source_state"]!=input_state_pin or core["source_surfaces"]!=state["source_surfaces"]:
            raise ValueError("volume prefix changed source pins or failed to pin its immediate input state")
        output_state_pin={"path":str(output_dir/"verified-state.json"),"sha256":sha(state_stage)}
        report={"schema":SCHEMA,"status":"PASS_CURRENT_SOURCE_VOLUME_METADATA_PREFIX",
          "producer":producer_pin,"owner_709":{"path":str(OWNER709),"sha256":OWNER709_SHA},
          "owner_677":{"path":str(OWNER677),"sha256":OWNER677_SHA},
          "source_evidence_verifier":{"path":str(SOURCE807),"sha256":SOURCE807_SHA,
            "base_path":str(SOURCE807_BASE),"base_sha256":SOURCE807_BASE_SHA},
          "input_state":input_state_pin,"source_proof":{"status":proof["status"],"mode":proof.get("mode"),
            "source_surfaces":proof["source_surfaces"],"all185":proof.get("all185")},
          "wall_config":pin(wall_config_path),"wall_assets":wall_assets,
          "volume_regions":role_reports,"output_state":output_state_pin,"geometry_changed":False,
          "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
          "surface_collision_audit":"NOT_RUN_METADATA_ONLY"}
        report_stage=stage/"volume-prefix-818b-report.json"
        report_stage.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        # The output paths were pinned before publication; atomically publish the
        # complete state and role files as one directory.
        os.replace(stage,output_dir);stage=None
        final_report=output_dir/"volume-prefix-818b-report.json"
        return {"status":report["status"],"report":pin(final_report),
          "verified_state":pin(output_dir/"verified-state.json"),"roles":role_reports,
          "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
    finally:
        if stage is not None:shutil.rmtree(stage,ignore_errors=True)

def state_prefix_control(source_surfaces,input_state):
    core={"source_surfaces":dict(source_surfaces),"Skin_current":{"path":"/out/skin.npz","sha256":"a"*64},
          "Fat_current":{"path":"/out/fat.npz","sha256":"b"*64}}
    core["source_state"]=input_state
    return core

def stale_plural_alias_control():
    parent={"tetrahedra":np.zeros((2,4),np.int64),"points_world_m_f32":np.zeros((5,3),np.float32),
      "output_cell_to_input_cell":np.asarray([1,0],np.int64),
      "output_cell_to_input_cell_rows":np.asarray([1,0],np.int64),
      "input_cell_to_output_cell":np.asarray([1,0],np.int64),
      "input_cell_to_output_cell_rows":np.asarray([1,0],np.int64),
      "retained_input_cell_rows":np.asarray([1,0],np.int64),
      "input_node_to_output_node":np.asarray([1,0,2,3,4],np.int64),
      "input_node_to_output_node_rows":np.asarray([1,0,2,3,4],np.int64),
      "output_node_to_input_node":np.asarray([1,0,2,3,4],np.int64),
      "output_node_to_input_node_rows":np.asarray([1,0,2,3,4],np.int64),
      "output_point_to_input_point_row":np.asarray([1,0,2,3,4],np.int64),
      "output_point_to_input_point_rows":np.asarray([1,0,2,3,4],np.int64),
      "source_interior_vertex_quotient":np.asarray([1,0,2,3,4],np.int64)}
    rebased=dict(parent)
    before=reset_present_identity_aliases(parent,rebased)
    if validate_identity_aliases(parent,rebased,"stale-alias-fixture")!=len(before):
        raise AssertionError("not all present identity aliases checked")
    if any(not np.array_equal(rebased[k],np.arange(2 if k in CELL_MAP_ALIAS_KEYS else 5,dtype=np.int64))
           for k in CELL_MAP_ALIAS_KEYS+POINT_MAP_ALIAS_KEYS):
        raise AssertionError("present singular/plural aliases were not reset to identity")
    bad=dict(rebased);bad["output_cell_to_input_cell_rows"]=np.asarray([1,0],np.int64)
    try:validate_identity_aliases(parent,bad,"tampered-fixture")
    except ValueError as e:
        if "output_cell_to_input_cell_rows" not in str(e):raise
    else:raise AssertionError("stale in-range plural map was accepted")
    missing=dict(rebased);del missing["input_cell_to_output_cell_rows"]
    try:validate_identity_aliases(parent,missing,"missing-fixture")
    except ValueError as e:
        if "input_cell_to_output_cell_rows" not in str(e):raise
    else:raise AssertionError("missing plural alias was accepted")
    return {"present_alias_count":len(before),"stale_values_were_in_range":True,"tamper_rejected":True,"missing_alias_rejected":True}

def self_test():
    alias_control=stale_plural_alias_control()
    v,helper,_=owners()
    p=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    t=np.asarray([[0,1,2,3]],np.int64)
    faces=np.asarray([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
    parent={"points_world_m":p.astype(np.float64),"points_world_m_f32":p.copy(),"tetrahedra":t,
      "material_indices":np.asarray([16],np.int32),"boundary_faces":faces.copy(),
      "desired_boundary_faces":faces[::-1].copy(),"desired_boundary_source_rows":np.arange(4)[::-1],
      "desired_boundary_source_group":np.ones(4,np.int8),"desired_boundary_source_face_rows":np.arange(4)[::-1],
      "mapped_current_boundary_faces":faces[::-1].copy(),"proposed_current_desired_boundary_faces":faces[::-1].copy(),
      "proposed_current_desired_boundary_source_rows":np.arange(4)[::-1],
      "proposed_current_desired_boundary_source_group":np.ones(4,np.int8),
      "proposed_current_desired_boundary_source_face_rows":np.arange(4)[::-1],
      "desired_boundary_degenerate_source_rows":np.asarray([2],np.int64),
      "typed_O_candidate_path":np.asarray("/stale/O.npz"),"typed_O_candidate_sha256":np.asarray("0"*64),
      "typed_I_candidate_path":np.asarray("/stale/I.npz"),"typed_I_candidate_sha256":np.asarray("0"*64),
      "typed_D_candidate_path":np.asarray("/stale/D.npz"),"typed_D_candidate_sha256":np.asarray("0"*64),
      "output_cell_to_input_cell":np.asarray([-1],np.int64),"input_cell_to_output_cell":np.asarray([-1],np.int64),
      "retained_input_cell_rows":np.asarray([],np.int64),"input_node_to_output_node":np.arange(4)[::-1],
      "output_node_to_input_node":np.arange(4)[::-1],"output_point_to_input_point_row":np.arange(4)[::-1],
      "source_interior_vertex_quotient":np.arange(4)[::-1],
      "replacement_parent_offsets":np.asarray([0,1]),"replacement_parent_cell_rows":np.asarray([9]),
      "replacement_tet_offsets":np.asarray([0,1]),"replacement_tetrahedra":t.copy(),
      "output_point_parent_offsets":np.asarray([0,1,1,1,1]),"output_point_parent_input_cell_rows":np.asarray([0]),
      "source_cell_operation_rows":np.asarray([7],np.int64),
      "canonical_parent_union_schema":np.asarray("absolute.v1"),
      "canonical_parent_union_offsets":np.asarray([0,1],np.int64),
      "canonical_parent_union_parent_rows":np.asarray([4242],np.int64),
      "canonical_parent_union_operation_order":np.asarray("old-step-order"),
      "quality_star_patch_union_offsets":np.asarray([0,1],np.int64),
      "quality_star_patch_union_source_original_ids":np.asarray([17],np.int64)}
    target=faces.copy();groups=np.zeros(4,np.int8);rows=np.arange(4,dtype=np.int64)
    sp={r:{"path":"/current/"+r+".npz","sha256":str(i)*64} for i,r in enumerate(SOURCE_NAMES,1)}
    source_out={r:(Path(sp[r]["path"]),sp[r]["sha256"]) for r in SOURCE_NAMES}
    assets={k:{"path":"/asset/"+k,"sha256":str(i)*64} for i,k in enumerate(("wall_csv","collar","wall_receipt"),1)}
    parent_path=Path("/raw/current-Fat.npz");parent_sha="f"*64
    rebased,detail=v.rebase_volume(parent,"Fat",parent_path,parent_sha,target,groups,rows,
       {"I":np.arange(4),"D":np.arange(4)},target.copy(),
       p[target].copy(),source_out,assets)
    check=check_volume_contract(parent,rebased,"Fat",target,groups,rows,{"I":np.arange(4),"D":np.arange(4)},
       target.copy(),p[target].copy(),sp,parent_path,parent_sha,v)
    assert check["current_target_equals_actual_oriented_boundary_chain"]
    assert np.array_equal(rebased["canonical_parent_union_parent_rows"],parent["canonical_parent_union_parent_rows"])
    src={"O":{"path":"/src/O","sha256":"a"*64},"I":{"path":"/src/I","sha256":"b"*64},"D":{"path":"/src/D","sha256":"c"*64}}
    parent_pin={"path":"/state/810.json","sha256":"d"*64}
    prefix=state_prefix_control(src,parent_pin)
    assert prefix["source_state"]==parent_pin and prefix["source_surfaces"]==src
    print(json.dumps({"status":"PASS_818B_VOLUME_METADATA_FIXTURE","stale_typed_aliases_rebound":True,
       "stale_plural_alias_control":alias_control,"geometry_and_absolute_ancestry_preserved":True,
       "canonical_parent_union_parent_rows_preserved":True,"point_cell_identity_maps_and_empty_csrs":True,
       "source_state_prefix_pin_control":True,"physical_admission":False,"native_steps":0},sort_keys=True))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state",type=Path)
    ap.add_argument("--wall-config",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--preflight-only",action="store_true")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:self_test();return
    if not (a.state and a.wall_config):ap.error("--state and --wall-config are required")
    if a.preflight_only:
        result=prepare(a.state,a.wall_config,preflight_only=True)
    else:
        if a.output is None:ap.error("--output is required unless --preflight-only or --self-test")
        result=prepare(a.state,a.wall_config,a.output)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":main()
