#!/usr/bin/env python3
"""Append typed O/I/D and interior registration rows to the existing 217 node-map CSV."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, importlib.util, json, math, sys, tempfile
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

INVALID=0xffffffff
LEGACY_MAX_NODES = 300_000
LEGACY_MAX_CELLS = 200_000
FULL_REFERENCE_MAX_NODES = 500_000
FULL_REFERENCE_MAX_CELLS = 2_000_000
MAX_NODES=LEGACY_MAX_NODES
MAX_CELLS=LEGACY_MAX_CELLS
CAPACITY_PROFILE = "legacy"

def configure_capacity_profile(name: str) -> dict[str, int | str]:
    """Apply an explicit child-output limit profile; source-parent limits stay fixed."""
    global MAX_NODES, MAX_CELLS, CAPACITY_PROFILE
    if name == "legacy":
        MAX_NODES, MAX_CELLS = LEGACY_MAX_NODES, LEGACY_MAX_CELLS
    elif name == "full-reference-500k-nodes-2m-cells":
        MAX_NODES, MAX_CELLS = FULL_REFERENCE_MAX_NODES, FULL_REFERENCE_MAX_CELLS
    else:
        raise ValueError(f"unknown capacity profile: {name}")
    CAPACITY_PROFILE = name
    return capacity_profile_receipt()

def capacity_profile_receipt() -> dict[str, int | str]:
    return {"name": CAPACITY_PROFILE, "max_nodes_per_child": MAX_NODES,
            "max_cells_per_child": MAX_CELLS, "max_nodes_combined": MAX_NODES,
            "max_cells_combined": MAX_CELLS}
BASE_HEADER=["joint_node_id","source_partition","source_tetgen_node_id",
 "registered_I_source_vertex_row","merge_peer_partition","merge_peer_tetgen_node_id",
 "registered_surface_layer","registered_surface_vertex_row"]
HEADER=BASE_HEADER+[
 "role","current_x_m","current_y_m","current_z_m","donor_body","donor_binding",
 "donor_local_x_m","donor_local_y_m","donor_local_z_m","source_domain","source_mapping_kind",
 "source_vertex_row","source_face_row","source_bary0","source_bary1","source_bary2",
 "source_residual_x_m","source_residual_y_m","source_residual_z_m","frame_owner_method",
 "parent_registration_kind","parent_tet_row","parent_bary0","parent_bary1","parent_bary2",
 "parent_bary3","parent_current_residual_x_m","parent_current_residual_y_m",
 "parent_current_residual_z_m","parent_registration_distance_m","parent_registration_tie_count",
 "incident_material_mask","geometric_donor_body","anatomical_region_id"]
PARENT_NODE_HEADER=["parent_node_index","compiled_global_node","current_x_m","current_y_m",
 "current_z_m","rest_x_m","rest_y_m","rest_z_m","compiled_lumped_mass_kg",
 "node_frame_owner_body","donor_local_x_m","donor_local_y_m","donor_local_z_m","mass_bits"]
REG_HEADER=["layer","surface_vertex_row","parent_registration_kind","parent_tet_row",
 "parent_bary0","parent_bary1","parent_bary2","parent_bary3","parent_residual_x_m",
 "parent_residual_y_m","parent_residual_z_m","parent_registration_distance_m",
 "parent_registration_tie_count","frame_owner_method","source_domain","source_vertex_row",
 "source_face_row","source_bary0","source_bary1","source_bary2","source_residual_x_m",
 "source_residual_y_m","source_residual_z_m"]
FRAME_HEADER=(["binding","donor","current_x","current_y","current_z"]+
 [f"current_R{i}{j}" for i in range(3) for j in range(3)]+
 ["rest_x","rest_y","rest_z"]+[f"rest_R{i}{j}" for i in range(3) for j in range(3)])
MATERIAL_MASK={15:1<<15,16:1<<16}
SOURCE_ABSENCE="derived_volume_interior_no_surface_ancestry"
NO_SOURCE_METHOD="derived_full_all_bodies_geometric_donor_pose"

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()

def u(value:Any,context:str,invalid=False)->int:
 try: x=int(value,10) if isinstance(value,str) else int(value)
 except (TypeError,ValueError) as e: raise ValueError(f"{context} is not an integer: {value!r}") from e
 if x<0 or x>INVALID or (x==INVALID and not invalid): raise ValueError(f"{context} outside uint32")
 return x

def ou(value:str,context:str)->int:
 return INVALID if value in ("","-1",str(INVALID)) else u(value,context)

def f(value:Any,context:str)->float:
 try: x=float(value)
 except (TypeError,ValueError) as e: raise ValueError(f"{context} is not numeric") from e
 if not math.isfinite(x): raise ValueError(f"{context} is nonfinite")
 return x

def fs(x:float)->str: return format(float(x),".17g")

def csvread(path:Path,header:list[str]|None=None):
 with path.open("r",encoding="utf-8",newline="") as stream:
  reader=csv.DictReader(stream)
  if reader.fieldnames is None or (header is not None and reader.fieldnames!=header):
   raise ValueError(f"unexpected CSV header at {path}: {reader.fieldnames}")
  rows=list(reader)
 if any(None in r for r in rows): raise ValueError(f"malformed CSV width at {path}")
 return rows

def label_module(directory:Path):
 p=directory/"export_child_semantic_labels_217.py"
 spec=importlib.util.spec_from_file_location("labels_owner_217_node_registration",p)
 if spec is None or spec.loader is None: raise ValueError(f"cannot load existing label owner {p}")
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 return m

def read_parent_current(base:Path,label_dir:Path):
 mod=label_module(label_dir); parent=mod.read_parent(base)
 rows=csvread(Path(str(base)+".nodes.csv"),PARENT_NODE_HEADER)
 if len(rows)!=len(parent["points"]): raise ValueError("parent current node count differs from validated parent packet")
 current=np.empty_like(parent["points"],dtype=np.float64)
 for i,r in enumerate(rows):
  if u(r["parent_node_index"],f"parent node {i}")!=i: raise ValueError("parent node rows are not contiguous")
  current[i]=[f(r[f"current_{a}_m"],f"parent current node {i}") for a in "xyz"]
 p=dict(parent);p["points"]=current
 return p,mod

def read_child(node:Path,ele:Path,labels:Path):
 def records(path):
  with path.open("r",encoding="utf-8") as stream:
   for line_no,line in enumerate(stream,1):
    s=line.partition("#")[0].strip()
    if s: yield line_no,s.split()
 it=iter(records(node))
 try: _,h=next(it)
 except StopIteration as e: raise ValueError("TetGen .node empty") from e
 if len(h)!=4: raise ValueError("TetGen .node header width")
 n,dim,attrs,markers=map(int,h)
 if not 4<=n<=MAX_NODES or dim!=3 or attrs!=0 or markers not in (0,1): raise ValueError("TetGen .node header invalid")
 points=np.empty((n,3),dtype=np.float32);lookup={}
 for row in range(n):
  try: line,v=next(it)
  except StopIteration as e: raise ValueError("TetGen .node truncated") from e
  if len(v)!=4+markers: raise ValueError(f"TetGen .node row {line} width")
  node_id=u(v[0],f"joint node ID {line}")
  if node_id!=row+1 or node_id in lookup: raise ValueError("joint TetGen IDs must be unique ordered 1-based rows")
  lookup[node_id]=row
  points[row]=[np.float32(f(x,f"node {row} coord")) for x in v[1:4]]
  if markers: int(v[4])
 if next(it,None) is not None: raise ValueError("TetGen .node has extra rows")
 it=iter(records(ele))
 try: _,h=next(it)
 except StopIteration as e: raise ValueError("TetGen .ele empty") from e
 if len(h)!=3: raise ValueError("TetGen .ele header width")
 cells,corners,nattrs=map(int,h)
 if not 1<=cells<=MAX_CELLS or corners!=4 or nattrs!=1: raise ValueError("TetGen .ele domain invalid")
 tets=[];materials=[];seen_ids=set();seen_cells=set()
 for row in range(cells):
  try: line,v=next(it)
  except StopIteration as e: raise ValueError("TetGen .ele truncated") from e
  if len(v)!=6: raise ValueError(f"TetGen .ele row {line} width")
  eid=u(v[0],f"element ID {line}")
  if eid in seen_ids: raise ValueError("duplicate TetGen element ID")
  seen_ids.add(eid)
  ids=[u(x,f"element {eid} node ID") for x in v[1:5]]
  if any(x not in lookup for x in ids): raise ValueError(f"element {eid} references absent node ID")
  tet=tuple(lookup[x] for x in ids)
  if len(set(tet))!=4 or tuple(sorted(tet)) in seen_cells: raise ValueError("repeated node or duplicate TetGen cell")
  xyz=points[np.asarray(tet)].astype(np.float64)
  vol=float(np.dot(xyz[1]-xyz[0],np.cross(xyz[2]-xyz[0],xyz[3]-xyz[0]))/6.0)
  if not math.isfinite(vol) or vol<=0: raise ValueError(f"TetGen cell {eid} is nonpositive after F32 emission")
  material=f(v[5],f"element {eid} material")
  if material not in MATERIAL_MASK or int(material)!=material: raise ValueError("unsupported TetGen material")
  seen_cells.add(tuple(sorted(tet)));tets.append(tet);materials.append(int(material))
 if next(it,None) is not None: raise ValueError("TetGen .ele has extra rows")
 label_rows=csvread(labels,["child_tet_row","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"])
 if len(label_rows)!=cells: raise ValueError("child labels and TetGen cells differ in count")
 tuples=[]
 for row,r in enumerate(label_rows):
  if u(r["child_tet_row"],f"child label {row}")!=row: raise ValueError("child labels not in exact tet row order")
  material=u(r["material_index"],f"child material {row}");donor=u(r["geometric_donor_body"],f"child donor {row}")
  region=u(r["anatomical_region_id"],f"child region {row}");face=u(r["stable_source_face_id"],f"child face {row}")
  if material!=materials[row] or donor==INVALID or region==INVALID or face==INVALID: raise ValueError(f"invalid semantic labels at child cell {row}")
  tuples.append((material,donor,region,face))
 incident=[[] for _ in range(n)]
 for cell,(tet,label) in enumerate(zip(tets,tuples)):
  for node_row in tet: incident[node_row].append(label)
 return {"points":points,"lookup":lookup,"tets":tets,"materials":materials,"labels":tuples,"incident":incident}

def read_poses(path:Path,frame_path:Path):
 data=json.loads(path.read_text(encoding="utf-8"));raw=data.get("all_bodies")
 if not isinstance(raw,list) or len(raw)!=157: raise ValueError("expected exact native all_bodies 157-pose table")
 poses=[]
 native_rows=0
 for i,row in enumerate(raw):
  if u(row.get("body_index"),f"all_bodies body {i}")!=i: raise ValueError("all_bodies is not exact body-index order")
  q={}
  for phase in ("current","rest"):
   x=row.get(phase,{})
   origin_raw=np.asarray(x.get("position_world_m"),dtype=np.float64)
   quat=np.asarray(x.get("orientation_world_xyzw"),dtype=np.float64)
   if origin_raw.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(origin_raw).all() or not np.isfinite(quat).all():
    raise ValueError(f"all_bodies {phase} pose {i} invalid")
   matrix_raw=x.get("local_to_world_row_major_f32")
   if matrix_raw is not None:
    matrix=np.asarray(matrix_raw,dtype=np.float64)
    if matrix.shape!=(9,) or not np.isfinite(matrix).all():
     raise ValueError(f"all_bodies {phase} native matrix {i} invalid")
    matrix_f32=matrix.astype(np.float32).astype(np.float64)
    origin_f32=origin_raw.astype(np.float32).astype(np.float64)
    if not np.array_equal(matrix_f32,matrix):
     raise ValueError(f"all_bodies {phase} native pose {i} is not exact Float32")
    R=matrix_f32.reshape(3,3);origin=origin_f32;native_rows+=1
   else:
    raise ValueError(f"all_bodies {phase} pose {i} lacks authoritative native Float32 local_to_world matrix")
   q[phase+"_origin"]=origin;q[phase+"_R"]=R;q[phase+"_world_to_local"]=np.linalg.inv(R)
  q["binding"]=INVALID;poses.append(q)
 bindings={};used=set()
 for i,r in enumerate(csvread(frame_path,FRAME_HEADER)):
  binding=u(r["binding"],f"registered binding {i}");donor=u(r["donor"],f"registered donor {i}")
  if donor>=157 or donor in bindings or binding in used: raise ValueError("duplicate registered donor or binding")
  p=poses[donor];bindings[donor]=binding;used.add(binding)
  for phase in ("current","rest"):
   origin=np.asarray([f(r[f"{phase}_{a}"],f"{phase} origin {donor}") for a in "xyz"])
   R=np.asarray([f(r[f"{phase}_R{i}{j}"],f"{phase} rotation {donor}") for i in range(3) for j in range(3)]).reshape(3,3)
   if not np.array_equal(origin,p[phase+"_origin"]) or not np.array_equal(R,p[phase+"_R"]):
    raise ValueError(f"registered donor frame {donor} differs from exact all_bodies {phase} pose")
  p["binding"]=binding
 return poses,bindings

def matches_current_world_to_local(candidate, pose):
 """Check a sidecar against the authoritative inverse of native Float32 R."""
 stored=np.asarray(candidate,dtype=np.float64)
 expected=np.asarray(pose["current_world_to_local"],dtype=np.float64)
 return (stored.shape==(3,3) and np.isfinite(stored).all() and
         np.allclose(stored,expected,rtol=0.0,atol=2e-10))

CURRENT_TYPED_SOURCE_DOMAIN = "derived_registered_skin_authoring"
CURRENT_FACE_FIELDS = ("face_source_layer", "face_source_body", "face_source_bone",
                       "face_source_face_id", "face_source_original_id",
                       "face_source_identity0", "face_source_identity1")


def verify_file_pin(pin, label):
    if not isinstance(pin, dict):
        raise ValueError(f"{label} pin is absent or malformed")
    path=Path(str(pin.get("path",""))).resolve()
    expected=str(pin.get("sha256",""))
    if len(expected)!=64 or not path.is_file() or sha(path)!=expected:
        raise ValueError(f"{label} file pin mismatch: {path}")
    return path

def same_file_pin(a,b):
    return (isinstance(a,dict) and isinstance(b,dict) and
            Path(str(a.get("path",""))).resolve()==Path(str(b.get("path",""))).resolve() and
            a.get("sha256")==b.get("sha256"))

def validate_current_230_source_gate(receipt_path:Path, receipt, expected_receipt_sha256:str,
                                     expected_source_proof_path:Path, expected_source_proof_sha256:str,
                                     expected_current_state_path:Path, expected_current_state_sha256:str,
                                     expected_parent_base:Path, expected_poses_path:Path, role_pins):
    receipt_path=receipt_path.resolve()
    if len(expected_receipt_sha256)!=64 or sha(receipt_path)!=expected_receipt_sha256:
        raise ValueError("current 230 sidecar receipt pin mismatch")
    proof_obj=receipt.get("currentTypedShellProof")
    if not isinstance(proof_obj,dict) or proof_obj.get("schema")!="numi.h706.current-230-source-proof.v1":
        raise ValueError("current 230 receipt lacks its typed source-proof schema")
    expected_source_proof_path=expected_source_proof_path.resolve()
    source_pin=proof_obj.get("sourceProof")
    if (not isinstance(source_pin,dict) or Path(str(source_pin.get("path",""))).resolve()!=expected_source_proof_path or
        source_pin.get("sha256")!=expected_source_proof_sha256 or len(expected_source_proof_sha256)!=64 or
        not expected_source_proof_path.is_file() or sha(expected_source_proof_path)!=expected_source_proof_sha256):
        raise ValueError("current 230 receipt does not bind the exact supplied 808 source proof")
    proof=json.loads(expected_source_proof_path.read_text(encoding="utf-8"))
    expected_state_path=expected_current_state_path.resolve()
    state_pin=proof.get("state")
    if (not isinstance(state_pin,dict) or Path(str(state_pin.get("path",""))).resolve()!=expected_state_path or
        state_pin.get("sha256")!=expected_current_state_sha256 or len(expected_current_state_sha256)!=64 or
        not expected_state_path.is_file() or sha(expected_state_path)!=expected_current_state_sha256 or
        proof_obj.get("state")!=state_pin):
        raise ValueError("808 proof is not bound to the exact current operational volume state")
    if (proof.get("status")!="PASS_CURRENT_EXACT_SOURCE_EVIDENCE" or
        proof.get("physical_admission") is not False or proof.get("native_steps")!=0 or
        proof.get("inherited_stale_pointer") is not None):
        raise ValueError("pinned 808 source proof is not a clean source-only PASS")
    all185=proof.get("all185")
    ranges=all185.get("ranges_by_domain",{}) if isinstance(all185,dict) else {}
    f32=ranges.get("source_order_native_f32_replay",{})
    f64=ranges.get("pinned_f64_exact",{})
    if (not isinstance(all185,dict) or all185.get("schema")!="existing-bone-record-remesh-barrier.v1" or
        int(all185.get("record_count",-1))!=938272 or
        int(all185.get("changed_self_count",-1))!=0 or int(all185.get("changed_vs_all185_count",-1))!=0 or
        f32!={"start":0,"end":469136,"records":185} or
        f64!={"start":469136,"end":938272,"records":185}):
        raise ValueError("pinned 808 all185 receipt does not prove the exact zero-change 185-row domains")
    if proof_obj.get("all185")!=all185:
        raise ValueError("230 embedded all185 summary differs from the pinned 808 proof")
    report_pin=proof.get("report")
    report_path=verify_file_pin(report_pin,"808 report")
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status")!="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185":
        raise ValueError("pinned 806 report did not pass exact O/I/D and walls/all185 audit")
    self_pairs=report.get("changed_OID_joint_self_pairs",{})
    retained_pairs=report.get("changed_vs_complete_retained_OID_both_walls_all185",{})
    if (self_pairs.get("triangle_pairs")!=[] or int(self_pairs.get("count",-1))!=0 or
        retained_pairs.get("triangle_pairs")!=[] or int(retained_pairs.get("count",-1))!=0):
        raise ValueError("pinned 806 report contains a self or retained-wall/all185 crossing")
    summary=report.get("all185_receipt",{})
    for key in ("schema","record_count","native_sha256","pinned_manifest_sha256"):
        if summary.get(key)!=all185.get(key):
            raise ValueError(f"pinned 806 report all185 {key} differs from 808")
    for key in ("state","config","report"):
        if proof_obj.get(key)!=proof.get(key):
            raise ValueError(f"230 embedded {key} binding differs from pinned 808")
        verify_file_pin(proof.get(key),f"808 {key}")
    parent=receipt.get("parentCurrent",{})
    parent_base=Path(expected_parent_base).resolve()
    if Path(str(parent.get("massPath",""))).resolve()!=parent_base or parent.get("massSHA256")!=sha(parent_base):
        raise ValueError("230 current parent mass packet differs from the source330 path/hash")
    for key,path_key,hash_key in (("nodes","nodesPath","nodeSHA256"),("source","sourcePath","sourceSHA256"),
                                  ("tets","tetsPath","tetsSHA256"),("labels","labelsPath","labelsSHA256")):
        path=Path(str(parent.get(path_key,""))).resolve()
        if not path.is_file() or sha(path)!=parent.get(hash_key):
            raise ValueError(f"230 source330 parent {key} packet path/hash mismatch")
    if parent.get("nodeCount")!=7233 or parent.get("tetCount")!=28632 or parent.get("currentPositiveTets")!=28632:
        raise ValueError("230 source330 parent row/material domain differs from pinned source")
    w86=receipt.get("parentFull86Weights",{})
    wpath=Path(str(w86.get("path",""))).resolve()
    wnodes=Path(str(w86.get("parentNodesPath",""))).resolve()
    if (not wpath.is_file() or sha(wpath)!=w86.get("sha256") or wpath!=Path(str(parent_base)+".weights.csv") or
        not wnodes.is_file() or sha(wnodes)!=w86.get("parentNodesSHA256") or wnodes!=Path(str(parent_base)+".nodes.csv") or
        w86.get("nodeCount")!=parent.get("nodeCount") or w86.get("bindingCount")!=86 or
        w86.get("rowDomain")!="parent_node_index"):
        raise ValueError("230 source330 full86 weight table is not bound to the exact parent node domain")
    pose=receipt.get("poseSnapshot",{})
    poses_path=Path(expected_poses_path).resolve()
    if (Path(str(pose.get("path",""))).resolve()!=poses_path or not poses_path.is_file() or
        sha(poses_path)!=pose.get("sha256") or pose.get("frameSourceMethod")!="native_all_bodies_f32_local_to_world_matrix_and_inverse" or
        pose.get("nativeMatrixFrameBodyCount")!=86 or pose.get("snapshotBindingCount")!=86):
        raise ValueError("230 all_bodies pose snapshot is not the exact native Float32 frame source")
    audit=proof.get("audit_implementation",{})
    for key in ("audit806","exact_owner","filtered_owner","audit_invocation"):
        pin=audit.get(key)
        verify_file_pin(pin,f"808 audit implementation {key}")
        if key=="audit_invocation":
            for stream in ("stdout","stderr"):
                verify_file_pin(pin.get(stream),f"808 audit invocation {stream}")
    for role in ("O","I","D"):
        source=proof.get("source_surfaces",{}).get(role,{})
        expected_path,expected_sha=role_pins[role]
        if (Path(str(source.get("path",""))).resolve()!=Path(expected_path).resolve() or
            source.get("sha256")!=expected_sha):
            raise ValueError(f"808 proof current {role} source pin differs from requested current candidate")
    shell_receipt_pin=proof_obj.get("shellReceipt")
    shell_binding_pin=proof_obj.get("shellBinding")
    handoff_pin=proof_obj.get("handoffReceipt")
    crosswalk_pin=proof_obj.get("crosswalk")
    shell_receipt_path=verify_file_pin(shell_receipt_pin,"current 1022 shell receipt")
    shell_binding_path=verify_file_pin(shell_binding_pin,"current 1022 shell binding")
    handoff_path=verify_file_pin(handoff_pin,"current 938 handoff receipt")
    crosswalk_path=verify_file_pin(crosswalk_pin,"current 938 role crosswalk")
    shell_receipt=json.loads(shell_receipt_path.read_text(encoding="utf-8"))
    shell_binding=json.loads(shell_binding_path.read_text(encoding="utf-8"))
    handoff=json.loads(handoff_path.read_text(encoding="utf-8"))
    if (shell_receipt.get("schema")!="numi.h706.current-paired-shell-1022.v1" or
        shell_receipt.get("status")!="PASS_CURRENT_938V2_PAIRED_SHELLS_SOURCE_ONLY" or
        shell_receipt.get("physical_admission") is not False or shell_receipt.get("native_steps")!=0 or
        not same_file_pin(shell_receipt.get("current_state"),state_pin)):
        raise ValueError("pinned current 1022 shell receipt is not bound to this source-only state")
    if (shell_binding.get("status")!="PASS_938V2_CURRENT_SOURCE_SHELL_RECEIPT_BINDING" or
        shell_binding.get("physical_admission") is not False or shell_binding.get("native_steps")!=0 or
        not same_file_pin(shell_binding.get("state"),state_pin) or not same_file_pin(shell_binding.get("receipt"),shell_receipt_pin) or
        any(not same_file_pin(shell_binding.get("source_surfaces",{}).get(role),proof.get("source_surfaces",{}).get(role)) for role in ("O","I","D"))):
        raise ValueError("current 1022 shell binding does not bind exact state, receipt, and O/I/D sources")
    if (handoff.get("schema")!="numi.h706.current-fine-217-combined-volume-handoff.v1" or
        handoff.get("status")!="PREPARED_EXISTING_217_COMBINED_CHILD_VOLUME_HANDOFF_NO_PHYSICAL_ADMISSION" or
        handoff.get("physical_admission") is not False or handoff.get("native_steps")!=0):
        raise ValueError("current 938 handoff is not the pinned non-admitting source handoff")
    if not same_file_pin(handoff.get("inputs",{}).get("current_state"),state_pin):
        raise ValueError("current 938 handoff does not bind the same operational state as 808")
    mesh=handoff.get("combined_skin_fat_child_mesh",{})
    if (mesh.get("nodes")!=311768 or mesh.get("combined_tetrahedra")!=1495294 or
        mesh.get("registered_surface_node_map_complete") is not True):
        raise ValueError("current 938 handoff mesh/node-map scope differs from pinned full-domain contract")
    map_name="skin-fat-combined-registered-surface-node-map.npz"
    map_hash=handoff.get("combined_skin_fat_child_mesh",{}).get("outputs",{}).get(map_name)
    if map_hash!=crosswalk_pin.get("sha256"):
        raise ValueError("current 938 handoff output does not bind the supplied role crosswalk")
    handoff_outputs=handoff.get("combined_skin_fat_child_mesh",{}).get("outputs",{})
    handoff_dir=handoff_path.parent
    handoff_output_paths={}
    for name in ("skin-fat-combined-node-map.csv.gz","skin-fat-combined.node","skin-fat-combined.ele",map_name):
        path=(handoff_dir/name).resolve()
        if not path.is_file() or handoff_outputs.get(name)!=sha(path):
            raise ValueError(f"current 938 materialized handoff output differs from its receipt: {name}")
        handoff_output_paths[name]=str(path)
    with np.load(crosswalk_path,allow_pickle=False) as z:
        if (str(np.asarray(z["schema"]).item())!="numi.h706.217-registered-surface-node-map.v1" or
            int(np.asarray(z["inactive_sentinel"]).item())!=-1):
            raise ValueError("current 938 role crosswalk schema/sentinel is invalid")
        source_domains=proof_obj.get("source_row_domains",{})
        active_rows={}
        for role in ("O","I","D"):
            role_path,role_sha=role_pins[role]
            with np.load(role_path,allow_pickle=False) as role_npz:
                faces=np.asarray(role_npz["faces"],dtype=np.int64)
                active=np.unique(faces.reshape(-1))
                n=len(np.asarray(role_npz["points_world_m_f32"])) if "points_world_m_f32" in role_npz.files else len(role_npz["points_world_m"])
            inactive=np.setdiff1d(np.arange(n,dtype=np.int64),active,assume_unique=True)
            map_key=f"{role}_source_row_to_joint_node"
            active_key=f"{role}_active_source_vertex_rows"
            inactive_key=f"{role}_inactive_source_vertex_rows"
            mapped=np.asarray(z[map_key],dtype=np.int64)
            cw_active=np.asarray(z[active_key],dtype=np.int64)
            cw_inactive=np.asarray(z[inactive_key],dtype=np.int64)
            domain=source_domains.get(role,{})
            active_sha=hashlib.sha256(active.astype("<i8").tobytes()).hexdigest()
            inactive_sha=hashlib.sha256(inactive.astype("<i8").tobytes()).hexdigest()
            if (mapped.shape!=(n,) or not np.array_equal(active,cw_active) or
                not np.array_equal(inactive,cw_inactive) or np.any(mapped[active]<0) or
                np.any(mapped[inactive]!=-1) or len(np.unique(mapped[active]))!=len(active) or
                domain.get("full_rows")!=n or domain.get("active_rows")!=len(active) or
                domain.get("inactive_rows")!=len(inactive) or domain.get("active_ids_sha256")!=active_sha or
                domain.get("inactive_ids_sha256")!=inactive_sha):
                raise ValueError(f"current 938 active/inactive row crosswalk differs from exact {role} face references")
            active_rows[role]=int(len(active))
        # Preserve the historical sparse-row example as diagnostic evidence only.
        # Production admission is governed by the complete face-derived active/inactive
        # crosswalk above, so no particular source row or joint-node number is privileged.
        o_map=np.asarray(z["O_source_row_to_joint_node"],dtype=np.int64)
        o_active=np.asarray(z["O_active_source_vertex_rows"],dtype=np.int64)
        o_active_set=set(map(int,o_active))
        source_553_control={
            "row_552": {"joint_node_id": int(o_map[552]) if len(o_map)>552 else None,
                        "face_active": 552 in o_active_set},
            "row_553": {"joint_node_id": int(o_map[553]) if len(o_map)>553 else None,
                        "face_active": 553 in o_active_set},
            "interpretation": "diagnostic_only_general_crosswalk_is_authoritative"
        }
    return {"schema":"numi.h706.current-230-source-gate-authentication.v2",
            "sidecar_receipt_path":str(receipt_path),"sidecar_receipt_sha256":expected_receipt_sha256,
            "source_proof_path":str(expected_source_proof_path),
            "source_proof_sha256":expected_source_proof_sha256,"source_proof_status":proof["status"],
            "current_state_path":str(expected_state_path),"current_state_sha256":expected_current_state_sha256,
            "audit_report_path":str(report_path),"audit_report_sha256":report_pin["sha256"],
            "audit_report_status":report["status"],"all185_record_count":all185["record_count"],
            "all185_changed_self_count":all185["changed_self_count"],
            "all185_changed_vs_all185_count":all185["changed_vs_all185_count"],
            "self_triangle_pairs":0,"retained_walls_all185_triangle_pairs":0,
            "active_source_rows":active_rows,"handoff_output_paths":handoff_output_paths,
            "parent_full86_weights":{"path":str(wpath),"sha256":w86["sha256"],"nodes":w86["nodeCount"],"bindings":w86["bindingCount"]},
            "pose_snapshot":{"path":str(poses_path),"sha256":pose["sha256"],"frame_source_method":pose["frameSourceMethod"],"registered_bodies":pose["registeredPoseBodyCount"]},
            "source_553_control":source_553_control,
            "physical_admission":False,"native_steps":0}

def validate_current_typed_sidecar_bundle(directory:Path, role_pins, maximum_parent_distance:float,
                                            expected_source_proof=None, expected_receipt_sha256=None, expected_current_state=None,
                                            expected_parent_base=None, expected_poses_path=None):
    """Authenticate 230 current-typed role rows and their exact emitted sidecars.

    Historical parent registrations remain provenance. The current 433 route later
    creates the exact child self-tet operational map; this check does not qualify it.
    """
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "registered-authoring-surface-sidecars-receipt.v1" or
        receipt.get("status") != "reference_registration_ready_geometry_not_admitted" or
        receipt.get("physical_admission") is not False or
        receipt.get("geometry_admission") is not False or receipt.get("native_steps") != 0):
        raise ValueError("current typed 228 sidecar receipt is not a non-admitting reference export")
    limits = receipt.get("limits", {})
    if (not math.isfinite(float(maximum_parent_distance)) or maximum_parent_distance <= 0 or
        float(limits.get("maximumParentRegistrationDistanceM", -1.0)) != float(maximum_parent_distance)):
        raise ValueError("current typed historical support limit differs from its 230 receipt")
    source_gate=None
    gate = receipt.get("exactGeometryGateReceipt")
    if gate is not None:
        if (gate.get("status") != "PASS_WHOLE_CURRENT_OID_VS_ALL185_F32_F64" or
            int(gate.get("all185Pairs", -1)) != 0 or
            any(int(gate.get("selfPairs", {}).get(role, -1)) != 0 for role in ("O", "I", "D")) or
            any(int(gate.get("crossPairs", {}).get(pair, -1)) != 0 for pair in ("O-I", "O-D", "I-D"))):
            raise ValueError("current typed sidecar receipt lacks the pinned whole-surface geometry gate")
    else:
        if expected_source_proof is None or expected_receipt_sha256 is None or expected_current_state is None:
            raise ValueError("current 230 source proof, sidecar receipt SHA, and operational state pins are required")
        source_gate=validate_current_230_source_gate(
            receipt_path,receipt,expected_receipt_sha256,
            Path(expected_source_proof[0]),str(expected_source_proof[1]),
            Path(expected_current_state[0]),str(expected_current_state[1]),
            Path(expected_parent_base),Path(expected_poses_path),role_pins)
    candidates = receipt.get("candidateSurfaces", {})
    output_hashes = receipt.get("outputSHA256", {})
    expected_roles = {"O", "I", "D"}
    if set(role_pins) != expected_roles or set(candidates) != expected_roles:
        raise ValueError("current typed sidecars require exact O/I/D role coverage")
    for name in ("registered-O.csv", "registered-I.csv", "registered-D.csv",
                 "registered-O-faces.csv", "registered-I-faces.csv", "registered-D-faces.csv",
                 "node-parent-registration.csv", "donor-frames.csv", "source-domain-registry.json"):
        path = directory / name
        if not path.is_file() or output_hashes.get(name) != sha(path):
            raise ValueError(f"current typed sidecar output hash differs from receipt: {name}")
    registry = json.loads((directory / "source-domain-registry.json").read_text(encoding="utf-8"))
    registry_by_role = {str(item.get("roleIndex")): item for item in registry}
    role_indices = {"O": "1", "I": "2", "D": "3"}
    role_names = {"O":"outerSurface","I":"innerSurface","D":"deepSurface"}
    result = {}
    for role in ("O", "I", "D"):
        path, expected_sha = role_pins[role]
        path = Path(path).resolve()
        if not path.is_file() or len(expected_sha) != 64 or sha(path) != expected_sha:
            raise ValueError(f"current typed {role} candidate hash mismatch")
        spec = candidates[role]
        if (Path(spec.get("path", "")).resolve() != path or spec.get("sha256") != expected_sha or
            spec.get("points", -1) <= 0 or spec.get("faces", -1) <= 0):
            raise ValueError(f"current typed {role} candidate pin differs from 230 receipt")
        domain = registry_by_role.get(role_indices[role])
        if (domain is None or domain.get("role") != role_names[role] or
            domain.get("sourceSHA256") != expected_sha or domain.get("vertexCount") != spec["points"] or
            domain.get("faceCount") != spec["faces"]):
            raise ValueError(f"current typed {role} source domain registry differs")
        with np.load(path, allow_pickle=False) as z:
            for key in ("points_world_m", "faces", *CURRENT_FACE_FIELDS):
                if key not in z.files:
                    raise ValueError(f"current typed {role} candidate lacks {key}")
            points = np.asarray(z["points_world_m"])
            faces = np.asarray(z["faces"], dtype=np.int64)
            if (points.ndim != 2 or points.shape[1] != 3 or len(points) != int(spec["points"]) or
                not np.isfinite(points).all() or not np.array_equal(points, points.astype(np.float32).astype(points.dtype)) or
                faces.shape != (int(spec["faces"]), 3) or np.any(faces < 0) or np.any(faces >= len(points))):
                raise ValueError(f"current typed {role} candidate geometry domain is invalid")
            typed = {key: np.asarray(z[key], dtype=np.int64).copy() for key in CURRENT_FACE_FIELDS}
        node_rows = csvread(directory / f"registered-{role}.csv")
        if len(node_rows) != len(points) or any(u(row.get("vertex"), f"{role} current vertex") != i for i,row in enumerate(node_rows)):
            raise ValueError(f"current typed {role} node sidecar is not complete contiguous row coverage")
        side_points = np.asarray([[f(row[f"w{axis}"], f"{role} row point") for axis in "xyz"] for row in node_rows], dtype=np.float64)
        if not np.array_equal(side_points, points.astype(np.float64)):
            raise ValueError(f"current typed {role} sidecar coordinates differ from exact current candidate rows")
        for i,row in enumerate(node_rows):
            if (row.get("source_domain") != CURRENT_TYPED_SOURCE_DOMAIN or row.get("source_mapping_kind") != "vertex" or
                ou(row.get("source_vertex_row", ""), f"{role} source vertex") != i or
                ou(row.get("source_face_row", ""), f"{role} source face") != INVALID or
                any(f(row[f"source_bary{k}"], f"{role} source bary") != 0.0 for k in range(3)) or
                any(f(row[f"source_residual_{axis}_m"], f"{role} source residual") != 0.0 for axis in "xyz")):
                raise ValueError(f"current typed {role} sidecar has nonidentity or false face ancestry at row {i}")
        face_rows = csvread(directory / f"registered-{role}-faces.csv")
        if len(face_rows) != len(faces):
            raise ValueError(f"current typed {role} face sidecar row count differs")
        for i,row in enumerate(face_rows):
            if u(row.get("face"), f"{role} face row") != i or [u(row[f"v{k}"], f"{role} face corner") for k in range(3)] != list(map(int,faces[i])):
                raise ValueError(f"current typed {role} face connectivity differs at row {i}")
            for key in CURRENT_FACE_FIELDS:
                try: actual_value=int(row[key],10)
                except (TypeError,ValueError) as exc: raise ValueError(f"malformed {role} {key} at row {i}") from exc
                if actual_value != int(typed[key][i]):
                    raise ValueError(f"current typed {role} face identity differs at row {i}: {key}")
        result[role] = {"candidate_path": str(path), "candidate_sha256": expected_sha,
                        "vertex_rows": len(points), "face_rows": len(faces),
                        "active_source_vertex_rows": np.unique(faces.reshape(-1)).astype(int).tolist(),
                        "inactive_source_vertex_rows": np.setdiff1d(np.arange(len(points),dtype=np.int64),
                            np.unique(faces.reshape(-1)),assume_unique=True).astype(int).tolist()}
    return {"schema": "numi.current-typed-230-sidecar-authorization.v1-active-full-source-rows",
            "roles": result, "receipt_path": str(receipt_path), "receipt_sha256": sha(receipt_path),
            "source_gate": source_gate,
            "historical_support_maximum_distance_m": float(maximum_parent_distance),
            "historical_registration_is_provenance": True,
            "current_operational_map": "created later as exact one-hot child self-tets by 433 opt-in",
            "source330_to_original_source_bound_changed": False,
            "conservative_mass_transport_bound_changed": False,
            "physical_admission": False}


def read_surfaces(directory:Path,current_typed=False,active_rows_by_role=None):
 out={}
 for layer in ("O","I","D"):
  rows=csvread(directory/f"registered-{layer}.csv")
  required={"vertex","donor","wx","wy","wz","lx","ly","lz","m00","m01","m02","m10","m11","m12","m20","m21","m22","source_domain","frame_owner_method"}
  if current_typed:
   required|={"source_mapping_kind","source_vertex_row","source_face_row","source_bary0","source_bary1","source_bary2","source_residual_x_m","source_residual_y_m","source_residual_z_m"}
  elif layer=="O": required|={"source_vertex_row","source_face_row","source_bary0","source_bary1","source_bary2","source_residual_x_m","source_residual_y_m","source_residual_z_m"}
  else: required|={"d37_parent_node_row","d37_parent_surface_face_row","beta0","beta1","beta2","source_residual_x_m","source_residual_y_m","source_residual_z_m"}
  if not rows or any(not required.issubset(r) for r in rows): raise ValueError(f"registered {layer} rows missing required existing columns")
  active=(set(range(len(rows))) if not current_typed or active_rows_by_role is None
          else set(map(int,active_rows_by_role[layer])))
  if any(i<0 or i>=len(rows) for i in active): raise ValueError(f"{layer} active source row outside full sidecar domain")
  parsed=[]
  for i,r in enumerate(rows):
   if u(r["vertex"],f"{layer} surface vertex")!=i: raise ValueError(f"{layer} surface vertex order is not contiguous")
   point=np.asarray([f(r["w"+a],f"{layer} point") for a in "xyz"])
   if current_typed:
    sv=ou(r["source_vertex_row"],f"{layer} current source vertex")
    sf=ou(r["source_face_row"],f"{layer} current source face")
    bary=np.asarray([f(r[f"source_bary{k}"],f"{layer} source bary") for k in range(3)])
    residual=np.asarray([f(r[f"source_residual_{axis}_m"],f"{layer} source residual") for axis in "xyz"])
    if (r["source_domain"]!=CURRENT_TYPED_SOURCE_DOMAIN or r["source_mapping_kind"]!="vertex" or
        sv!=i or sf!=INVALID or np.any(bary!=0.0) or np.any(residual!=0.0)):
     raise ValueError(f"{layer} current typed sidecar must retain exact full source row identity at {i}")
    if i in active:
     donor=u(r["donor"],f"{layer} active donor")
     local=np.asarray([f(r["l"+a],f"{layer} active local") for a in "xyz"])
     matrix=np.asarray([f(r[f"m{x}{y}"],f"{layer} active frame") for x in range(3) for y in range(3)]).reshape(3,3)
     frame_method=r["frame_owner_method"]
     if not frame_method: raise ValueError(f"{layer} active row {i} lacks a frame owner")
    else:
     # Inactive source-only rows must not carry a physical donor/frame registration.
     blank=("donor","lx","ly","lz","m00","m01","m02","m10","m11","m12","m20","m21","m22","frame_owner_method")
     if any(r.get(k,"")!="" for k in blank):
      raise ValueError(f"{layer} inactive provenance row {i} contains physical donor/frame data")
     donor=None;local=None;matrix=None;frame_method=""
   else:
    donor=u(r["donor"],f"{layer} donor")
    local=np.asarray([f(r["l"+a],f"{layer} local") for a in "xyz"])
    matrix=np.asarray([f(r[f"m{x}{y}"],f"{layer} frame") for x in range(3) for y in range(3)]).reshape(3,3)
    frame_method=r["frame_owner_method"]
    if layer=="O":
     sv=ou(r["source_vertex_row"],"O source vertex");sf=ou(r["source_face_row"],"O source face")
     bary=np.asarray([f(r[f"source_bary{k}"],"O source bary") for k in range(3)])
    else:
     sv=ou(r["d37_parent_node_row"],f"{layer} source vertex")
     sf=ou(r["d37_parent_surface_face_row"],f"{layer} source face")
     bary=np.asarray([f(r[f"beta{k}"],f"{layer} source bary") for k in range(3)])
    residual=np.asarray([f(r[f"source_residual_{axis}_m"],f"{layer} source residual") for axis in "xyz"])
    if (sv==INVALID)==(sf==INVALID): raise ValueError(f"{layer} vertex {i} needs exactly one direct or face identity")
    if sf!=INVALID and (np.min(bary)<-1e-8 or abs(float(bary.sum())-1)>1e-8): raise ValueError(f"{layer} source bary invalid")
   parsed.append({"row":r,"donor":donor,"point":point,"local":local,"matrix":matrix,
      "source_vertex":sv,"source_face":sf,"source_bary":bary,"source_residual":residual,
      "source_domain":r["source_domain"],"frame_method":frame_method,"physical_active":i in active})
  out[layer]=parsed
 return out

def read_parent_registrations(directory:Path,surfaces,parent,maxdist,current_typed=False,active_rows_by_role=None):
 path=directory/"node-parent-registration.csv"
 if current_typed:
  with path.open("r",encoding="utf-8",newline="") as stream:
   reader=csv.DictReader(stream)
   if reader.fieldnames is None or not set(REG_HEADER).issubset(reader.fieldnames): raise ValueError("current typed 230 registration sidecar lacks existing 217 registration columns")
   rows=list(reader)
  if any(None in r for r in rows): raise ValueError("malformed current typed 230 parent registration rows")
 else: rows=csvread(path,REG_HEADER)
 expected={}
 for layer in ("O","I","D"):
  expected[layer]=(set(range(len(surfaces[layer]))) if not current_typed or active_rows_by_role is None
                   else set(map(int,active_rows_by_role[layer])))
 out={}
 for line,r in enumerate(rows,2):
  layer=r["layer"];row=u(r["surface_vertex_row"],f"registration row {line}")
  if layer not in surfaces or row>=len(surfaces[layer]) or (layer,row) in out: raise ValueError("invalid/duplicate surface parent registration")
  if current_typed and row not in expected[layer]:
   raise ValueError(f"current typed 230 supplied a physical parent registration for inactive {layer} source row {row}")
  s=surfaces[layer][row]
  if r["source_domain"]!=s["source_domain"] or r["frame_owner_method"]!=s["frame_method"]: raise ValueError("registered parent source domain/method mismatch")
  sv=ou(r["source_vertex_row"],"registration source vertex");sf=ou(r["source_face_row"],"registration source face")
  if sv!=s["source_vertex"] or sf!=s["source_face"]: raise ValueError("registered parent source identity mismatch")
  barysrc=np.asarray([f(r[f"source_bary{k}"],"registration source bary") for k in range(3)])
  residualsrc=np.asarray([f(r[f"source_residual_{a}_m"],"registration source residual") for a in "xyz"])
  if not np.array_equal(barysrc,s["source_bary"]) or not np.array_equal(residualsrc,s["source_residual"]): raise ValueError("registered source embedding changed in parent sidecar")
  kind=r["parent_registration_kind"];tet=u(r["parent_tet_row"],"parent tet")
  if kind not in ("source_parent_tet_reference_registration","inferred_current_parent_tet_same_material_donor_region") or tet>=len(parent["tets"]): raise ValueError("unsupported/missing registered parent tet")
  beta=np.asarray([f(r[f"parent_bary{k}"],"parent beta") for k in range(4)])
  residual=np.asarray([f(r[f"parent_residual_{a}_m"],"parent residual") for a in "xyz"])
  distance=f(r["parent_registration_distance_m"],"parent distance");ties=u(r["parent_registration_tie_count"],"parent tie count")
  if np.min(beta)<-1e-8 or np.max(beta)>1+1e-8 or abs(float(beta.sum())-1)>1e-8 or distance<0 or distance>maxdist or abs(float(np.linalg.norm(residual))-distance)>2e-10:
   raise ValueError("registered current parent row fails barycentric/residual/distance bounds")
  out[(layer,row)]={"kind":kind,"tet":tet,"beta":beta,"residual":residual,"distance":distance,"ties":ties}
 actual={l:set() for l in ("O","I","D")}
 for layer,row in out: actual[layer].add(row)
 for layer in ("O","I","D"):
  if actual[layer]!=expected[layer]:
   missing=sorted(expected[layer]-actual[layer])[:8];extra=sorted(actual[layer]-expected[layer])[:8]
   raise ValueError(f"current typed parent registrations do not exactly cover active {layer} rows; missing={missing} extra={extra}")
 return out

def parent_tuple(parent,row): return (int(parent["materials"][row]),int(parent["donors"][row]),int(parent["regions"][row]))

def make_lookup(parent,owner):
 groups={}
 for row in range(len(parent["tets"])): groups.setdefault(parent_tuple(parent,row),[]).append(row)
 lookup={}
 for key,rows in groups.items():
  rr=np.asarray(rows,dtype=np.int64)
  sub={"points":parent["points"],"tets":parent["tets"][rr],"materials":parent["materials"][rr],
       "donors":parent["donors"][rr],"regions":parent["regions"][rr],"faces":parent["faces"][rr]}
  lookup[key]=(sub,owner.build_index(sub),rr)
 return lookup

def nearest_for_tuple(point,key,lookup,owner,maxdist,tol):
 if key not in lookup: raise ValueError(f"no current parent cells for semantic tuple {key}")
 parent,index,orig=lookup[key]
 center,seed=index["tree"].query(point,k=1)
 _,_,seedDist=owner.closest_point_on_parent_tet(parent,index,int(seed),point)
 cand=index["tree"].query_ball_point(point,seedDist+index["max_radius"]+tol+1e-12)
 measured=[]
 for local in cand:
  xyz,beta,d=owner.closest_point_on_parent_tet(parent,index,int(local),point)
  measured.append((d,int(orig[int(local)]),beta,xyz))
 if not measured: raise ValueError(f"no closest parent tet for tuple {key}")
 best=min(x[0] for x in measured)
 ties=sorted((x for x in measured if abs(x[0]-best)<=tol),key=lambda x:x[1])
 d,row,beta,xyz=ties[0]
 return {"tuple":key,"tet":row,"beta":beta,"registered":xyz,"residual":point-xyz,"distance":d,"ties":len(ties)-1}

class ParentSemanticResolutionBatchError(ValueError):
 def __init__(self,witnesses):
  self.witnesses=list(witnesses)
  self.counts={}
  for w in self.witnesses:
   role=str(w.get("role","unknown"));self.counts[role]=self.counts.get(role,0)+1
  super().__init__(f"no bounded current incident-semantic parent for {len(self.witnesses)} nodes: {self.counts}")

def select_incident_parent_registration(point,role,original_tuple,incident_set,lookup,owner,maxdist,tol,source_row,node_row):
 """Select a bounded current parent tet from exact child incident semantic tuples."""
 if role not in ("O","I","D"): raise ValueError(f"unsupported role for semantic parent selection: {role}")
 allowed=sorted(tuple(map(int,k)) for k in incident_set
                if len(k)==3 and (role=="I" or int(k[0])==(15 if role=="O" else 16)))
 candidates=[]
 for key in allowed:
  try: candidates.append(nearest_for_tuple(point,key,lookup,owner,maxdist,tol))
  except ValueError as exc:
   if str(exc).startswith("no current parent cells for semantic tuple"): continue
   raise
 bounded=[c for c in candidates if math.isfinite(float(c["distance"])) and float(c["distance"])<=maxdist]
 if not bounded:
  ordered=sorted(candidates,key=lambda c:(float(c["distance"]),tuple(c["tuple"]),int(c["tet"])))
  witness={"role":role,"source_row":int(source_row),"joint_node_row":int(node_row),
   "original_parent_semantic_tuple":list(map(int,original_tuple)),
   "incident_child_semantic_tuples":[list(k) for k in allowed],
   "nearest_parent_candidates":[{"tuple":list(map(int,c["tuple"])),"tet":int(c["tet"]),
      "distance_m":float(c["distance"])} for c in ordered[:8]],
   "maximum_parent_distance_m":float(maxdist),
   "failure":"no incident semantic parent candidate within declared maximum distance" if candidates else "no parent cells for any incident semantic tuple"}
  return None,witness
 if role=="I":
  bounded.sort(key=lambda c:(0 if int(c["tuple"][0])==int(original_tuple[0]) else 1,
                             float(c["distance"]),tuple(c["tuple"]),int(c["tet"])))
 else:
  bounded.sort(key=lambda c:(float(c["distance"]),tuple(c["tuple"]),int(c["tet"])))
 return bounded[0],None

def validate_active_surface_coverage(mapped,surfaces,active_rows_by_role=None):
 expected={}
 for layer in ("O","I","D"):
  expected[layer]=(set(range(len(surfaces[layer]))) if active_rows_by_role is None
                   else set(map(int,active_rows_by_role[layer])))
  if mapped[layer]!=expected[layer]:
   missing=sorted(expected[layer]-mapped[layer])[:8];extra=sorted(mapped[layer]-expected[layer])[:8]
   raise ValueError(f"{layer} map does not exactly cover face-active full source rows; missing={missing} extra={extra}")
 return {layer:len(expected[layer]) for layer in ("O","I","D")}

def build_rows(maprows,child,surfaces,registrations,parent,owner,poses,maxdist,tol,active_rows_by_role=None):
 n=len(child["points"]);per=[[] for _ in range(n)];source_ids=set()
 for r in maprows:
  jid=u(r["joint_node_id"],"joint node ID")
  if jid==0 or jid>n: raise ValueError("node map joint ID outside child domain")
  row=child["lookup"].get(jid)
  if row is None or row!=jid-1: raise ValueError("joint node IDs do not map to canonical TetGen node rows")
  partition=r["source_partition"];sid=u(r["source_tetgen_node_id"],"partition node ID")
  if partition not in ("skin","fat") or (partition,sid) in source_ids: raise ValueError("duplicate partition source node")
  source_ids.add((partition,sid));per[row].append(r)
 if any(not rows for rows in per): raise ValueError("node map omits joint TetGen nodes")
 incident=[[] for _ in range(n)]
 for tet,lab in zip(child["tets"],child["labels"]):
  for node in tet: incident[node].append(lab)
 lookup=make_lookup(parent,owner);mapped={l:set() for l in ("O","I","D")}
 output=[];shared=0;parent_rebindings=[];unresolved_parent_semantics=[]
 for node,rows in enumerate(per):
  if len(rows) not in (1,2): raise ValueError(f"node {node} has invalid partition map multiplicity")
  entries=[]
  for r in rows:
   l=r["registered_surface_layer"];sr=r["registered_surface_vertex_row"]
   if l=="":
    if sr not in ("","-1",str(INVALID)): raise ValueError("interior map claims a surface row")
   else:
    if l not in ("O","I","D"): raise ValueError("unknown registered surface role")
    srow=u(sr,f"{l} surface row")
    if srow>=len(surfaces[l]): raise ValueError("registered surface row out of domain")
    entries.append((l,srow))
  if len(rows)==1:
   r=rows[0]
   if (len(entries)>1 or (entries and entries[0][0]=="I") or
       r["registered_I_source_vertex_row"] not in ("","-1",str(INVALID)) or
       r["merge_peer_partition"] or r["merge_peer_tetgen_node_id"]):
    raise ValueError("non-I map claims interface ownership or a shared I row is missing its peer")
  else:
   if len(entries)!=2 or any(x[0]!="I" for x in entries): raise ValueError("only I rows may be shared")
   bypart={r["source_partition"]:r for r in rows}
   if set(bypart)!={"skin","fat"}: raise ValueError("shared I needs exactly one skin and one fat source")
   srset={x[1] for x in entries}
   if len(srset)!=1: raise ValueError("shared I maps to different registered surface rows")
   sr=next(iter(srset))
   if any(r["registered_I_source_vertex_row"]!=str(sr) for r in rows): raise ValueError("shared I source vertex row mismatch")
   skin,fat=bypart["skin"],bypart["fat"]
   sk=u(skin["source_tetgen_node_id"],"skin source ID");fa=u(fat["source_tetgen_node_id"],"fat source ID")
   if skin["merge_peer_partition"]!="fat" or fat["merge_peer_partition"]!="skin" or \
      u(skin["merge_peer_tetgen_node_id"],"skin peer")!=fa or u(fat["merge_peer_tetgen_node_id"],"fat peer")!=sk:
    raise ValueError("shared I map is not reciprocal")
   shared+=1
  if entries:
   if len(set(entries))!=1: raise ValueError("joint node maps to multiple surface identities")
   layer,sr=entries[0]
   if sr in mapped[layer]: raise ValueError(f"{layer} surface row maps more than once")
   mapped[layer].add(sr);role=layer;s=surfaces[layer][sr];point=child["points"][node].astype(np.float64)
   if not np.array_equal(s["point"].astype(np.float32),child["points"][node]): raise ValueError(f"{layer} volume point differs from exact F32 surface point")
   donor=s["donor"];pose=poses[donor];local=s["local"]
   if not matches_current_world_to_local(s["matrix"],pose) or np.linalg.norm(pose["current_origin"]+pose["current_R"]@local-point)>2e-8:
    raise ValueError(f"{layer} donor frame/local point mismatch")
   domain=s["source_domain"];sv=s["source_vertex"];sf=s["source_face"]
   kind="vertex" if sv!=INVALID else "face";sb=s["source_bary"];sres=s["source_residual"];fmethod=s["frame_method"]
   original_pr=registrations[(layer,sr)];pr=dict(original_pr);pt=pr["tet"];sem=parent_tuple(parent,pt)
   incident_set={tuple(x[:3]) for x in incident[node]}
   if sem not in incident_set:
    selected,witness=select_incident_parent_registration(point,layer,sem,incident_set,lookup,owner,maxdist,tol,sr,node)
    if witness is not None:
     unresolved_parent_semantics.append(witness)
     continue
    newpt=int(selected["tet"]);newsem=tuple(map(int,selected["tuple"]))
    newbeta=np.asarray(selected["beta"],dtype=np.float64);newres=np.asarray(selected["residual"],dtype=np.float64)
    base_check=newbeta@parent["points"][parent["tets"][newpt]]
    measured_distance=float(np.linalg.norm(newres))
    if (newsem not in incident_set or parent_tuple(parent,newpt)!=newsem or
        np.linalg.norm(base_check+newres-point)>2e-8 or abs(measured_distance-float(selected["distance"]))>2e-10):
     raise ValueError(f"{layer} source row {sr} selected incident parent fails exact tuple/reconstruction/distance checks")
    parent_rebindings.append({"role":layer,"source_row":int(sr),"joint_node_row":int(node),
      "child_incident_semantic_tuples":[list(map(int,k)) for k in sorted(incident_set)],
      "original_230":{"kind":str(original_pr["kind"]),"tet":int(original_pr["tet"]),"tuple":list(map(int,sem)),
       "beta":np.asarray(original_pr["beta"],dtype=np.float64).tolist(),"residual":np.asarray(original_pr["residual"],dtype=np.float64).tolist(),
       "distance":float(original_pr["distance"]),"ties":int(original_pr["ties"])},
      "current_217":{"kind":"inferred_current_parent_tet_same_material_donor_region","tet":newpt,"tuple":list(newsem),
       "beta":newbeta.tolist(),"residual":newres.tolist(),"distance":measured_distance,"ties":int(selected["ties"])},
      "selection_policy":"nearest bounded incident tuple; O material 15; D material 16; I prefers original material among bounded incident choices"})
    pr={"kind":"inferred_current_parent_tet_same_material_donor_region","tet":newpt,"beta":newbeta,
        "residual":newres,"distance":measured_distance,"ties":int(selected["ties"])}
    pt=newpt;sem=newsem
   if layer=="O" and (sem[0]!=15 or any(r["source_partition"]!="skin" for r in rows)):
    raise ValueError("O node parent/material or source partition is not skin 15/skin")
   if layer=="D" and (sem[0]!=16 or any(r["source_partition"]!="fat" for r in rows)):
    raise ValueError("D node parent/material or source partition is not fat 16/fat")
   base=pr["beta"]@parent["points"][parent["tets"][pt]]
   if np.linalg.norm(base+pr["residual"]-point)>2e-8: raise ValueError(f"{layer} current parent registration misses its point")
   parentkind=pr["kind"];pb=pr["beta"];pres=pr["residual"];pd=pr["distance"];ties=pr["ties"]
   semantic=sem
  else:
   role="interior";point=child["points"][node].astype(np.float64)
   incident_set={tuple(x[:3]) for x in incident[node]}
   if not incident_set:
    unresolved_parent_semantics.append({"role":"interior","joint_node_row":int(node),"incident_child_semantic_tuples":[],
     "failure":"interior node has no incident child cells","maximum_parent_distance_m":float(maxdist)})
    continue
   choices=[]
   for key in sorted(incident_set):
    try: choices.append(nearest_for_tuple(point,key,lookup,owner,maxdist,tol))
    except ValueError as exc:
     if str(exc).startswith("no current parent cells for semantic tuple"): continue
     raise
   bounded=[x for x in choices if math.isfinite(float(x["distance"])) and x["distance"]<=maxdist]
   if not bounded:
    ordered=sorted(choices,key=lambda x:(float(x["distance"]),tuple(x["tuple"]),int(x["tet"])))
    unresolved_parent_semantics.append({"role":"interior","joint_node_row":int(node),
     "incident_child_semantic_tuples":[list(map(int,k)) for k in sorted(incident_set)],
     "nearest_parent_candidates":[{"tuple":list(map(int,x["tuple"])),"tet":int(x["tet"]),"distance_m":float(x["distance"])} for x in ordered[:8]],
     "maximum_parent_distance_m":float(maxdist),
     "failure":"no incident semantic parent candidate within declared maximum distance" if choices else "no parent cells for any incident semantic tuple"})
    continue
   choices=bounded
   choices.sort(key=lambda x:(x["distance"],x["tuple"],x["tet"]))
   selected=choices[0];semantic=selected["tuple"];donor=semantic[1];pose=poses[donor]
   local=pose["current_world_to_local"]@(point-pose["current_origin"])
   if np.linalg.norm(pose["current_origin"]+pose["current_R"]@local-point)>1e-8: raise ValueError("interior donor local point does not reconstruct")
   domain=SOURCE_ABSENCE;kind="none";sv=sf=INVALID;sb=np.zeros(3);sres=np.zeros(3)
   fmethod=NO_SOURCE_METHOD;parentkind="inferred_current_parent_tet_same_material_donor_region"
   pt=selected["tet"];pb=selected["beta"];pres=selected["residual"];pd=selected["distance"];ties=selected["ties"]
  if not incident[node]: raise ValueError(f"node {node} has no incident child cells")
  mask=0
  for lab in incident[node]:
   if lab[0] not in MATERIAL_MASK: raise ValueError("node incident to unsupported material")
   mask|=MATERIAL_MASK[lab[0]]
  material,geodonor,region=semantic
  if parent_tuple(parent,pt)!=(material,geodonor,region): raise ValueError("selected parent does not match selected incident semantic tuple")
  if material not in MATERIAL_MASK or not(mask&MATERIAL_MASK[material]): raise ValueError("selected parent material absent from node incident cells")
  if pd>maxdist: raise ValueError("node parent registration exceeds declared maximum")
  binding=poses[donor]["binding"]
  out=dict(rows[0]);out.update({
   "role":role,"current_x_m":fs(point[0]),"current_y_m":fs(point[1]),"current_z_m":fs(point[2]),
   "donor_body":str(donor),"donor_binding":"-1" if binding==INVALID else str(binding),
   "donor_local_x_m":fs(local[0]),"donor_local_y_m":fs(local[1]),"donor_local_z_m":fs(local[2]),
   "source_domain":domain,"source_mapping_kind":kind,
   "source_vertex_row":"-1" if sv==INVALID else str(sv),"source_face_row":"-1" if sf==INVALID else str(sf),
   "source_bary0":fs(sb[0]),"source_bary1":fs(sb[1]),"source_bary2":fs(sb[2]),
   "source_residual_x_m":fs(sres[0]),"source_residual_y_m":fs(sres[1]),"source_residual_z_m":fs(sres[2]),
   "frame_owner_method":fmethod,"parent_registration_kind":parentkind,"parent_tet_row":str(pt),
   "parent_bary0":fs(pb[0]),"parent_bary1":fs(pb[1]),"parent_bary2":fs(pb[2]),"parent_bary3":fs(pb[3]),
   "parent_current_residual_x_m":fs(pres[0]),"parent_current_residual_y_m":fs(pres[1]),"parent_current_residual_z_m":fs(pres[2]),
   "parent_registration_distance_m":fs(pd),"parent_registration_tie_count":str(ties),
   "incident_material_mask":str(mask),"geometric_donor_body":str(geodonor),"anatomical_region_id":str(region)})
  for source_row in rows:
   emitted=dict(out)
   for column in BASE_HEADER: emitted[column]=source_row[column]
   output.append(emitted)
 validate_active_surface_coverage(mapped,surfaces,active_rows_by_role)
 if active_rows_by_role is not None and shared!=len(active_rows_by_role["I"]):
  raise ValueError(f"reciprocal current I interface count {shared} differs from exact active I rows {len(active_rows_by_role['I'])}")
 if unresolved_parent_semantics:
  raise ParentSemanticResolutionBatchError(unresolved_parent_semantics)
 unique={}
 for row in output: unique.setdefault(row["joint_node_id"],row)
 return output,{"volume_nodes":n,"map_rows":len(output),"shared_interface_nodes":shared,
  "registered_surface_rows":{l:len(surfaces[l]) for l in ("O","I","D")},
  "node_roles":{role:sum(x["role"]==role for x in unique.values()) for role in ("O","I","D","interior")},
  "parent_semantic_rebindings":parent_rebindings}

def preflight_current_217_inputs(args):
 if args.maximum_parent_distance_m<=0 or args.ambiguity_tolerance_m<0: raise ValueError("invalid registration bounds")
 scriptdir=Path(__file__).resolve().parent
 child=read_child(args.node.resolve(),args.ele.resolve(),args.labels.resolve())
 maprows=read_combined_node_map(args.node_map.resolve())
 parent,owner=read_parent_current(args.parent_base.resolve(),scriptdir)
 poses,bindings=read_poses(args.poses.resolve(),args.registered_dir.resolve()/"donor-frames.csv")
 pins={}
 for role,path,expected in (getattr(args,"current_typed_role",None) or []):
  if role in pins: raise ValueError(f"duplicate current typed role pin {role}")
  pins[role]=(Path(path),expected)
 role_auth=validate_current_typed_sidecar_bundle(args.registered_dir.resolve(),pins,args.maximum_parent_distance_m,
      args.current_230_source_proof,args.current_230_receipt_sha256,args.current_volume_state,args.parent_base,args.poses)
 hp=role_auth["source_gate"]["handoff_output_paths"]
 for arg,name in ((args.node_map,"skin-fat-combined-node-map.csv.gz"),(args.node,"skin-fat-combined.node"),(args.ele,"skin-fat-combined.ele")):
  if Path(arg).resolve()!=Path(hp[name]).resolve():
   raise ValueError(f"217 input {Path(arg).name} is not the exact authenticated 938 handoff output")
 active={r:set(role_auth["roles"][r]["active_source_vertex_rows"]) for r in ("O","I","D")}
 surfaces=read_surfaces(args.registered_dir.resolve(),current_typed=True,active_rows_by_role=active)
 registrations=read_parent_registrations(args.registered_dir.resolve(),surfaces,parent,args.maximum_parent_distance_m,
                                           current_typed=True,active_rows_by_role=active)
 result={"schema":"numi.h706.current-217-input-preflight.v1","status":"PASS_217_INPUT_READERS_AND_CURRENT_230_PROOF",
  "physical_admission":False,"native_steps":0,
  "capacities":capacity_profile_receipt(),
  "child":{"points":len(child["points"]),"tets":len(child["tets"]),"labels":len(child["labels"])},
  "base_node_map_rows":len(maprows),"parent":{"points":len(parent["points"]),"tets":len(parent["tets"])},
  "poses":{"all_bodies":len(poses),"registered_bindings":len(bindings)},
  "active_source_rows":{r:len(active[r]) for r in ("O","I","D")},
  "full_source_rows":{r:len(surfaces[r]) for r in ("O","I","D")},
  "parent_registrations":{r:sum(1 for (layer,_) in registrations if layer==r) for r in ("O","I","D")},
  "source_gate":role_auth.get("source_gate"),
  "input_pins":{k:{"path":str(Path(v).resolve()),"sha256":sha(Path(v).resolve())} for k,v in {
    "node_map":args.node_map,"node":args.node,"ele":args.ele,"labels":args.labels,"parent_base":args.parent_base,
    "poses":args.poses}.items()},
  "parent_full86_weights":role_auth["source_gate"].get("parent_full86_weights"),
  "frame_source_method":"native_all_bodies_f32_local_to_world_matrix_and_inverse",
  "sparse_O_source_row_553":role_auth["source_gate"]["source_553_control"]}
 if args.preflight_receipt:
  args.preflight_receipt.parent.mkdir(parents=True,exist_ok=True)
  args.preflight_receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
 return result

def emit(args):
 if args.output is None: raise ValueError('production emit requires --output')
 if args.maximum_parent_distance_m<=0 or args.ambiguity_tolerance_m<0: raise ValueError("invalid registration bounds")
 scriptdir=Path(__file__).resolve().parent
 child=read_child(args.node.resolve(),args.ele.resolve(),args.labels.resolve())
 maprows=read_combined_node_map(args.node_map.resolve())
 parent,owner=read_parent_current(args.parent_base.resolve(),scriptdir)
 poses,bindings=read_poses(args.poses.resolve(),args.registered_dir.resolve()/"donor-frames.csv")
 current_typed=bool(getattr(args,"current_typed_sidecars",False))
 role_authorization=None; pins={}
 if current_typed:
  pins={}
  for raw in (getattr(args,"current_typed_role",None) or []):
   role,path,expected=raw
   if role in pins: raise ValueError(f"duplicate current typed role pin {role}")
   pins[role]=(Path(path),expected)
  role_authorization=validate_current_typed_sidecar_bundle(args.registered_dir.resolve(),pins,args.maximum_parent_distance_m, getattr(args,"current_230_source_proof",None), getattr(args,"current_230_receipt_sha256",None), getattr(args,"current_volume_state",None), args.parent_base, args.poses)
  hp=role_authorization["source_gate"]["handoff_output_paths"]
  for input_path,name in ((args.node_map,"skin-fat-combined-node-map.csv.gz"),(args.node,"skin-fat-combined.node"),(args.ele,"skin-fat-combined.ele")):
   if Path(input_path).resolve()!=Path(hp[name]).resolve():
    raise ValueError(f"217 input {Path(input_path).name} is not the exact authenticated 938 handoff output")
 active_rows_by_role=({r:set(role_authorization["roles"][r]["active_source_vertex_rows"]) for r in ("O","I","D")}
                       if current_typed else None)
 surfaces=read_surfaces(args.registered_dir.resolve(),current_typed=current_typed,active_rows_by_role=active_rows_by_role)
 registrations=read_parent_registrations(args.registered_dir.resolve(),surfaces,parent,args.maximum_parent_distance_m,
                                           current_typed=current_typed,active_rows_by_role=active_rows_by_role)
 try:
  output,measured=build_rows(maprows,child,surfaces,registrations,parent,owner,poses,
                             args.maximum_parent_distance_m,args.ambiguity_tolerance_m,
                             active_rows_by_role=active_rows_by_role)
 except ParentSemanticResolutionBatchError as exc:
  diagnostic={"schema":"numi.h706.current-217-parent-semantic-resolution-failure.v1",
   "status":"NO_OUTPUT_UNRESOLVED_PARENT_SEMANTICS","physical_admission":False,"native_steps":0,
   "child":{"points":len(child["points"]),"tets":len(child["tets"])},
   "maximum_parent_distance_m":float(args.maximum_parent_distance_m),
   "unresolved_count":len(exc.witnesses),"unresolved_count_by_role":exc.counts,
   "source_sidecar_receipt":{"path":str((args.registered_dir.resolve()/"receipt.json")),
     "sha256":sha(args.registered_dir.resolve()/"receipt.json")},
   "original_230_parent_registration":{"path":str((args.registered_dir.resolve()/"node-parent-registration.csv")),
     "sha256":sha(args.registered_dir.resolve()/"node-parent-registration.csv")},
   "source_proof":{"path":str(Path(args.current_230_source_proof[0]).resolve()),
     "sha256":str(args.current_230_source_proof[1])},
   "operational_state":{"path":str(Path(args.current_volume_state[0]).resolve()),
     "sha256":str(args.current_volume_state[1])},
   "candidate_surfaces":{role:{"path":str(path.resolve()),"sha256":expected} for role,(path,expected) in pins.items()},
   "unresolved_witnesses":exc.witnesses}
  diag_path=args.output.parent/"parent-semantic-resolution-failure.json"
  diag_path.write_text(json.dumps(diagnostic,sort_keys=True,indent=2)+"\n",encoding="utf-8")
  raise ValueError(f"current 217 parent semantics unresolved for {len(exc.witnesses)} nodes; witnesses saved to {diag_path}") from exc
 parent_rebindings=measured.pop("parent_semantic_rebindings",[])
 if active_rows_by_role is not None:
  measured["face_active_source_rows"]={r:len(active_rows_by_role[r]) for r in ("O","I","D")}
  measured["inactive_source_provenance_rows_not_physical"]={r:len(surfaces[r])-len(active_rows_by_role[r]) for r in ("O","I","D")}
  retained_nodes=len({int(x["joint_node_id"]) for x in output})
  measured["all_child_volume_nodes_retained"]=retained_nodes==len(child["points"])
  if retained_nodes!=len(child["points"]):
   raise ValueError(f"output omitted child volume nodes: retained={retained_nodes} expected={len(child['points'])}")
  if len(child["points"])!=311768 or measured["shared_interface_nodes"]!=118197:
   raise ValueError(f"current 217 domain count mismatch: nodes={len(child['points'])} expected=311768; reciprocal_I={measured['shared_interface_nodes']} expected=118197")
 if len(output)!=len(maprows): raise ValueError("typed extension changed source-row count")
 args.output.parent.mkdir(parents=True,exist_ok=True)
 rebind_path=args.output.with_name("parent-semantic-rebindings.csv")
 rebind_fields=["role","source_row","joint_node_row","incident_child_semantic_tuples_json",
  "original_230_kind","original_230_tet","original_230_tuple_json","original_230_beta_json",
  "original_230_residual_json","original_230_distance_m","original_230_ties",
  "current_217_kind","current_217_tet","current_217_tuple_json","current_217_beta_json",
  "current_217_residual_json","current_217_distance_m","current_217_ties","selection_policy"]
 with rebind_path.open("w",encoding="utf-8",newline="") as stream:
  writer=csv.DictWriter(stream,fieldnames=rebind_fields,lineterminator="\n");writer.writeheader()
  for item in parent_rebindings:
   old=item["original_230"];new=item["current_217"]
   writer.writerow({"role":item["role"],"source_row":item["source_row"],"joint_node_row":item["joint_node_row"],
    "incident_child_semantic_tuples_json":json.dumps(item["child_incident_semantic_tuples"],separators=(",",":")),
    "original_230_kind":old["kind"],"original_230_tet":old["tet"],
    "original_230_tuple_json":json.dumps(old["tuple"],separators=(",",":")),
    "original_230_beta_json":json.dumps(old["beta"],separators=(",",":")),
    "original_230_residual_json":json.dumps(old["residual"],separators=(",",":")),
    "original_230_distance_m":old["distance"],"original_230_ties":old["ties"],
    "current_217_kind":new["kind"],"current_217_tet":new["tet"],
    "current_217_tuple_json":json.dumps(new["tuple"],separators=(",",":")),
    "current_217_beta_json":json.dumps(new["beta"],separators=(",",":")),
    "current_217_residual_json":json.dumps(new["residual"],separators=(",",":")),
    "current_217_distance_m":new["distance"],"current_217_ties":new["ties"],
    "selection_policy":item["selection_policy"]})
 with tempfile.NamedTemporaryFile("w",encoding="utf-8",newline="",dir=args.output.parent,delete=False) as stream:
  temp=Path(stream.name);writer=csv.DictWriter(stream,fieldnames=HEADER,lineterminator="\n")
  writer.writeheader();writer.writerows(output);stream.flush()
 temp.replace(args.output)
 input_paths={"node_map":args.node_map.resolve(),"tetgen_node":args.node.resolve(),
  "tetgen_ele":args.ele.resolve(),"child_labels":args.labels.resolve(),"parent_mass":args.parent_base.resolve(),
  "parent_nodes":Path(str(args.parent_base.resolve())+".nodes.csv"),
  "parent_source":Path(str(args.parent_base.resolve())+".source.csv"),
  "parent_tets":Path(str(args.parent_base.resolve())+".tets.csv"),
  "parent_labels":Path(str(args.parent_base.resolve())+".labels.csv"),
  "O_nodes":args.registered_dir.resolve()/"registered-O.csv",
  "I_nodes":args.registered_dir.resolve()/"registered-I.csv",
  "D_nodes":args.registered_dir.resolve()/"registered-D.csv",
  "node_parent_registration":args.registered_dir.resolve()/"node-parent-registration.csv",
  "donor_frames":args.registered_dir.resolve()/"donor-frames.csv","all_bodies":args.poses.resolve()}
 if role_authorization is not None:
  input_paths["current_typed_230_receipt"]=args.registered_dir.resolve()/"receipt.json"
  input_paths["current_230_source_proof"]=Path(args.current_230_source_proof[0]).resolve()
  input_paths["current_volume_state"]=Path(args.current_volume_state[0]).resolve()
  for role,(path,_) in pins.items(): input_paths[f"current_typed_{role}_asset"]=path.resolve()
 input_paths["original_230_node_parent_registration"]=args.registered_dir.resolve()/"node-parent-registration.csv"
 for layer in ("O","I","D"):
  p=args.registered_dir.resolve()/f"registered-{layer}-faces.csv"
  if p.exists(): input_paths[f"{layer}_faces"]=p
 receipt={"schema":"existing-combined-TetGen-node-map-registered-extension-v2-active-full-source-rows",
  "registrar_derivation":{"source_path":str(Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/downstream-003-current-typed-228/native-domain-217-source-005/register_combined_volume_nodes_217.py")),"source_sha256":"14452cc73ee20c50fad46ab369b65b799e80f5f00577cb6b21a8c51192ec676c","inactive_provenance_policy":"full typed point/face rows are retained; only face-active full source point rows receive current parent registrations or physical nodes"},
  "capacity_profile":capacity_profile_receipt(),
  "physical_admission":False,"source_ancestry":"exact registered O/I/D rows; explicit no-surface ancestry for volume interiors",
  "interior_parent_method":"inferred_current_parent_tet_same_material_donor_region",
  "parent_coordinates":"actual current-world parent nodes","maximum_parent_distance_m":args.maximum_parent_distance_m,
  "ambiguity_tolerance_m":args.ambiguity_tolerance_m,"all_bodies":len(poses),"registered_bindings":len(bindings),
  "current_typed_sidecars":role_authorization,
  "parent_semantic_rebinding":{"schema":"numi.h706.current-217-parent-semantic-rebinding.v1",
   "selection_policy":"nearest bounded incident child tuple; O material 15; D material 16; I prefers original material among bounded incident choices, then falls back to another incident I material",
   "rebound_count":len(parent_rebindings),
   "rebound_count_by_role":{r:sum(x["role"]==r for x in parent_rebindings) for r in ("O","I","D")},
   "audit_csv":{"path":str(rebind_path.resolve()),"sha256":sha(rebind_path)},
   "original_230_registration_preserved":True,
   "original_230_registration":{"path":str((args.registered_dir.resolve()/"node-parent-registration.csv")),
      "sha256":sha(args.registered_dir.resolve()/"node-parent-registration.csv")},
   "child_geometry_or_labels_changed":False,"physical_admission":False,"native_steps":0},
  "inputs":{k:{"path":str(v),"sha256":sha(v)} for k,v in input_paths.items()},
  "output":{"path":str(args.output.resolve()),"sha256":sha(args.output.resolve())},**measured}
 if args.receipt:
  args.receipt.parent.mkdir(parents=True,exist_ok=True)
  args.receipt.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n",encoding="utf-8")
 return receipt

def read_combined_node_map(path):
 opener=gzip.open if path.suffix==".gz" else open
 with opener(path,"rt",encoding="utf-8",newline="") as stream:
  r=csv.DictReader(stream)
  if r.fieldnames!=BASE_HEADER: raise ValueError(f"unexpected existing node map header {r.fieldnames}")
  rows=list(r)
 if not rows or len(rows)>2*MAX_NODES or any(None in x for x in rows): raise ValueError("existing node map empty/malformed/oversized")
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--node-map",type=Path,required=True);p.add_argument("--node",type=Path,required=True)
 p.add_argument("--ele",type=Path,required=True);p.add_argument("--labels",type=Path,required=True)
 p.add_argument("--parent-base",type=Path,required=True);p.add_argument("--registered-dir",type=Path,required=True)
 p.add_argument("--poses",type=Path,required=True);p.add_argument("--maximum-parent-distance-m",type=float,required=True)
 p.add_argument("--ambiguity-tolerance-m",type=float,default=1e-10);p.add_argument("--output",type=Path)
 p.add_argument("--receipt",type=Path)
 p.add_argument("--current-typed-sidecars",action="store_true",
                 help="explicitly consume authenticated current O/I/D 230 rows; preserves historical support as provenance")
 p.add_argument("--current-typed-role",nargs=3,action="append",metavar=("ROLE","NPZ","SHA256"),
                 help="repeat once for O, I, and D when --current-typed-sidecars is enabled")
 p.add_argument("--current-230-source-proof",nargs=2,metavar=("PATH","SHA256"),
                 help="explicit pinned 808 proof used by the current 230 sidecar receipt")
 p.add_argument("--current-230-receipt-sha256",
                 help="exact SHA-256 of registered-dir/receipt.json from the current 230 export")
 p.add_argument("--current-volume-state",nargs=2,metavar=("PATH","SHA256"),
                 help="exact 818c operational state bound by the current 808 source proof")
 p.add_argument("--preflight-only",action="store_true",
                 help="read and authenticate all current 217 inputs without constructing the combined map")
 p.add_argument("--preflight-receipt",type=Path,
                 help="write a compact source-only 217 input preflight receipt")
 p.add_argument("--capacity-profile",choices=("legacy","full-reference-500k-nodes-2m-cells"),default="legacy",
                 help="explicit child mesh capacity; default preserves legacy 300k/200k limits")
 a=p.parse_args()
 configure_capacity_profile(a.capacity_profile)
 if a.current_typed_sidecars and not a.current_typed_role: p.error("--current-typed-sidecars requires explicit --current-typed-role O/I/D pins")
 if not a.current_typed_sidecars and a.current_typed_role: p.error("--current-typed-role requires --current-typed-sidecars")
 if a.current_typed_sidecars and (not a.current_230_source_proof or not a.current_230_receipt_sha256 or not a.current_volume_state):
  p.error("current typed sidecars require explicit 230 source-proof, sidecar receipt, and operational state pins")
 if not a.current_typed_sidecars and (a.current_230_source_proof or a.current_230_receipt_sha256 or a.current_volume_state):
  p.error("current 230 proof pins require --current-typed-sidecars")
 if a.preflight_only and a.output is None: pass
 elif not a.preflight_only and a.output is None: p.error("--output is required outside preflight mode")
 if a.preflight_only and a.receipt is None and a.preflight_receipt is None:
  p.error("--preflight-only requires --preflight-receipt")
 try: result=preflight_current_217_inputs(a) if a.preflight_only else emit(a)
 except (OSError,ValueError,KeyError,csv.Error,json.JSONDecodeError) as e:
  print(f"FAIL combined registered node-map preparation: {e}",file=sys.stderr);return 2
 print(json.dumps(result,sort_keys=True,separators=(",",":")));return 0

if __name__=="__main__": raise SystemExit(main())
