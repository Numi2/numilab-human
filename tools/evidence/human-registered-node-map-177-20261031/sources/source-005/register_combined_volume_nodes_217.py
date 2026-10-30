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
 for i,row in enumerate(raw):
  if u(row.get("body_index"),f"all_bodies body {i}")!=i: raise ValueError("all_bodies is not exact body-index order")
  q={}
  for phase in ("current","rest"):
   x=row.get(phase,{})
   origin=np.asarray(x.get("position_world_m"),dtype=np.float64)
   quat=np.asarray(x.get("orientation_world_xyzw"),dtype=np.float64)
   if origin.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(origin).all() or not np.isfinite(quat).all():
    raise ValueError(f"all_bodies {phase} pose {i} invalid")
   q[phase+"_origin"]=origin;q[phase+"_R"]=Rotation.from_quat(quat).as_matrix()
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

CURRENT_TYPED_SOURCE_DOMAIN = "derived_registered_skin_authoring"
CURRENT_FACE_FIELDS = ("face_source_layer", "face_source_body", "face_source_bone",
                       "face_source_face_id", "face_source_original_id",
                       "face_source_identity0", "face_source_identity1")


def validate_current_typed_sidecar_bundle(directory:Path, role_pins, maximum_parent_distance:float):
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
    gate = receipt.get("exactGeometryGateReceipt", {})
    if (gate.get("status") != "PASS_WHOLE_CURRENT_OID_VS_ALL185_F32_F64" or
        int(gate.get("all185Pairs", -1)) != 0 or
        any(int(gate.get("selfPairs", {}).get(role, -1)) != 0 for role in ("O", "I", "D")) or
        any(int(gate.get("crossPairs", {}).get(pair, -1)) != 0 for pair in ("O-I", "O-D", "I-D"))):
        raise ValueError("current typed sidecar receipt lacks the pinned whole-surface geometry gate")
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
                        "vertex_rows": len(points), "face_rows": len(faces)}
    return {"schema": "numi.current-typed-228-sidecar-authorization.v1",
            "roles": result, "receipt_path": str(receipt_path), "receipt_sha256": sha(receipt_path),
            "historical_support_maximum_distance_m": float(maximum_parent_distance),
            "historical_registration_is_provenance": True,
            "current_operational_map": "created later as exact one-hot child self-tets by 433 opt-in",
            "source330_to_original_source_bound_changed": False,
            "conservative_mass_transport_bound_changed": False,
            "physical_admission": False}


def read_surfaces(directory:Path,current_typed=False):
 out={}
 for layer in ("O","I","D"):
  rows=csvread(directory/f"registered-{layer}.csv")
  required={"vertex","donor","wx","wy","wz","lx","ly","lz","m00","m01","m02","m10","m11","m12","m20","m21","m22","source_domain","frame_owner_method"}
  if current_typed:
   required|={"source_mapping_kind","source_vertex_row","source_face_row","source_bary0","source_bary1","source_bary2","source_residual_x_m","source_residual_y_m","source_residual_z_m"}
  elif layer=="O": required|={"source_vertex_row","source_face_row","source_bary0","source_bary1","source_bary2","source_residual_x_m","source_residual_y_m","source_residual_z_m"}
  else: required|={"d37_parent_node_row","d37_parent_surface_face_row","beta0","beta1","beta2","source_residual_x_m","source_residual_y_m","source_residual_z_m"}
  if not rows or any(not required.issubset(r) for r in rows): raise ValueError(f"registered {layer} rows missing required existing columns")
  parsed=[]
  for i,r in enumerate(rows):
   if u(r["vertex"],f"{layer} surface vertex")!=i: raise ValueError(f"{layer} surface vertex order is not contiguous")
   donor=u(r["donor"],f"{layer} donor")
   point=np.asarray([f(r["w"+a],f"{layer} point") for a in "xyz"])
   local=np.asarray([f(r["l"+a],f"{layer} local") for a in "xyz"])
   matrix=np.asarray([f(r[f"m{x}{y}"],f"{layer} frame") for x in range(3) for y in range(3)]).reshape(3,3)
   if current_typed:
    sv=ou(r["source_vertex_row"],f"{layer} current source vertex");sf=ou(r["source_face_row"],f"{layer} current source face")
    bary=np.asarray([f(r[f"source_bary{k}"],f"{layer} source bary") for k in range(3)])
    if (r["source_domain"]!=CURRENT_TYPED_SOURCE_DOMAIN or r["source_mapping_kind"]!="vertex" or
        sv!=i or sf!=INVALID or np.any(bary!=0.0)):
     raise ValueError(f"{layer} current typed sidecar does not identify exact same-row candidate vertex {i}")
   elif layer=="O":
    sv=ou(r["source_vertex_row"],"O source vertex");sf=ou(r["source_face_row"],"O source face")
    bary=np.asarray([f(r[f"source_bary{k}"],"O source bary") for k in range(3)])
   else:
    sv=ou(r["d37_parent_node_row"],f"{layer} source vertex")
    sf=ou(r["d37_parent_surface_face_row"],f"{layer} source face")
    bary=np.asarray([f(r[f"beta{k}"],f"{layer} source bary") for k in range(3)])
   residual=np.asarray([f(r[f"source_residual_{a}_m"],f"{layer} source residual") for a in "xyz"])
   if (sv==INVALID)==(sf==INVALID): raise ValueError(f"{layer} vertex {i} needs exactly one direct or face identity")
   if sf!=INVALID and (np.min(bary)<-1e-8 or abs(float(bary.sum())-1)>1e-8): raise ValueError(f"{layer} source bary invalid")
   if current_typed and np.any(residual!=0.0): raise ValueError(f"{layer} current typed direct vertex identity must have zero source residual")
   parsed.append({"row":r,"donor":donor,"point":point,"local":local,"matrix":matrix,
      "source_vertex":sv,"source_face":sf,"source_bary":bary,"source_residual":residual,
      "source_domain":r["source_domain"],"frame_method":r["frame_owner_method"]})
  out[layer]=parsed
 return out

def read_parent_registrations(directory:Path,surfaces,parent,maxdist,current_typed=False):
 path=directory/"node-parent-registration.csv"
 if current_typed:
  with path.open("r",encoding="utf-8",newline="") as stream:
   reader=csv.DictReader(stream)
   if reader.fieldnames is None or not set(REG_HEADER).issubset(reader.fieldnames): raise ValueError("current typed 230 registration sidecar lacks existing 217 registration columns")
   rows=list(reader)
  if any(None in r for r in rows): raise ValueError("malformed current typed 230 parent registration rows")
 else: rows=csvread(path,REG_HEADER)
 out={}
 for line,r in enumerate(rows,2):
  layer=r["layer"];row=u(r["surface_vertex_row"],f"registration row {line}")
  if layer not in surfaces or row>=len(surfaces[layer]) or (layer,row) in out: raise ValueError("invalid/duplicate surface parent registration")
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
 expected={(l,i) for l in ("O","I","D") for i in range(len(surfaces[l]))}
 if set(out)!=expected: raise ValueError("registered current parent rows omit an O/I/D point")
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

def build_rows(maprows,child,surfaces,registrations,parent,owner,poses,maxdist,tol):
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
 output=[];shared=0
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
   if not np.array_equal(s["matrix"],pose["current_R"].T) or np.linalg.norm(pose["current_origin"]+pose["current_R"]@local-point)>2e-8:
    raise ValueError(f"{layer} donor frame/local point mismatch")
   domain=s["source_domain"];sv=s["source_vertex"];sf=s["source_face"]
   kind="vertex" if sv!=INVALID else "face";sb=s["source_bary"];sres=s["source_residual"];fmethod=s["frame_method"]
   pr=registrations[(layer,sr)];pt=pr["tet"];sem=parent_tuple(parent,pt)
   incident_set={tuple(x[:3]) for x in incident[node]}
   if sem not in incident_set: raise ValueError(f"{layer} registered parent semantic tuple is not incident")
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
   if not incident_set: raise ValueError(f"interior node {node} has no incident child cells")
   choices=[]
   for key in sorted(incident_set):
    try: choices.append(nearest_for_tuple(point,key,lookup,owner,maxdist,tol))
    except ValueError as exc:
     if str(exc).startswith("no current parent cells for semantic tuple"): continue
     raise
   bounded=[x for x in choices if x["distance"]<=maxdist]
   if not bounded:
    nearest=min((x["distance"] for x in choices),default=math.inf)
    raise ValueError(f"interior node {node} has no bounded parent for any incident semantic tuple; nearest={nearest:.17g} m")
   choices=bounded
   choices.sort(key=lambda x:(x["distance"],x["tuple"],x["tet"]))
   selected=choices[0];semantic=selected["tuple"];donor=semantic[1];pose=poses[donor]
   local=pose["current_R"].T@(point-pose["current_origin"])
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
 for layer in ("O","I","D"):
  if mapped[layer]!=set(range(len(surfaces[layer]))): raise ValueError(f"{layer} map does not bijectively cover its registered surface rows")
 unique={}
 for row in output: unique.setdefault(row["joint_node_id"],row)
 return output,{"volume_nodes":n,"map_rows":len(output),"shared_interface_nodes":shared,
  "registered_surface_rows":{l:len(surfaces[l]) for l in ("O","I","D")},
  "node_roles":{role:sum(x["role"]==role for x in unique.values()) for role in ("O","I","D","interior")}}

def emit(args):
 if args.maximum_parent_distance_m<=0 or args.ambiguity_tolerance_m<0: raise ValueError("invalid registration bounds")
 scriptdir=Path(__file__).resolve().parent
 child=read_child(args.node.resolve(),args.ele.resolve(),args.labels.resolve())
 maprows=read_combined_node_map(args.node_map.resolve())
 parent,owner=read_parent_current(args.parent_base.resolve(),scriptdir)
 poses,bindings=read_poses(args.poses.resolve(),args.registered_dir.resolve()/"donor-frames.csv")
 current_typed=bool(getattr(args,"current_typed_sidecars",False))
 role_authorization=None
 if current_typed:
  pins={}
  for raw in (getattr(args,"current_typed_role",None) or []):
   role,path,expected=raw
   if role in pins: raise ValueError(f"duplicate current typed role pin {role}")
   pins[role]=(Path(path),expected)
  role_authorization=validate_current_typed_sidecar_bundle(args.registered_dir.resolve(),pins,args.maximum_parent_distance_m)
 surfaces=read_surfaces(args.registered_dir.resolve(),current_typed=current_typed)
 registrations=read_parent_registrations(args.registered_dir.resolve(),surfaces,parent,args.maximum_parent_distance_m,current_typed=current_typed)
 output,measured=build_rows(maprows,child,surfaces,registrations,parent,owner,poses,
                            args.maximum_parent_distance_m,args.ambiguity_tolerance_m)
 if len(output)!=len(maprows): raise ValueError("typed extension changed source-row count")
 args.output.parent.mkdir(parents=True,exist_ok=True)
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
  input_paths["current_typed_228_receipt"]=args.registered_dir.resolve()/"receipt.json"
  for role,(path,_) in pins.items(): input_paths[f"current_typed_{role}_asset"]=path.resolve()
 for layer in ("O","I","D"):
  p=args.registered_dir.resolve()/f"registered-{layer}-faces.csv"
  if p.exists(): input_paths[f"{layer}_faces"]=p
 receipt={"schema":"existing-combined-TetGen-node-map-registered-extension",
  "capacity_profile":capacity_profile_receipt(),
  "physical_admission":False,"source_ancestry":"exact registered O/I/D rows; explicit no-surface ancestry for volume interiors",
  "interior_parent_method":"inferred_current_parent_tet_same_material_donor_region",
  "parent_coordinates":"actual current-world parent nodes","maximum_parent_distance_m":args.maximum_parent_distance_m,
  "ambiguity_tolerance_m":args.ambiguity_tolerance_m,"all_bodies":len(poses),"registered_bindings":len(bindings),
  "current_typed_sidecars":role_authorization,
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
 p.add_argument("--ambiguity-tolerance-m",type=float,default=1e-10);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--receipt",type=Path)
 p.add_argument("--current-typed-sidecars",action="store_true",
                 help="explicitly consume authenticated current O/I/D 230 rows; preserves historical support as provenance")
 p.add_argument("--current-typed-role",nargs=3,action="append",metavar=("ROLE","NPZ","SHA256"),
                 help="repeat once for O, I, and D when --current-typed-sidecars is enabled")
 p.add_argument("--capacity-profile",choices=("legacy","full-reference-500k-nodes-2m-cells"),default="legacy",
                 help="explicit child mesh capacity; default preserves legacy 300k/200k limits")
 a=p.parse_args()
 configure_capacity_profile(a.capacity_profile)
 if a.current_typed_sidecars and not a.current_typed_role: p.error("--current-typed-sidecars requires explicit --current-typed-role O/I/D pins")
 if not a.current_typed_sidecars and a.current_typed_role: p.error("--current-typed-role requires --current-typed-sidecars")
 try: result=emit(a)
 except (OSError,ValueError,KeyError,csv.Error,json.JSONDecodeError) as e:
  print(f"FAIL combined registered node-map preparation: {e}",file=sys.stderr);return 2
 print(json.dumps(result,sort_keys=True,separators=(",",":")));return 0

if __name__=="__main__": raise SystemExit(main())
