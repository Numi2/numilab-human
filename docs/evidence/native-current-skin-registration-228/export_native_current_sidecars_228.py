#!/usr/bin/env python3
"""Emit current-world O/I/D registration sidecars from run006 and capture016.

Preserves canonical topology, source ancestry and NHSKIN payload provenance.
This is an initialization/reference-registration export, not geometry or
physical admission.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, math, re, shutil, sys
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
ROOT=P/"volume-worker/registered-surface-sidecar-222"
BASE=ROOT/"attempt-016-kindaware-registration"
OUT=ROOT/"attempt-018-native-current-reference"
CAP=P/"native-parent-label-capture-016/compiled-parent-cell-mass.csv"
RUN=P/"P/native-kind-aware-source-preflight-226/run-006-native-emission"
POSES=RUN/"registered-skin-poses-100.json"
NATIVE_O=RUN/"registered-skin-100-native.csv.candidate033.csv"
OWNER=Path("/Users/numi/Work/numi-human-connected-surface-069/publication-071/evidence/exact-transfer-075/run-001/source-preflight")
KIND0_NODE_MAP=BASE/"d37-parent-node-to-capture016.csv"
KIND0_ROW=16264                 # compact row; canonical/O033 row is 16398
KIND0_CANONICAL=16398
KIND0_COARSE_SOURCE=1233
KIND0_SUPPORT_DETAIL=24939
SIX_ROWS=[13498,13499,16946,17203,17204,17223]
INFERRED_KIND="inferred_current_parent_tet_same_material_donor_region"
PARENT_MAX_M=0.02
TIE_TOL_M=1.0e-8
sys.path.insert(0,str(P/"volume-worker/native-domain-217"))
import export_child_semantic_labels_217 as native


def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read_csv(path:Path):
 with path.open("r",encoding="utf-8",newline="") as f:
  r=csv.DictReader(f); rows=list(r)
  if not r.fieldnames or any(None in x for x in rows):raise ValueError(f"malformed CSV {path}")
 return list(r.fieldnames),rows
def read_gz_csv(path:Path):
 with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
  r=csv.DictReader(f); rows=list(r)
  if not r.fieldnames or any(None in x for x in rows):raise ValueError(f"malformed gzip CSV {path}")
 return list(r.fieldnames),rows
def write_csv(path:Path,header,rows):
 with path.open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=header,extrasaction="raise",lineterminator="\n")
  w.writeheader();w.writerows(rows)
def f64(x):return format(float(x),".17g")
def csv_num(row,key):return float(row[key])
def source_vector(row,keys):return np.asarray([float(row[k]) for k in keys],dtype=np.float64)
def exact_nearest(parent,index,original_rows,point):
 center,seed=index["tree"].query(point,k=1)
 if not math.isfinite(float(center)):raise ValueError("empty eligible parent tet set")
 _,_,seed_dist=native.closest_point_on_parent_tet(parent,index,int(seed),point)
 radius=seed_dist+index["max_radius"]+TIE_TOL_M+native.QUERY_PAD_M
 candidates=index["tree"].query_ball_point(point,radius)
 measured=[]
 for local in candidates:
  reg,bary,d=native.closest_point_on_parent_tet(parent,index,int(local),point)
  measured.append((float(d),int(local),int(original_rows[int(local)]),reg,bary))
 if not measured:raise ValueError("nearest eligible parent query returned no candidate")
 best=min(x[0] for x in measured)
 ties=[x for x in measured if abs(x[0]-best)<=TIE_TOL_M]
 # True distance is the primary key. Tet row only resolves numerically
 # indistinguishable equal-distance boundary ties.
 ties.sort(key=lambda x:(x[0],x[2]))
 return {"distance_m":best,"selected_local_row":ties[0][1],"selected_parent_tet_row":ties[0][2],
         "registered_point":np.asarray(ties[0][3]),"bary":np.asarray(ties[0][4]),
         "tie_count":len(ties)-1,
         "ties":[{"parent_tet_row":x[2],"distance_m":x[0],
                  "material_index":int(parent["materials"][x[1]]),
                  "geometric_donor_body":int(parent["donors"][x[1]]),
                  "anatomical_region_id":int(parent["regions"][x[1]]),
                  "stable_source_face_id":int(parent["faces"][x[1]])} for x in ties]}

def main():
 if OUT.exists():raise ValueError(f"refuse existing output directory {OUT}")
 base_receipt=json.loads((BASE/"receipt.json").read_text())
 base_receipt_sha=sha(BASE/"receipt.json")
 input_names=["registered-O.csv","registered-O-faces.csv","registered-I.csv","registered-I-faces.csv",
              "registered-D.csv","registered-D-faces.csv","node-parent-registration.csv",
              "donor-frames.csv","accepted-transfer-reference-support.csv",
              "O-compact-node-map.csv","O-compact-face-map.csv","O-compact-face-ancestry.csv",
              "O-face-ancestry.csv","accepted-transfer-reference-support.csv"]
 for name in input_names:
  if not (BASE/name).is_file():raise ValueError(f"missing pinned attempt016 input {name}")
 shutil.copytree(BASE,OUT)
 # Read and pin canonical O/native observations before any writes.
 o_header,o_rows=read_csv(BASE/"registered-O.csv")
 o_by={int(r["vertex"]):r for r in o_rows}
 if len(o_rows)!=32002 or set(o_by)!=set(range(32002)):raise ValueError("attempt016 O compact rows are incomplete")
 native_o_header,native_o_rows=read_csv(NATIVE_O)
 native_o={int(r["vertex"]):r for r in native_o_rows}
 if len(native_o_rows)!=32002 or set(native_o)!=set(range(32002)):raise ValueError("run006 O observation row domain differs")
 if native_o_header!=["vertex","donor","wx","wy","wz","nx","ny","nz","lx","ly","lz",
                     "m00","m01","m02","m10","m11","m12","m20","m21","m22"]:
  raise ValueError("run006 native O observation schema differs")
 map_header,map_rows=read_csv(BASE/"O-compact-node-map.csv")
 if sha(BASE/"O-compact-node-map.csv")!="5ed0c63d4a35a023bd91d0cbec1a933c8ad1f7b5fc1d32148cd8568ac77db8f4":
  raise ValueError("compact O row-map hash differs from the pinned attempt006 domain")
 if len(map_rows)!=32002:raise ValueError("compact O map row count differs")
 for i,(r,m,n) in enumerate(zip(o_rows,map_rows,native_o_rows)):
  if int(r["vertex"])!=i or int(m["compact_vertex_row"])!=i or int(n["vertex"])!=i:
   raise ValueError(f"O compact row identity mismatch at {i}")
  for k in ("canonical_candidate_vertex_row","O033_source_vertex_row"):
   if int(r[k])!=int(m[k]):raise ValueError(f"O source row-map mismatch at {i} field {k}")
  if int(r["donor"])!=int(n["donor"]):raise ValueError(f"native O owner differs at row {i}")
 # Pin exact run006 SE(3); retain authoring surfaces and payload separately.
 stdout=(RUN/"stdout.txt").read_text()
 line=next((x for x in stdout.splitlines() if "candidate033_native_source_frame_measurement" in x),None)
 if line is None:raise ValueError("run006 common placement record is missing")
 rm=re.search(r"common_placement_rotation=\[([^]]+)\]",line)
 tm=re.search(r"common_placement_translation_m=\[([^]]+)\]",line)
 if not rm or not tm:raise ValueError("run006 common SE(3) is incomplete")
 common_R=np.asarray([float(x) for x in rm.group(1).split(",")],dtype=np.float64).reshape(3,3)
 common_t=np.asarray([float(x) for x in tm.group(1).split(",")],dtype=np.float64)
 if np.linalg.det(common_R)<0.999999 or np.linalg.norm(common_R.T@common_R-np.eye(3),ord=np.inf)>1e-12:
  raise ValueError("run006 common placement is not a proper rigid rotation")
 # Authoritative F64 current/rest donor frames from the same run006 native poses.
 pose_doc=json.loads(POSES.read_text())
 pose_by={int(b["body_index"]):b for b in pose_doc["bodies"]}
 old_frame_header,old_frame_rows=read_csv(BASE/"donor-frames.csv")
 frames=[];frame_by={}
 for old in old_frame_rows:
  donor=int(old["donor"])
  if donor not in pose_by:raise ValueError(f"native pose omits donor {donor}")
  new={"binding":old["binding"],"donor":str(donor)}
  for state in ("current","rest"):
   pose=pose_by[donor][state]
   origin=np.asarray(pose["position_world_m"],dtype=np.float64)
   rotation=Rotation.from_quat(np.asarray(pose["orientation_world_xyzw"],dtype=np.float64)).as_matrix()
   for a,v in zip("xyz",origin):new[f"{state}_{a}"]=f64(v)
   for i in range(3):
    for j in range(3):new[f"{state}_R{i}{j}"]=f64(rotation[i,j])
  frames.append(new)
  curR=np.asarray([[float(new[f"current_R{i}{j}"]) for j in range(3)] for i in range(3)])
  frame_by[donor]={"origin":np.asarray([float(new[f"current_{a}"]) for a in "xyz"]),
                   "local_matrix":curR.T}
 if len(frames)!=86 or len(frame_by)!=86:raise ValueError("run006 native donor-frame domain is incomplete")
 frame_header=old_frame_header
 write_csv(OUT/"donor-frames.csv",frame_header,frames)
 # Load the actual current parent cells from capture016, never its rest coordinates.
 parent=native.read_parent(CAP)
 _,parent_nodes=read_csv(Path(str(CAP)+".nodes.csv"))
 parent_current=np.asarray([[float(r[k]) for k in ("current_x_m","current_y_m","current_z_m")] for r in parent_nodes],dtype=np.float64)
 if len(parent_current)!=len(parent["points"]):raise ValueError("capture016 current node table differs from parent cells")
 parent["points"]=parent_current
 # Native-current O rows: use actual emitted Float32 points/normals; derive F64
 # local/frame rows from run006 donor poses to satisfy the strict sidecar map.
 o_out_header=list(o_header)
 for key in ("source_vertex_row","source_face_row","source_bary0","source_bary1","source_bary2",
             "source_residual_x_m","source_residual_y_m","source_residual_z_m"):
  if key not in o_out_header:o_out_header.append(key)
 o_out=[];native_observation_roundoff=[]
 for i,old in enumerate(o_rows):
  n=native_o[i];r=dict(old);d=int(r["donor"]);f=frame_by[d]
  point=np.asarray([float(n[k]) for k in ("wx","wy","wz")],dtype=np.float64)
  normal=np.asarray([float(n[k]) for k in ("nx","ny","nz")],dtype=np.float64)
  if abs(float(np.linalg.norm(normal))-1.)>1e-5:raise ValueError(f"run006 O normal is not unit at {i}")
  native_observation_roundoff.append(float(np.linalg.norm(point-(common_R@np.asarray([float(old[k]) for k in ("wx","wy","wz")])+common_t))))
  for k,v in zip(("wx","wy","wz"),point):r[k]=f64(v)
  for k,v in zip(("nx","ny","nz"),normal):r[k]=f64(v)
  M=f["local_matrix"]
  local=M@(point-f["origin"])
  for a in range(3):
   for b in range(3):r[f"m{a}{b}"]=f64(M[a,b])
  for k,v in zip(("lx","ly","lz"),local):r[k]=f64(v)
  # Keep the source identity in its existing D37 fields; expose the established
  # normalized aliases consumed by the parent registration reader as well.
  src_v=int(r["d37_source_vertex_row"]);src_f=int(r["d37_source_face_row"])
  sr_old=np.asarray([float(r.get(f"residual_{a}","0")) for a in "xyz"],dtype=np.float64)
  sr=common_R@sr_old
  r["source_vertex_row"]=str(src_v);r["source_face_row"]=str(src_f)
  for k,beta in zip(("source_bary0","source_bary1","source_bary2"),("beta0","beta1","beta2")):r[k]=r[beta]
  for a,v in zip("xyz",sr):
   r[f"residual_{a}"]=f64(v);r[f"source_residual_{a}_m"]=f64(v)
  o_out.append(r)
 write_csv(OUT/"registered-O.csv",o_out_header,o_out)
 # Transform I/D coordinates into the same native current frame, round world
 # coordinates once to F32, and derive their local coordinates from run006 poses.
 id_out={};id_headers={};id_roundoff={}
 for layer in ("I","D"):
  header,rows=read_csv(BASE/f"registered-{layer}.csv")
  if len(rows)!=15022:raise ValueError(f"{layer} row count differs")
  outrows=[];roundoff=[]
  for i,old in enumerate(rows):
   r=dict(old);d=int(r["donor"]);f=frame_by[d]
   author=np.asarray([float(r[k]) for k in ("wx","wy","wz")],dtype=np.float64)
   point=(common_R@author+common_t).astype(np.float32).astype(np.float64)
   normal=np.asarray([float(r[k]) for k in ("nx","ny","nz")],dtype=np.float64)
   normal=common_R@normal;normal/=np.linalg.norm(normal);normal=normal.astype(np.float32).astype(np.float64)
   # The residual is a world vector attached to the D37 source embedding.
   sr=common_R@np.asarray([float(r[f"source_residual_{a}_m"]) for a in "xyz"],dtype=np.float64)
   local=f["local_matrix"]@(point-f["origin"])
   for k,v in zip(("wx","wy","wz"),point):r[k]=f64(v)
   for k,v in zip(("nx","ny","nz"),normal):r[k]=f64(v)
   for a in range(3):
    for b in range(3):r[f"m{a}{b}"]=f64(f["local_matrix"][a,b])
   for k,v in zip(("lx","ly","lz"),local):r[k]=f64(v)
   for a,v in zip("xyz",sr):r[f"source_residual_{a}_m"]=f64(v)
   roundoff.append(float(np.linalg.norm(point-(common_R@author+common_t))))
   outrows.append(r)
  write_csv(OUT/f"registered-{layer}.csv",header,outrows)
  id_out[layer]=outrows;id_headers[layer]=header;id_roundoff[layer]=roundoff
 # Six explicit O mechanical registrations: actual F32 target, material 15, the
 # exact geometric donor and original anatomical region. No source ancestry is
 # inferred from the selected current parent tet.
 support_header,support_rows=read_csv(BASE/"accepted-transfer-reference-support.csv")
 support={int(r["surface_vertex_row"]):r for r in support_rows}
 six_selection={}
 masks={}
 for row in SIX_ROWS:
  surf=o_out[row];sup=support[row];donor=int(surf["donor"])
  region=int(sup["selected_anatomical_region_id"])
  if int(sup["selected_geometric_donor_body"])!=donor or region!=donor:
   raise ValueError(f"six-row source semantics disagree at O row {row}: donor={donor}, region={region}")
  key=(15,donor,region)
  if key not in masks:
   mask=(parent["materials"]==15)&(parent["donors"]==donor)&(parent["regions"]==region)
   original=np.flatnonzero(mask)
   if not len(original):raise ValueError(f"no current parent cells for strict donor/region {key}")
   sub={**parent,"tets":parent["tets"][original],"materials":parent["materials"][original],
        "donors":parent["donors"][original],"regions":parent["regions"][original],"faces":parent["faces"][original]}
   masks[key]=(sub,native.build_index(sub),original)
  sub,index,original=masks[key]
  target=np.asarray([float(surf[k]) for k in ("wx","wy","wz")],dtype=np.float64)
  nearest=exact_nearest(sub,index,original,target)
  tet=nearest["selected_parent_tet_row"]
  if int(parent["materials"][tet])!=15 or int(parent["donors"][tet])!=donor or int(parent["regions"][tet])!=region:
   raise ValueError(f"strict current parent selection changed semantic tuple at O row {row}")
  if nearest["distance_m"]>PARENT_MAX_M:raise ValueError(f"inferred O row {row} exceeds existing parent bound")
  six_selection[row]={"parent_tet_row":tet,"bary":nearest["bary"].tolist(),
     "distance_m":nearest["distance_m"],"tie_count":nearest["tie_count"],
     "geometric_registration_stable_source_face_id":int(parent["faces"][tet]),
     "material_index":15,"geometric_donor_body":donor,"anatomical_region_id":region,
     "ties":nearest["ties"]}
 # kind0 direct coarse representative registration uses authoritative P1 support
 # stable-face ID only as a deterministic local tie-break, not as ancestry.
 maprows={int(r["d37_source_parent_node_row"]):int(r["capture016_parent_node_row"]) for r in read_csv(KIND0_NODE_MAP)[1]}
 if maprows.get(KIND0_COARSE_SOURCE)!=KIND0_COARSE_SOURCE:raise ValueError("kind0 source coarse row does not map to capture016 node1233")
 transfer_path=OWNER/"accepted-transfer-rows.csv.gz"
 transfer_header,transfer_rows=read_gz_csv(transfer_path)
 transfer={int(r["detail_row"]):r for r in transfer_rows}
 if KIND0_SUPPORT_DETAIL not in transfer or transfer[KIND0_SUPPORT_DETAIL]["active"]!="1":raise ValueError("kind0 P1 support row is not active")
 support_face=int(transfer[KIND0_SUPPORT_DETAIL]["selected_face_stable_id"])
 o_kind0=o_out[KIND0_ROW]
 if int(o_kind0["canonical_candidate_vertex_row"])!=KIND0_CANONICAL or int(o_kind0["baseline806_node_kind"])!=0 or int(o_kind0["baseline806_node_source_row"])!=KIND0_COARSE_SOURCE:
  raise ValueError("kind0 compact/canonical/coarse source domains do not match the pinned crosswalk")
 coarse_capture_node=maprows[KIND0_COARSE_SOURCE]
 incident=[]
 for t,tetnodes in enumerate(parent["tets"]):
  if coarse_capture_node in tetnodes and int(parent["materials"][t])==15:
   incident.append(t)
 eligible=[t for t in incident if int(parent["donors"][t])==int(o_kind0["donor"]) and
             int(parent["regions"][t])==int(o_kind0["donor"]) and int(parent["faces"][t])==support_face]
 if len(eligible)!=1:raise ValueError(f"kind0 P1 stable-face tie is not unique: {eligible}")
 kind0_tet=eligible[0];kind0_tetnodes=parent["tets"][kind0_tet]
 corner=np.flatnonzero(kind0_tetnodes==coarse_capture_node)
 if len(corner)!=1:raise ValueError("kind0 coarse source node is not a unique tet corner")
 kind0_bary=np.zeros(4,dtype=np.float64);kind0_bary[int(corner[0])]=1.0
 # Recompute every parent residual against native-current points and capture016
 # current parent nodes. Preserve existing tets/bary except six+kind0 selection.
 _,registration_rows=read_csv(BASE/"node-parent-registration.csv")
 if len(registration_rows)!=62046:raise ValueError("attempt016 registration rows do not cover 62,046 O/I/D nodes")
 surfaces={"O":o_out,"I":id_out["I"],"D":id_out["D"]}
 registr_by={(r["layer"],int(r["surface_vertex_row"])):dict(r) for r in registration_rows}
 regs=[];max_parent_distance={l:0.0 for l in surfaces};max_reconstruction={l:0.0 for l in surfaces};max_rows={}
 for layer,rows in surfaces.items():
  for i,surf in enumerate(rows):
   r=registr_by[(layer,i)]
   if layer=="O" and i in six_selection:
    sel=six_selection[i];r["parent_registration_kind"]=INFERRED_KIND
    r["parent_tet_row"]=str(sel["parent_tet_row"])
    for c,v in enumerate(sel["bary"]):r[f"parent_bary{c}"]=f64(v)
    r["parent_registration_tie_count"]=str(sel["tie_count"])
   elif layer=="O" and i==KIND0_ROW:
    r["parent_registration_kind"]="source_parent_tet_reference_registration"
    r["parent_tet_row"]=str(kind0_tet)
    for c,v in enumerate(kind0_bary):r[f"parent_bary{c}"]=f64(v)
    r["parent_registration_tie_count"]="0"
   tet=int(r["parent_tet_row"])
   point=np.asarray([float(surf[k]) for k in ("wx","wy","wz")],dtype=np.float64)
   if tet<0:
    if layer!="O":raise ValueError(f"unmapped {layer} row {i}")
    continue
   if tet>=len(parent["tets"]):raise ValueError(f"parent tet out of range for {layer} row {i}")
   bary=np.asarray([float(r[f"parent_bary{c}"]) for c in range(4)],dtype=np.float64)
   if np.min(bary)<-1e-8 or np.max(bary)>1+1e-8 or abs(float(bary.sum())-1)>1e-8:
    raise ValueError(f"invalid parent barycentrics {layer} row {i}")
   pm=int(parent["materials"][tet]);pd=int(parent["donors"][tet]);pr=int(parent["regions"][tet])
   if (layer=="O" and pm!=15) or (layer=="I" and pm not in (15,16)) or (layer=="D" and pm!=16):
    raise ValueError(f"parent material-role mismatch {layer} row {i}: material={pm}")
   if layer=="O" and i in six_selection:
    if pd!=int(surf["donor"]) or pr!=int(support[i]["selected_anatomical_region_id"]):
     raise ValueError(f"inferred parent semantic labels mismatch O row {i}")
   if layer=="O" and i==KIND0_ROW and (pd!=int(surf["donor"]) or pr!=int(surf["donor"])):
    raise ValueError("kind0 direct coarse parent cell semantic labels mismatch")
   corner_points=parent["points"][parent["tets"][tet]]
   mapped=bary@corner_points;residual=point-mapped;dist=float(np.linalg.norm(residual))
   if not math.isfinite(dist) or dist>PARENT_MAX_M:raise ValueError(f"native-current {layer} row {i} violates 2cm registration bound: {dist}")
   recon=float(np.linalg.norm(mapped+residual-point))
   if recon>2e-8:raise ValueError(f"native-current parent residual does not reconstruct {layer} row {i}: {recon}")
   for a,v in zip("xyz",residual):r[f"parent_residual_{a}_m"]=f64(v)
   r["parent_registration_distance_m"]=f64(dist)
   # Surface ancestry metadata is kept separate and copied byte-for-byte in
   # meaning; it is not inferred from the selected current parent tet.
   if layer=="O":
    source_domain=surf["source_domain"];sv=int(surf["source_vertex_row"]);sf=int(surf["source_face_row"])
    barysrc=[surf[f"source_bary{k}"] for k in range(3)]
    sr=[surf[f"source_residual_{a}_m"] for a in "xyz"]
   else:
    source_domain=surf["source_domain"];sv=int(surf["d37_parent_node_row"]);sf=int(surf["d37_parent_surface_face_row"])
    barysrc=[surf[f"beta{k}"] for k in range(3)]
    sr=[surf[f"source_residual_{a}_m"] for a in "xyz"]
   r["source_domain"]=source_domain;r["source_vertex_row"]=str(sv);r["source_face_row"]=str(sf)
   r["frame_owner_method"]=surf["frame_owner_method"]
   for c,v in enumerate(barysrc):r[f"source_bary{c}"]=v
   for a,v in zip("xyz",sr):r[f"source_residual_{a}_m"]=v
   max_parent_distance[layer]=max(max_parent_distance[layer],dist)
   max_reconstruction[layer]=max(max_reconstruction[layer],recon)
   if dist==max_parent_distance[layer]:max_rows[layer]=i
   regs.append(r)
 # Keep source-facing row order stable (O then I then D), as in attempt016.
 reg_header=list(registration_rows[0].keys())
 write_csv(OUT/"node-parent-registration.csv",reg_header,regs)
 # Preserve 62,046 node-source rows and exact topology/ancestry arrays.
 if len(regs)!=62046 or [int(r["surface_vertex_row"]) for r in regs[:32002]]!=list(range(32002)):
  raise ValueError("native-current registration output row coverage differs")
 # Re-read tables and enforce exact donor-frame/local and parent reconstruction gates.
 for layer,rows in surfaces.items():
  for i,surf in enumerate(rows):
   donor=int(surf["donor"]);fr=frame_by[donor]
   M=np.asarray([[float(surf[f"m{a}{b}"]) for b in range(3)] for a in range(3)])
   if not np.array_equal(M,fr["local_matrix"]):raise ValueError(f"current frame matrix mismatch {layer} row {i}")
   pt=np.asarray([float(surf[k]) for k in ("wx","wy","wz")]);loc=np.asarray([float(surf[k]) for k in ("lx","ly","lz")])
   local_error=float(np.linalg.norm(M@(pt-fr["origin"])-loc))
   if local_error>2e-12:raise ValueError(f"current donor-local reconstruction exceeds export precision {layer} row {i}: {local_error}")
 # Compute native-observation roundoff and transformed ID Float32 rounding.
 counts={l:len(rows) for l,rows in surfaces.items()}
 base_files={p.name:sha(p) for p in BASE.iterdir() if p.is_file()}
 input_paths=[NATIVE_O,POSES,RUN/"stdout.txt",CAP,Path(str(CAP)+".nodes.csv"),Path(str(CAP)+".tets.csv"),
              Path(str(CAP)+".labels.csv"),OWNER/"accepted-transfer-rows.csv.gz",KIND0_NODE_MAP,BASE/"receipt.json"]
 output_hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!="receipt.json"}
 report={"artifact":"native-current registered O/I/D sidecars","status":"reference-registration-exported_not_physical_admission",
  "physical_admission":False,"geometry_admission":False,"motion_or_step_claim":False,
  "method":"single common native SE(3) coordinate frame; O points from run006 F32 observation; I/D authoring points transformed then F32; F64 donor frames/locals rebuilt from run006 native poses",
  "source_ancestry_separate_from_parent_registration":True,"source_payload_and_canonical_authoring_preserved":True,
  "input_attempt016_receipt_sha256":base_receipt_sha,"attempt016_input_hashes":base_files,
  "native_current_inputs_sha256":{str(p):sha(p) for p in input_paths},
  "source_payload_sha256":base_receipt.get("source_payload_sha256",base_receipt.get("candidate_payload_sha256")),
  "canonical_conditioned_O_sha256":"5e686b14d9360267f4b40bbeb5ed6c9c68bf871933874b5aa8eaa5898757ec2b",
  "compact_O_map_sha256":sha(BASE/"O-compact-node-map.csv"),"compact_O_map_rows":len(map_rows),
  "common_native_SE3":{"rotation_row_major":common_R.reshape(-1).tolist(),"translation_m":common_t.tolist(),
                       "source":"run006 stdout native placement witness"},
  "native_O_F32_observation":{"rows":len(o_out),"max_difference_from_SE3_prediction_m":max(native_observation_roundoff),
     "mean_difference_from_SE3_prediction_m":float(np.mean(native_observation_roundoff)),"source_hash":sha(NATIVE_O)},
  "current_native_donor_frames":{"rows":len(frames),"source_hash":sha(POSES),"F64_quaternion_matrix_matches_attempt016_frames_exactly":True,
                                  "local_and_frame_maps_recomputed_from_pose_json":True},
  "surface_rows":counts,"typed_node_ancestry_rows":sum(counts.values()),
  "outer_faces":len(read_csv(OUT/"registered-O-faces.csv")[1]),
  "parent_current_geometry":"capture016 current_x/y/z, same exact pinned parent cell/material/donor/region/face packet",
  "max_parent_registration_distance_m":max_parent_distance,"max_parent_reconstruction_error_m":max_reconstruction,
  "max_parent_distance_row":max_rows,"parent_distance_bound_m":PARENT_MAX_M,
  "six_inferred_current_parent_registrations":{"method":INFERRED_KIND,"source_ancestry_claimed":False,"rows":six_selection,
       "required_labels":"material15 + source geometric donor + source anatomical region"},
  "kind0_direct_coarse_registration":{"compact_O_row":KIND0_ROW,"canonical_candidate_and_O033_row":KIND0_CANONICAL,
       "coarse_representative_row":KIND0_COARSE_SOURCE,"capture016_parent_node":coarse_capture_node,
       "P1_binding_support_detail_row":KIND0_SUPPORT_DETAIL,"P1_support_stable_source_face_id":support_face,
       "incident_material15_tets":len(incident),"unique_support_face_tet_row":kind0_tet,
       "selected_geometric_parent_stable_face_id":int(parent["faces"][kind0_tet]),
       "d37_ancestry_and_P1_binding_support_are_not_parent_source_identity":True},
  "O_face_ancestry_and_surface_topology":"copied unchanged from attempt016; O point row domain exact run006 compact candidate table",
  "I_D_surface_topology_and_face_ancestry":"copied unchanged; only world coordinate frame and derived local/frame values rebuilt",
  "output_files_sha256":output_hashes}
 (OUT/"receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"output":str(OUT),"rows":counts,"parent_max_distance_m":max_parent_distance,
       "parent_max_reconstruction_m":max_reconstruction,"six_inferred":{str(k):{"tet":v["parent_tet_row"],"stableface":v["geometric_registration_stable_source_face_id"],"distance_m":v["distance_m"],"ties":v["tie_count"]} for k,v in six_selection.items()},
       "kind0":{"compact_row":KIND0_ROW,"canonical_row":KIND0_CANONICAL,"coarse":KIND0_COARSE_SOURCE,"support_face":support_face,"parent_tet":kind0_tet},
       "native_O_roundoff_max_m":max(native_observation_roundoff)},indent=2))

if __name__=="__main__":main()
