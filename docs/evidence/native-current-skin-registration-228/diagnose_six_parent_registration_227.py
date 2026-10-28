#!/usr/bin/env python3
"""Bounded exact nearest-parent diagnostic for six O rows above the 2 cm gate.

This changes no surface or parent data. It applies the native run006 common
SE(3) only to comparison targets and labels every result as an inferred
reference registration diagnostic, never geometric ancestry or mass transfer.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, math, re, sys
from pathlib import Path
import numpy as np

P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
ROOT=P/"volume-worker/registered-surface-sidecar-222"
INPUT=ROOT/"attempt-016-kindaware-registration"
CAP=P/"native-parent-label-capture-016/compiled-parent-cell-mass.csv"
RUN=P/"P/native-kind-aware-source-preflight-226/run-006-native-emission"
D37_SOURCE=Path("/Users/numi/Work/numi-human-connected-surface-069/publication-070/evidence/mass/compiled-cell-export-068/run-003/source-preflight/registered-skin.csv.gz")
NATIVE_O=RUN/"registered-skin-100-native.csv.candidate033.csv"
OUT=ROOT/"attempt-016-kindaware-registration/six-parent-registration-diagnostic-227.json"
IDS=[13498,13499,16946,17203,17204,17223]
MAX_DIAGNOSTIC_DISTANCE_M=0.10
AMBIGUITY_TOLERANCE_M=1.0e-8
sys.path.insert(0,str(P/"volume-worker/native-domain-217"))
import export_child_semantic_labels_217 as native

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def csvrows(path:Path):
 with path.open(newline="") as f:
  r=csv.DictReader(f); rows=list(r)
  if not r.fieldnames or any(None in x for x in rows):raise ValueError(f"malformed CSV: {path}")
 return r.fieldnames,rows

def nearest(parent,index,original_rows,point):
 """Exact point-to-tetrahedron distance over a KD-tree bounded candidate set."""
 center_dist,seed_local=index["tree"].query(point,k=1)
 if not math.isfinite(float(center_dist)):raise ValueError("empty nearest-parent index")
 _,_,seed_dist=native.closest_point_on_parent_tet(parent,index,int(seed_local),point)
 radius=seed_dist+index["max_radius"]+AMBIGUITY_TOLERANCE_M+native.QUERY_PAD_M
 local_candidates=index["tree"].query_ball_point(point,radius)
 measured=[]
 for local in local_candidates:
  reg,bary,d=native.closest_point_on_parent_tet(parent,index,int(local),point)
  measured.append((float(d),int(local),int(original_rows[int(local)]),reg,bary))
 if not measured:raise ValueError("bounded candidate query found no parent tet")
 best=min(x[0] for x in measured)
 tied=[x for x in measured if abs(x[0]-best)<=AMBIGUITY_TOLERANCE_M]
 tied.sort(key=lambda x:(x[0],x[2]))
 return {"nearest_distance_m":best,"minimum_distance_parent_tet_row":tied[0][2],"candidate_tets_within_ambiguity_tolerance":len(tied),
         "ties":[{"parent_tet_row":row,"distance_m":d,"material_index":int(parent["materials"][local]),
                  "geometric_donor_body":int(parent["donors"][local]),
                  "anatomical_region_id":int(parent["regions"][local]),
                  "stable_source_face_id":int(parent["faces"][local]),
                  "barycentric":np.asarray(bary).tolist(),
                  "registered_current_world_m":np.asarray(reg).tolist()} for d,local,row,reg,bary in tied[:32]],
         "ties_truncated":len(tied)>32}

def main():
 ocols,orows=csvrows(INPUT/"registered-O.csv")
 _,support=csvrows(INPUT/"accepted-transfer-reference-support.csv")
 support={int(r["surface_vertex_row"]):r for r in support}
 o={int(r["vertex"]):r for r in orows}
 _,native_o_rows=csvrows(NATIVE_O)
 native_o={int(r["vertex"]):r for r in native_o_rows}
 if len(native_o)!=32002 or any(int(native_o[i]["vertex"])!=i for i in range(32002)):raise ValueError("run006 native O Float32 table is not the exact compact row domain")
 source_rows={}
 with gzip.open(D37_SOURCE,"rt",newline="") as stream:
  for q in csv.DictReader(stream):
   if int(q["vertex"]) in {int(o[i]["d37_source_vertex_row"]) for i in IDS}:source_rows[int(q["vertex"])]=q
 if len(source_rows)!=len(IDS):raise ValueError("D37 exact detail source rows are incomplete")
 parent=native.read_parent(CAP)
 _,nodes=csvrows(Path(str(CAP)+".nodes.csv"))
 current=np.asarray([[float(r[k]) for k in ("current_x_m","current_y_m","current_z_m")] for r in nodes],dtype=np.float64)
 if len(current)!=len(parent["points"]):raise ValueError("current node rows differ from actual parent table")
 parent["points"]=current
 stdout=(RUN/"stdout.txt").read_text()
 line=next((x for x in stdout.splitlines() if "candidate033_native_source_frame_measurement" in x),None)
 if line is None:raise ValueError("run006 common placement measurement is missing")
 rm=re.search(r"common_placement_rotation=\[([^]]+)\]",line)
 tm=re.search(r"common_placement_translation_m=\[([^]]+)\]",line)
 if not rm or not tm:raise ValueError("run006 did not serialize the full common SE(3)")
 R=np.asarray([float(x) for x in rm.group(1).split(",")],dtype=np.float64).reshape(3,3)
 t=np.asarray([float(x) for x in tm.group(1).split(",")],dtype=np.float64)
 if np.linalg.det(R)<0.999999 or np.linalg.norm(R.T@R-np.eye(3),ord=np.inf)>1e-12:raise ValueError("run006 placement is not a proper rigid rotation")
 masks={"material15":(parent["materials"]==15)}
 for donor in (42,150):
  masks[f"material15_donor{donor}"]=(parent["materials"]==15)&(parent["donors"]==donor)
  masks[f"material15_donor{donor}_region{donor}"]=(parent["materials"]==15)&(parent["donors"]==donor)&(parent["regions"]==donor)
 indexes={}; rows_by={}
 for name,mask in masks.items():
  rows=np.flatnonzero(mask)
  if len(rows)==0:raise ValueError(f"empty semantic parent subset {name}")
  sub={k:(v[rows] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(parent["tets"]) else v) for k,v in parent.items()}
  rows_by[name]=rows; indexes[name]=(sub,native.build_index(sub))
 results=[]
 for rowid in IDS:
  r=o[rowid]; s=support[rowid]
  raw=np.asarray([float(r[k]) for k in ("wx","wy","wz")],dtype=np.float64)
  predicted=(R@raw+t).astype(np.float32).astype(np.float64)
  target=np.asarray([float(native_o[rowid][k]) for k in ("wx","wy","wz")],dtype=np.float64)
  transform_residual=float(np.linalg.norm(target-predicted))
  donor=int(r["donor"]); region=int(s["selected_anatomical_region_id"])
  source_faces=[int(x) for x in s["accepted_transfer_stable_face_ids"].split(";") if x]
  rec={"surface_vertex_row":rowid,"canonical_candidate_vertex_row":int(r["canonical_candidate_vertex_row"]),
       "baseline806_node_row":int(r["baseline806_node_row"]),"baseline806_node_kind":int(r["baseline806_node_kind"]),
       "baseline806_node_source_row":int(r["baseline806_node_source_row"]),
       "source_domain":r["source_domain"],"source_mapping_kind":r["source_mapping_kind"],
       "d37_source_vertex_row":int(r["d37_source_vertex_row"]),"donor":donor,
       "source_anatomical_region_id_from_accepted_transfer":region,"source_stable_face_ids":source_faces,
       "canonical_authoring_point_m":raw.tolist(),"run006_native_f32_current_point_m":target.tolist(),
       "run006_native_vs_rigid_transform_residual_m":transform_residual,
       "d37_exact_detail_source_point_m":[float(source_rows[int(r["d37_source_vertex_row"])][k]) for k in ("wx","wy","wz")],
       "distance_to_exact_d37_detail_source_m":float(np.linalg.norm(raw-np.asarray([float(source_rows[int(r["d37_source_vertex_row"])][k]) for k in ("wx","wy","wz")],dtype=np.float64))),
       "accepted_transfer_selected_parent_tet_row":int(s["parent_tet_row"]),
       "accepted_transfer_registration_distance_m":float(s["candidate_parent_registration_distance_m"]),
       "parent_queries":{}}
  for name in ["material15",f"material15_donor{donor}",f"material15_donor{donor}_region{region}"]:
   sub,idx=indexes[name]
   rec["parent_queries"][name]=nearest(sub,idx,rows_by[name],target)
  # Source stable-face restriction is useful as an ancestry-support diagnostic,
  # but it is intentionally separate from the geometry-only closest search.
  for face in source_faces:
   mask=(parent["materials"]==15)&(parent["donors"]==donor)&(parent["regions"]==region)&(parent["faces"]==face)
   rows=np.flatnonzero(mask)
   key=f"material15_donor{donor}_region{region}_stableface{face}"
   if len(rows):
    sub={k:(v[rows] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(parent["tets"]) else v) for k,v in parent.items()}
    rec["parent_queries"][key]=nearest(sub,native.build_index(sub),rows,target)
   else:rec["parent_queries"][key]={"available_parent_tets":0}
  results.append(rec)
 inputs=[INPUT/"registered-O.csv",INPUT/"accepted-transfer-reference-support.csv",D37_SOURCE,NATIVE_O,CAP,Path(str(CAP)+".nodes.csv"),Path(str(CAP)+".tets.csv"),Path(str(CAP)+".labels.csv"),RUN/"stdout.txt"]
 report={"status":"diagnostic_only_not_admission","method":"exact point-to-F32-current-parent-tetrahedra distance over conservative KD-tree candidate sets",
         "geometry_ancestry_claimed":False,"mass_transfer_claimed":False,"physical_admission":False,
         "source_parent_capture":"native-parent-label-capture-016",
         "source_geometry_domain":"exact D37 detail vertex coordinates from compiled-cell-export-068 run003; source identity does not imply parent-cell ancestry","target_coordinates":"actual run006 native-emitted O Float32 rows; exact SE(3) retained and per-row comparison residual reported",
         "diagnostic_max_distance_m":MAX_DIAGNOSTIC_DISTANCE_M,"tie_ambiguity_tolerance_m":AMBIGUITY_TOLERANCE_M,
         "common_placement_rotation_row_major":R.reshape(-1).tolist(),"common_placement_translation_m":t.tolist(),
         "inputs_sha256":{str(p):sha(p) for p in inputs},"rows":results}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"output":str(OUT),"rows":len(results),"per_row":[{"row":x["surface_vertex_row"],"donor":x["donor"],"region":x["source_anatomical_region_id_from_accepted_transfer"],"queries":{k:{"distance_m":v.get("nearest_distance_m"),"ties":v.get("candidate_tets_within_ambiguity_tolerance"),"first":v.get("ties",[])[:1]} for k,v in x["parent_queries"].items()}} for x in results]},indent=2))
if __name__=="__main__":main()
