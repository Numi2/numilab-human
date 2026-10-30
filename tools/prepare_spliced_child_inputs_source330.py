#!/usr/bin/env python3
"""Package a local retet using direct, bounded source330 centroid registration.

This fork preserves unchanged outside cells and protected-node identities, but it
does not require changed cells to lie inside an old current006 parent tet. Each
changed child instead carries the existing 217 source330 semantic-registration
row (parent tet, barycentric point, labels, and measured residual). It emits no
mass fractions and makes no physical-admission claim.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, shutil
from pathlib import Path
import numpy as np
from human_skin_fat_retet_mapping import parse_operational_node_map, validate_combined_node_map

NODE_FIELDS=("points_donor_local_m_f64","parent_volume_node_row","full86_weights",
             "target_current_world_m","refinement_node_source_offsets",
             "refinement_node_source_rows","refinement_node_source_weights")
ROLE_FIELDS=("coarse_boundary_faces_O","coarse_boundary_faces_I","coarse_boundary_faces_D")

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def readnpz(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def readcsv(p):
 with Path(p).open("r",encoding="utf-8",newline="") as f:
  r=csv.DictReader(f)
  if r.fieldnames is None:raise ValueError("missing CSV header")
  out=list(r)
  if any(None in x for x in out):raise ValueError("malformed CSV row width")
  return r.fieldnames,out
def outface(t,o):
 q=[int(t[j]) for j in range(4) if j!=o]
 if o%2:q[1],q[2]=q[2],q[1]
 return tuple(q)
def key(f):return tuple(sorted(map(int,f)))
def cyc(f):
 q=list(map(int,f));i=q.index(min(q));return tuple(q[i:]+q[:i])
def volume(p,t):
 q=np.asarray(p,dtype=np.float64)[np.asarray(t,dtype=np.int64)]
 return np.einsum("ij,ij->i",q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))/6
def incidence(t,m):
 out={}
 for ci,tt in enumerate(np.asarray(t,dtype=np.int64)):
  for o in range(4):
   f=outface(tt,o);out.setdefault(key(f),[]).append((ci,f))
 for k,h in out.items():
  if len(h)>2:raise ValueError(f"nonmanifold face {k}")
  if len(h)==2 and cyc(h[0][1])==cyc(h[1][1]):raise ValueError(f"same-wound internal face {k}")
 return out
def write_node(path,p):
 with Path(path).open("w",encoding="utf-8",newline="\n") as f:
  f.write(f"{len(p)} 3 0 0\n")
  for i,x in enumerate(np.asarray(p,dtype=np.float32)):
   f.write(f"{i} {float(x[0]):.17g} {float(x[1]):.17g} {float(x[2]):.17g}\n")
def write_ele(path,t,m):
 with Path(path).open("w",encoding="utf-8",newline="\n") as f:
  f.write(f"{len(t)} 4 1\n")
  for i,(tt,mat) in enumerate(zip(np.asarray(t,dtype=np.int64),np.asarray(m,dtype=np.int64))):
   f.write(f"{i} {tt[0]} {tt[1]} {tt[2]} {tt[3]} {int(mat)}\n")
def write_labels(path,m,d,r,face):
 with Path(path).open("w",encoding="utf-8",newline="") as f:
  w=csv.writer(f,lineterminator="\n");w.writerow(["child_tet_row","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"])
  for i in range(len(m)):w.writerow([i,int(m[i]),int(d[i]),int(r[i]),int(face[i])])
def write_map(path,src,m,d,r,face,kind):
 with Path(path).open("w",encoding="utf-8",newline="") as f:
  w=csv.writer(f,lineterminator="\n");w.writerow(["child_tet_row","source330_parent_tet_row","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id","mapping_kind"])
  for i in range(len(src)):w.writerow([i,int(src[i]),int(m[i]),int(d[i]),int(r[i]),int(face[i]),kind[i]])
def parse_node(path):
 lines=[x.split("#",1)[0].strip() for x in Path(path).read_text().splitlines() if x.split("#",1)[0].strip()]
 n,dim,attrs,mark=map(int,lines[0].split()[:4])
 if dim!=3 or attrs!=0 or mark not in (0,1):raise ValueError("compare node header invalid")
 vals=[]
 for i,row in enumerate(lines[1:1+n]):
  q=row.split()
  if int(q[0])!=i:raise ValueError("compare node IDs are not contiguous zero-based rows")
  vals.append([np.float32(q[1]),np.float32(q[2]),np.float32(q[3])])
 if len(vals)!=n:raise ValueError("compare node rows truncated")
 return np.asarray(vals,dtype=np.float32)
def parse_ele(path):
 lines=[x.split("#",1)[0].strip() for x in Path(path).read_text().splitlines() if x.split("#",1)[0].strip()]
 n,nv,na=map(int,lines[0].split()[:3])
 if nv!=4 or na!=1:raise ValueError("compare ele header invalid")
 t=[];m=[]
 for row in lines[1:1+n]:
  q=row.split();t.append([int(x) for x in q[1:5]]);m.append(int(float(q[5])))
  if int(q[0])!=len(t)-1:raise ValueError("compare element IDs are not contiguous zero-based rows")
 if len(t)!=n:raise ValueError("compare ele rows truncated")
 return np.asarray(t,dtype=np.int64),np.asarray(m,dtype=np.int64)
def check_nodes(base,geom,side):
 p=np.asarray(geom["points_current_world_m_f32"],dtype=np.float32)
 ref=np.asarray(geom["points_reference_world_m_f64"],dtype=np.float64)
 if p.ndim!=2 or p.shape[1]!=3 or ref.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(ref).all():
  raise ValueError("geometry point fields invalid")
 n0=len(base["points_current_world_m_f32"]);pins=np.asarray(base.get("pinned_volume_node_rows",np.empty(0)),dtype=np.int64)
 if side is None:
  if len(p)!=n0 or not np.array_equal(p,base["points_current_world_m_f32"]) or not np.array_equal(ref,base["points_reference_world_m_f64"]):
   raise ValueError("changed/retired/new nodes require explicit sidecars")
  fields={k:np.asarray(base[k]).copy() for k in NODE_FIELDS}
  base_rows=np.arange(n0,dtype=np.int64)
  protected_base=pins.copy();protected_out=pins.copy()
  protected_kind=np.full(len(pins),"retained_original_node",dtype="U32")
  protected_local=np.full(len(pins),-1,dtype=np.int64)
 else:
  required=set(NODE_FIELDS)|{"base_node_output_rows","protected_base_node_rows",
      "protected_node_output_rows","protected_node_mapping_kind","protected_node_local_rows"}
  missing=required-set(side)
  if missing: raise ValueError(f"node sidecars missing {sorted(missing)}")
  base_rows=np.asarray(side["base_node_output_rows"],dtype=np.int64)
  if base_rows.shape!=(n0,) or np.any(base_rows < -1) or np.any(base_rows>=len(p)) or len(np.unique(base_rows[base_rows>=0]))!=int(np.sum(base_rows>=0)):
   raise ValueError("base node output map invalid")
  if "base_node_output_rows" in geom and not np.array_equal(base_rows,np.asarray(geom["base_node_output_rows"],dtype=np.int64)):
   raise ValueError("geometry/sidecar base-node maps differ")
  retired=np.flatnonzero(base_rows<0).astype(np.int64)
  if "retired_base_node_rows" not in geom or not np.array_equal(retired,np.asarray(geom["retired_base_node_rows"],dtype=np.int64)):
   raise ValueError("retired base-node rows lack exact geometry manifest")
  for old,new in enumerate(base_rows):
   if new>=0 and not np.array_equal(p[new],base["points_current_world_m_f32"][old]):
    raise ValueError(f"retained original node moved at base row {old}")
  fields={k:np.asarray(side[k]).copy() for k in NODE_FIELDS}
  if fields["points_donor_local_m_f64"].shape!=(len(p),3) or fields["target_current_world_m"].shape!=(len(p),3):
   raise ValueError("node coordinate sidecar row domain mismatch")
  if fields["parent_volume_node_row"].shape!=(len(p),) or fields["full86_weights"].shape!=(len(p),86):
   raise ValueError("parent/full86 node sidecar row domain mismatch")
  offsets=np.asarray(fields["refinement_node_source_offsets"],dtype=np.int64)
  src=np.asarray(fields["refinement_node_source_rows"],dtype=np.int64)
  weights=np.asarray(fields["refinement_node_source_weights"],dtype=np.float64)
  if offsets.shape!=(len(p)+1,) or offsets[0]!=0 or offsets[-1]!=len(src) or len(src)!=len(weights) or np.any(np.diff(offsets)<0) or np.any(src<0) or not np.isfinite(weights).all():
   raise ValueError("node source CSR invalid")
  for name in ("points_donor_local_m_f64","parent_volume_node_row","full86_weights","target_current_world_m"):
   for old,new in enumerate(base_rows):
    if new>=0 and not np.array_equal(fields[name][new],np.asarray(base[name])[old]):
     raise ValueError(f"retained node sidecar changed: {name} at base row {old}")
  oldoff=np.asarray(base["refinement_node_source_offsets"],dtype=np.int64)
  oldsrc=np.asarray(base["refinement_node_source_rows"],dtype=np.int64)
  oldw=np.asarray(base["refinement_node_source_weights"],dtype=np.float64)
  for old,new in enumerate(base_rows):
   if new<0: continue
   a,b=oldoff[old:old+2];c,d=offsets[new:new+2]
   if not np.array_equal(src[c:d],oldsrc[a:b]) or not np.array_equal(weights[c:d],oldw[a:b]):
    raise ValueError(f"retained source CSR changed at base row {old}")
  protected_base=np.asarray(side["protected_base_node_rows"],dtype=np.int64)
  protected_out=np.asarray(side["protected_node_output_rows"],dtype=np.int64)
  protected_kind=np.asarray(side["protected_node_mapping_kind"]).astype("U32")
  protected_local=np.asarray(side["protected_node_local_rows"],dtype=np.int64)
  if not np.array_equal(protected_base,pins) or protected_out.shape!=pins.shape or protected_kind.shape!=pins.shape or protected_local.shape!=pins.shape:
   raise ValueError("canonical protected-node routing has wrong row domain")
  for keyname,expected in (("protected_base_node_rows",protected_base),("protected_node_output_rows",protected_out),
                           ("protected_node_mapping_kind",protected_kind),("protected_node_local_rows",protected_local)):
   if keyname not in geom or not np.array_equal(np.asarray(geom[keyname]),expected):
    raise ValueError(f"geometry/sidecar protected map differs: {keyname}")
  if not np.array_equal(np.asarray(geom.get("pinned_volume_node_rows",np.empty(0)),dtype=np.int64),protected_out):
   raise ValueError("output protected-node target list differs")
  if np.any(protected_out<0) or np.any(protected_out>=len(p)) or len(np.unique(protected_out))!=len(protected_out):
   raise ValueError("protected target node rows are invalid or duplicated")
  for old,outrow,kind,local in zip(protected_base,protected_out,protected_kind,protected_local):
   if kind=="retained_original_node":
    if base_rows[int(old)]!=outrow or not np.array_equal(p[outrow],base["points_current_world_m_f32"][int(old)]):
     raise ValueError(f"retained protected row {old} is not exact")
    if local!=-1: raise ValueError("retained protected target unexpectedly names a retet-local row")
   elif kind=="explicit_retet_descendant":
    if local<0: raise ValueError(f"retargeted protected row {old} lacks explicit local mapping")
    local_outputs=np.asarray(geom.get("retet_node_output_rows",np.empty(0)),dtype=np.int64)
    if local>=len(local_outputs) or int(local_outputs[local])!=outrow:
     raise ValueError(f"retargeted protected row {old} local/output mapping mismatch")
    if base_rows[int(old)]!=-1:
     raise ValueError(f"retargeted protected base row {old} was not retired")
   else: raise ValueError("unsupported protected-node mapping kind")
 if not np.array_equal(p.astype(np.float32),ref.astype(np.float32)):
  raise ValueError("current/reference F32 geometry differs")
 offsets=np.asarray(fields["refinement_node_source_offsets"],dtype=np.int64)
 src=np.asarray(fields["refinement_node_source_rows"],dtype=np.int64)
 weights=np.asarray(fields["refinement_node_source_weights"],dtype=np.float64)
 max_parent_node=int(np.max(base["refinement_node_source_rows"])) if len(base["refinement_node_source_rows"]) else -1
 if offsets.shape!=(len(p)+1,) or offsets[0]!=0 or offsets[-1]!=len(src) or len(src)!=len(weights) or np.any(np.diff(offsets)<1):
  raise ValueError("node source CSR has empty or invalid row domain")
 if np.any(src<0) or np.any(src>max_parent_node) or not np.isfinite(weights).all() or np.any(weights<0):
  raise ValueError("node source CSR rows/weights outside original330 node domain")
 rowsums=np.asarray([weights[offsets[i]:offsets[i+1]].sum() for i in range(len(p))])
 if np.max(np.abs(rowsums-1.0))>1e-8: raise ValueError("node source CSR weights do not sum to one")
 if not np.isfinite(fields["full86_weights"]).all() or np.any(fields["full86_weights"] < -1e-10) or np.max(np.abs(fields["full86_weights"].sum(axis=1)-1))>1e-5:
  raise ValueError("full86 node weights invalid")
 if not np.isfinite(fields["points_donor_local_m_f64"]).all() or not np.isfinite(fields["target_current_world_m"]).all():
  raise ValueError("node coordinate sidecar contains nonfinite values")
 if np.any(fields["parent_volume_node_row"] < -1): raise ValueError("parent-volume node row invalid")
 return p,ref,fields,base_rows,protected_base,protected_out,protected_kind,protected_local



def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--base-npz",type=Path,required=True);ap.add_argument("--geometry-npz",type=Path,required=True)
 ap.add_argument("--cell-ancestry-csv",type=Path,required=True);ap.add_argument("--parent-labels-csv",type=Path,required=True)
 ap.add_argument("--parent-nodes-csv",type=Path,required=True);ap.add_argument("--parent-tets-csv",type=Path,required=True)
 ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--node-sidecars-npz",type=Path)
 ap.add_argument("--boundary-sidecars-npz",type=Path);ap.add_argument("--compare-cli-dir",type=Path)
 ap.add_argument("--maximum-operational-reference-distance-m",type=float,default=.0004)
 ap.add_argument("--combined-node-map-csv",type=Path,required=True)
 ap.add_argument("--receipt-json",type=Path,required=True)
 a=ap.parse_args()
 if not math.isfinite(a.maximum_operational_reference_distance_m) or a.maximum_operational_reference_distance_m<=0:raise ValueError("operational-reference distance bound invalid")
 base=readnpz(a.base_npz);g=readnpz(a.geometry_npz);side=readnpz(a.node_sidecars_npz) if a.node_sidecars_npz else None
 bside=readnpz(a.boundary_sidecars_npz) if a.boundary_sidecars_npz else {}
 fields,maprows=readcsv(a.cell_ancestry_csv);pfields,parentrows=readcsv(a.parent_labels_csv)
 required={"child_tet_row","selected_parent_tet_row","label_match_kind",
  "centroid_barycentric0","centroid_barycentric1","centroid_barycentric2","centroid_barycentric3",
  "centroid_registered_reference_x_m","centroid_registered_reference_y_m","centroid_registered_reference_z_m",
  "centroid_reference_residual_x_m","centroid_reference_residual_y_m","centroid_reference_residual_z_m",
  "centroid_reference_residual_norm_m","reference_registration_distance_m",
  "reference_registration_max_distance_m","reference_registration_ambiguity_tolerance_m",
  "matching_parent_tet_count","semantic_tie_count","material_index","geometric_donor_body",
  "anatomical_region_id","stable_source_face_id"}
 if not required.issubset(fields):raise ValueError(f"217 source330 cell map missing required fields {sorted(required-set(fields))}")
 if pfields!=["parent_tet_row","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"]:raise ValueError("source330 label header mismatch")
 parent={int(r["parent_tet_row"]):r for r in parentrows}
 if set(parent)!=set(range(len(parentrows))):raise ValueError("source330 label rows not contiguous")
 _,parent_node_rows=readcsv(a.parent_nodes_csv);_,parent_tet_rows=readcsv(a.parent_tets_csv)
 if any(int(row["parent_node_index"])!=i for i,row in enumerate(parent_node_rows)):raise ValueError("source330 node rows not contiguous")
 if any(int(row["parent_tet_row"])!=i for i,row in enumerate(parent_tet_rows)) or len(parent_tet_rows)!=len(parentrows):raise ValueError("source330 tet rows not contiguous or label count mismatch")
 source_points=np.asarray([[float(row[k]) for k in ("current_x_m","current_y_m","current_z_m")] for row in parent_node_rows],dtype=np.float64)
 source_tets=np.asarray([[int(row[f"parent_node{k}"]) for k in range(4)] for row in parent_tet_rows],dtype=np.int64)
 if source_points.ndim!=2 or source_points.shape[1]!=3 or not np.isfinite(source_points).all() or np.any(source_tets<0) or np.any(source_tets>=len(source_points)):raise ValueError("source330 geometry malformed")
 for i,row in enumerate(parent_tet_rows):
  if int(row["material_index"])!=int(parent[i]["material_index"]) or int(row["donor_body"])!=int(parent[i]["geometric_donor_body"]):raise ValueError(f"source330 geometry/label identity mismatch at row {i}")
 for keyname in ("points_current_world_m_f32","points_reference_world_m_f64","volume_tetrahedra","material_index","geometric_donor_body","anatomical_region_id","outside_source_tet_rows"):
  if keyname not in g:raise ValueError(f"full-splice geometry missing {keyname}")
 p,ref,nodefields,base_node_rows,protected_base,protected_out,protected_kind,protected_local=check_nodes(base,g,side)
 source_node_rows=np.asarray(nodefields["refinement_node_source_rows"],dtype=np.int64)
 parent_volume_rows=np.asarray(nodefields["parent_volume_node_row"],dtype=np.int64)
 if np.any(source_node_rows>=len(parent_node_rows)) or np.any(parent_volume_rows>=len(parent_node_rows)):
  raise ValueError("node source ancestry references a source330 node outside the pinned node domain")
 t=np.asarray(g["volume_tetrahedra"],dtype=np.int64);m=np.asarray(g["material_index"],dtype=np.int64)
 d=np.asarray(g["geometric_donor_body"],dtype=np.int64);r=np.asarray(g["anatomical_region_id"],dtype=np.int64)
 outside=np.asarray(g["outside_source_tet_rows"],dtype=np.int64)
 retet_source=np.asarray(g.get("retet_source_tet_rows",np.empty(0)),dtype=np.int64)
 if t.ndim!=2 or t.shape[1]!=4 or m.shape!=(len(t),) or d.shape!=(len(t),) or r.shape!=(len(t),):raise ValueError("full-splice cell arrays malformed")
 if np.any(t<0) or np.any(t>=len(p)) or np.any(t[:,0]==t[:,1]):raise ValueError("full-splice tet node indices invalid")
 if len(np.unique(outside))!=len(outside) or np.any(outside<0) or np.any(outside>=len(base["volume_tetrahedra"])):raise ValueError("outside source cell rows invalid")
 if len(np.unique(retet_source))!=len(retet_source) or np.any(retet_source<0) or np.any(retet_source>=len(base["volume_tetrahedra"])):
  raise ValueError("selected retet source cell rows invalid")
 if set(map(int,outside)) & set(map(int,retet_source)) or set(map(int,outside)) | set(map(int,retet_source)) != set(range(len(base["volume_tetrahedra"]))):
  raise ValueError("outside and retet source rows are not an exact partition of base cells")
 mapped_base_tets=base_node_rows[np.asarray(base["volume_tetrahedra"],dtype=np.int64)[outside]]
 if np.any(mapped_base_tets<0):raise ValueError("outside cell uses a retired base node")
 if not np.array_equal(t[:len(outside)],mapped_base_tets):raise ValueError("outside cell coordinates/indices are not exactly retained")
 if not np.array_equal(m[:len(outside)],base["material_index"][outside]) or not np.array_equal(d[:len(outside)],base["geometric_donor_body"][outside]) or not np.array_equal(r[:len(outside)],base["anatomical_region_id"][outside]):
  raise ValueError("outside material/donor/region labels changed")
 # Every retired old node must have a nonempty base star wholly in the retet selection.
 old_t=np.asarray(base["volume_tetrahedra"],dtype=np.int64)
 retired=np.flatnonzero(base_node_rows<0)
 selected_set=set(map(int,retet_source));outside_set=set(map(int,outside))
 for oldrow in retired:
  star={ci for ci,tet in enumerate(old_t) if int(oldrow) in tet}
  if not star or not star.issubset(selected_set) or star & outside_set:
   raise ValueError(f"retired base node {oldrow} is not selected-only")


 # Canonical protected-node routing is explicit. Outside-star identities cannot move;
 # selected-only descendants are separately typed and are not claimed as one-hot source nodes.
 base_star=[set() for _ in range(len(base["points_current_world_m_f32"]))]
 for ci,tet in enumerate(old_t):
  for node in tet:base_star[int(node)].add(ci)
 for old,target,mode in zip(protected_base,protected_out,protected_kind):
  outside_star=base_star[int(old)] & outside_set
  if outside_star and mode!="retained_original_node":
   raise ValueError(f"protected base row {old} touches unchanged outside cells and cannot be retargeted")
  if not np.any(t==int(target)):
   raise ValueError(f"protected output target {target} has no incident child cell")


 byall={int(x["child_tet_row"]):x for x in maprows}
 newrows=np.arange(len(outside),len(t),dtype=np.int64)
 if len(byall)!=len(maprows):raise ValueError("duplicate child_tet_row in 217 source map")
 newset=set(map(int,newrows))
 if set(byall) not in (set(range(len(t))),newset):
  raise ValueError("217 source map must cover every full child row or exactly the changed row suffix")
 bychild={row:byall[row] for row in newset}
 oldvol=volume(base["points_current_world_m_f32"].astype(np.float64),base["volume_tetrahedra"])
 newvol=volume(p,t)
 if np.any(oldvol<=0) or np.any(newvol<=0):raise ValueError("nonpositive source or spliced F32 tet")
 newsource=np.empty(len(newrows),dtype=np.int64);newface=np.empty(len(newrows),dtype=np.int64);resids=[]
 for j,cr in enumerate(newrows):
  x=bychild[int(cr)];sr=int(x["selected_parent_tet_row"])
  if sr not in parent:raise ValueError(f"source330 parent row {sr} out of domain")
  label=parent[sr]
  lab=(int(label["material_index"]),int(label["geometric_donor_body"]),int(label["anatomical_region_id"]))
  childlab=(int(m[cr]),int(d[cr]),int(r[cr]))
  maplab=(int(x["material_index"]),int(x["geometric_donor_body"]),int(x["anatomical_region_id"]))
  if lab!=childlab or maplab!=childlab:
   raise ValueError(f"source330 semantic registration mismatch at changed child {cr}")
  if int(x["stable_source_face_id"])!=int(label["stable_source_face_id"]):
   raise ValueError(f"source330 stable-face registration mismatch at changed child {cr}")
  beta=np.asarray([float(x[f"centroid_barycentric{k}"]) for k in range(4)],dtype=np.float64)
  if not np.isfinite(beta).all() or beta.min() < -1e-8 or beta.max()>1+1e-8 or abs(float(beta.sum())-1)>1e-8:
   raise ValueError(f"source330 centroid barycentric invalid at child {cr}")
  centroid=p[t[cr]].astype(np.float64).mean(axis=0)
  registered=beta @ source_points[source_tets[sr]]
  claimed=np.asarray([float(x[f"centroid_registered_reference_{axis}_m"]) for axis in "xyz"],dtype=np.float64)
  declared_resid=np.asarray([float(x[f"centroid_reference_residual_{axis}_m"]) for axis in "xyz"],dtype=np.float64)
  if not np.isfinite(claimed).all() or np.linalg.norm(registered-claimed)>1e-10:
   raise ValueError(f"source330 registered endpoint does not reconstruct at child {cr}")
  measured_vector=centroid-registered
  measured=float(np.linalg.norm(measured_vector))
  declared=float(x["centroid_reference_residual_norm_m"])
  if not np.isfinite(declared_resid).all() or np.linalg.norm(measured_vector-declared_resid)>1e-10:
   raise ValueError(f"source330 residual vector does not reconstruct at child {cr}")
  if not math.isfinite(declared) or declared<0 or abs(measured-declared)>1e-10:
   raise ValueError(f"source330 historical centroid residual does not match its authored vector at child {cr}")
  kind=x["label_match_kind"]
  if kind not in ("exact_f32_parent_tet_geometry","parent_centroid_containment",
                  "bounded_closest_parent_tet_reference_registration"):
   raise ValueError(f"unsupported source330 registration kind at child {cr}: {kind}")
  candidate_count=int(x["matching_parent_tet_count"]);tie_count=int(x["semantic_tie_count"])
  if candidate_count<1 or tie_count<0 or tie_count!=candidate_count-1:
   raise ValueError(f"source330 parent candidate/tie counts invalid at child {cr}")
  ambiguity=x["reference_registration_ambiguity_tolerance_m"]
  if ambiguity not in ("",None) and (not math.isfinite(float(ambiguity)) or float(ambiguity)<0):
   raise ValueError(f"source330 ambiguity tolerance invalid at child {cr}")
  declared_max=x["reference_registration_max_distance_m"]
  if kind=="bounded_closest_parent_tet_reference_registration":
   if declared_max in ("",None) or not math.isfinite(float(declared_max)) or measured>float(declared_max)+1e-10:
    raise ValueError(f"source map contradicts its own historical registration limit at child {cr}")
  elif declared_max not in ("",None) and (not math.isfinite(float(declared_max)) or float(declared_max)<0):
   raise ValueError(f"source map historical registration limit is invalid at child {cr}")
  declared_distance=x["reference_registration_distance_m"]
  if declared_distance not in ("",None) and abs(float(declared_distance)-measured)>1e-10:
   raise ValueError(f"source map registration distance differs at child {cr}")
  resids.append(measured)
  newsource[j]=sr;newface[j]=int(label["stable_source_face_id"])
 src=np.empty(len(t),dtype=np.int64);sf=np.empty(len(t),dtype=np.int64);kind=["retained_current006_child_map"]*len(t)
 src[:len(outside)]=np.asarray(base["child_parent_tet_row"],dtype=np.int64)[outside]
 sf[:len(outside)]=np.asarray(base["stable_source_face_id"],dtype=np.int64)[outside]
 src[newrows]=newsource;sf[newrows]=newface
 for row in newrows:kind[int(row)]="retet_direct_source330_registered"
 for row in range(len(outside)):
  if src[row] not in parent:raise ValueError(f"retained base child source row {src[row]} invalid")
  l=parent[int(src[row])]
  if (int(l["material_index"]),int(l["geometric_donor_body"]),int(l["anatomical_region_id"]))!=(int(m[row]),int(d[row]),int(r[row])):
   raise ValueError(f"retained outside child semantics mismatch at row {row}")
  if int(l["stable_source_face_id"])!=int(sf[row]):raise ValueError(f"retained outside stable source-face identity mismatch at row {row}")
 # Role surfaces come from the assembled candidate. Any retired face vertices require
 # explicit replacement role rows; base rows are remapped only when all three survive.
 roles={};changed=[]
 for role in ("O","I","D"):
  nm=f"coarse_boundary_faces_{role}"
  if nm not in base: continue
  oldfaces=np.asarray(base[nm],dtype=np.int64)
  old_valid=np.all(base_node_rows[oldfaces]>=0,axis=1)
  mapped_old=base_node_rows[oldfaces[old_valid]]
  if nm in bside:faces=np.asarray(bside[nm],dtype=np.int64)
  elif nm in g:faces=np.asarray(g[nm],dtype=np.int64)
  elif np.all(old_valid):faces=mapped_old.copy()
  else:raise ValueError(f"retired base nodes touch {role} faces without explicit replacement")
  roles[role]=faces
  if not np.all(old_valid) or not np.array_equal(faces,mapped_old):changed.append(role)
 if changed:
  for role in changed:
   for nm in (f"coarse_to_fine_face_{role}",f"coarse_to_fine_bary_{role}"):
    if nm not in bside and nm not in g:raise ValueError(f"changed {role} surface lacks explicit fine map {nm}")
 inc=incidence(t,m)
 other={};rolecounts={}
 node_map_rows=parse_operational_node_map(a.combined_node_map_csv)
 pinned_source=None
 if len(protected_base):
  pinned_source=np.full(len(p),-1,dtype=np.int64)
  sourceids=np.asarray(base.get("parent_volume_node_row",np.full(len(base["points_current_world_m_f32"]),-1)),dtype=np.int64)
  if sourceids.shape!=(len(base["points_current_world_m_f32"]),):raise ValueError("base source-parent node rows malformed")
  bynode={int(row["joint_node_id"])-1:row for row in node_map_rows}
  for old,target,mode in zip(protected_base,protected_out,protected_kind):
   row=bynode.get(int(target))
   if row is None:raise ValueError(f"protected output target {target} lacks combined node-map row")
   if mode=="retained_original_node":
    sourceid=int(sourceids[int(old)])
    if sourceid>=0:
     if int(row["source_parent_node_row"])!=sourceid or int(nodefields["parent_volume_node_row"][int(target)])!=sourceid:
      raise ValueError(f"retained protected row {old} lost exact source-node identity")
     pinned_source[int(target)]=sourceid
   elif mode=="explicit_retet_descendant":
    if int(row["source_parent_node_row"])!=-1:
     raise ValueError(f"retargeted protected row {old} falsely claims one-hot source-parent identity")
   else:raise ValueError("protected mapping kind unsupported")
 node_map_report=validate_combined_node_map(node_map_rows,p,ref,t,m,d,r,source_points,source_tets,
  [parent[i] for i in range(len(parent))],a.maximum_operational_reference_distance_m,
  role_faces={k:v for k,v in roles.items()},pinned_source_node_rows=pinned_source)


 for role,faces in roles.items():
  if faces.ndim!=2 or faces.shape[1]!=3:raise ValueError(f"role {role} faces malformed")
  rr=np.full(len(faces),-1,dtype=np.int64)
  if len(np.unique(np.sort(faces,axis=1),axis=0))!=len(faces):raise ValueError(f"role {role} has duplicate face rows")
  for i,f in enumerate(faces):
   hits=inc.get(key(f),[])
   if role in ("O","D"):
    if len(hits)!=1 or cyc(f)!=cyc(hits[0][1]):raise ValueError(f"{role} face {i} is not an oriented external face")
   else:
    if len(hits)!=2 or {int(m[h[0]]) for h in hits}!={15,16}:raise ValueError(f"I face {i} is not the skin/fat shared interface")
    if cyc(f) not in {cyc(hits[0][1]),cyc(hits[1][1])}:raise ValueError(f"I face {i} winding mismatch")
    rr[i]=next(h[0] for h in hits if int(m[h[0]])==16)
  other[role]=rr;rolecounts[role]=len(faces)
 # The selected region/outside cut collar must stay the same triangle-by-triangle.
 selected=set(range(len(outside),len(t)));oldinc=incidence(base["volume_tetrahedra"],base["material_index"])
 cuts=[]
 outside_set=set(map(int,outside))
 for k,h in oldinc.items():
  if len(h)==2 and ((h[0][0] in outside_set) != (h[1][0] in outside_set)):cuts.append(k)
 missing=[]
 for k in cuts:
  mapped=key(base_node_rows[np.asarray(k,dtype=np.int64)])
  now=inc.get(mapped,[])
  if len(now)!=2 or cyc(now[0][1])==cyc(now[1][1]):missing.append(k)
 if missing:raise ValueError(f"retained regional cut collar not preserved: {len(missing)} faces")
 # Existing sidecars are copied only for unchanged node rows; for added rows, caller must
 # provide full mapped arrays above. This stage does not make mass fractions.
 outdir=a.output_dir
 if outdir.exists(): raise FileExistsError(f"output directory already exists: {outdir}")
 if a.receipt_json.exists(): raise FileExistsError(f"receipt already exists: {a.receipt_json}")
 outdir.mkdir(parents=True,exist_ok=False)
 node=outdir/"mesh.node";ele=outdir/"mesh.ele";labels=outdir/"child.labels.csv";cellmap=outdir/"child-source330-map.csv"
 source_registration=outdir/"source330-registered-child-centroids.csv"
 combined_map=outdir/"combined-node-map.csv"
 shutil.copyfile(a.cell_ancestry_csv,source_registration)
 shutil.copyfile(a.combined_node_map_csv,combined_map)
 write_node(node,p);write_ele(ele,t,m);write_labels(labels,m,d,r,sf);write_map(cellmap,src,m,d,r,sf,kind)
 # Optional parity against the already imported identity-control child files.
 comparison=None
 if a.compare_cli_dir:
  nd=parse_node(a.compare_cli_dir/"mesh.node");ct,cm=parse_ele(a.compare_cli_dir/"mesh.ele")
  _,cl=readcsv(a.compare_cli_dir/"child.labels.csv")
  lab=np.asarray([[int(x[k]) for k in ("child_tet_row","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id")] for x in cl],dtype=np.int64)
  expected=np.column_stack((np.arange(len(t)),m,d,r,sf))
  comparison={"nodes_exact_f32":bool(np.array_equal(nd,p)),"tetrahedra_exact":bool(np.array_equal(ct,t)),
              "materials_exact":bool(np.array_equal(cm,m)),"labels_exact":bool(np.array_equal(lab,expected)),
              "all_exact":bool(np.array_equal(nd,p) and np.array_equal(ct,t) and np.array_equal(cm,m) and np.array_equal(lab,expected))}
 report={"schema":"regional-splice-native-child-inputs-direct-source330.v1","status":"INPUTS_PREPARED_DIRECT_SOURCE330_REGISTRATION_SOURCE_ONLY",
  "physical_admission":False,"mass_fractions_emitted":False,"native_or_GPU_run":False,
  "scope":"full-splice geometry with direct source330 registered child centroids packaged for the native mass-transfer owner; no current006 parent containment claim, mass result, or physical qualification",
  "inputs":{"base":{"path":str(a.base_npz),"sha256":sha(a.base_npz)},"geometry":{"path":str(a.geometry_npz),"sha256":sha(a.geometry_npz)},
            "cell_ancestry":{"path":str(a.cell_ancestry_csv),"sha256":sha(a.cell_ancestry_csv)},"source330_labels":{"path":str(a.parent_labels_csv),"sha256":sha(a.parent_labels_csv)},
            "source330_nodes":{"path":str(a.parent_nodes_csv),"sha256":sha(a.parent_nodes_csv)},
            "source330_tets":{"path":str(a.parent_tets_csv),"sha256":sha(a.parent_tets_csv)},
            "node_sidecars_sha256":sha(a.node_sidecars_npz) if a.node_sidecars_npz else None,"boundary_sidecars_sha256":sha(a.boundary_sidecars_npz) if a.boundary_sidecars_npz else None,
            "combined_node_map":{"path":str(a.combined_node_map_csv),"sha256":sha(a.combined_node_map_csv)}},
  "counts":{"nodes":len(p),"cells":len(t),"retained_outside_cells":len(outside),"changed_regional_cells":len(newrows),
            "changed_label_tuples":sorted({(int(m[i]),int(d[i]),int(r[i])) for i in newrows}),
             "protected149_rows":len(protected_base),"protected_rows_retargeted":int(np.sum(protected_kind!="retained_original_node")),
            "retained_base_node_rows":int(np.sum(base_node_rows>=0)),"retired_base_node_rows":int(np.sum(base_node_rows<0)),
            "cut_collar_faces_checked":len(cuts),"cut_collar_faces_preserved":len(cuts),
            "changed_child_registration_rows":len(resids),"current006_parent_centroid_checks":0,"direct_source330_centroid_registration_checks":len(newrows),"max_historical_source330_centroid_residual_m":float(max(resids)),
             "max_operational_reference_distance_m":node_map_report["max_operational_residual_m"],
            "retired_base_node_row_ids":retired.astype(int).tolist(),
            "protected_targets":[{"base_node_row":int(x),"output_node_row":int(y),"mapping_kind":str(k)}
                for x,y,k in zip(protected_base,protected_out,protected_kind)]},
  "role_surfaces":{"counts":rolecounts,"changed_roles":changed,"I_material16_other_parent_rows_recomputed":len(other.get("I", []))},
  "outputs":{str(x.name):{"path":str(x),"sha256":sha(x)} for x in (node,ele,labels,cellmap,source_registration,combined_map)},
  "combined_node_map_validation":node_map_report,
  "identity_cli_comparison":comparison,"mass_measure":"not generated here; source330 cell registration supplies semantic/provenance linkage only; use existing conservative spatial mass-transfer owner",
  "source_cell_map":{"retained_cell_parents":len(outside),"changed_cell_parents":len(newrows),"source_measure_fractions_not_written":True}}
 a.receipt_json.parent.mkdir(parents=True,exist_ok=True);a.receipt_json.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps(report,sort_keys=True,separators=(",",":")))
if __name__=="__main__":main()
