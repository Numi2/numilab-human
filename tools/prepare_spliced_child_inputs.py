#!/usr/bin/env python3
"""Prepare existing native child-mesh inputs from a full regional-splice geometry packet.

Consumes the existing full-splice NPZ and retet-cell-parent-ancestry CSV. It does
not assign mass fractions or claim physical admission; the native conservative
spatial mass-transfer owner computes mass. Changed node domains require explicit
existing node sidecars, and changed role boundaries require explicit O/I/D maps.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np

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
 if p.ndim!=2 or p.shape[1]!=3 or ref.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(ref).all():raise ValueError("geometry point fields invalid")
 n0=len(base["points_current_world_m_f32"])
 if side is None:
  if len(p)!=n0 or not np.array_equal(p,base["points_current_world_m_f32"]) or not np.array_equal(ref,base["points_reference_world_m_f64"]):
   raise ValueError("changed/new nodes require explicit existing node sidecars")
  fields={k:base[k].copy() for k in NODE_FIELDS}
  base_rows=np.arange(n0,dtype=np.int64)
 else:
  required=set(NODE_FIELDS)|{"base_node_output_rows"}
  if required-set(side):raise ValueError(f"node sidecars missing {sorted(required-set(side))}")
  base_rows=np.asarray(side["base_node_output_rows"],dtype=np.int64)
  if base_rows.shape!=(n0,) or np.any(base_rows<0) or np.any(base_rows>=len(p)) or len(np.unique(base_rows))!=n0:
   raise ValueError("base node output map invalid")
  if not np.array_equal(p[base_rows],base["points_current_world_m_f32"]):raise ValueError("retained original nodes moved")
  fields={k:np.asarray(side[k]).copy() for k in NODE_FIELDS}
  if fields["points_donor_local_m_f64"].shape!=(len(p),3) or fields["target_current_world_m"].shape!=(len(p),3):
   raise ValueError("node coordinate sidecar row domain mismatch")
  if not np.isfinite(fields["target_current_world_m"]).all():raise ValueError("node target coordinates nonfinite")
  if fields["parent_volume_node_row"].shape!=(len(p),) or fields["full86_weights"].shape!=(len(p),86):
   raise ValueError("parent/full86 node sidecar row domain mismatch")
  offsets=np.asarray(fields["refinement_node_source_offsets"],dtype=np.int64)
  src=np.asarray(fields["refinement_node_source_rows"],dtype=np.int64)
  weights=np.asarray(fields["refinement_node_source_weights"],dtype=np.float64)
  if offsets.shape!=(len(p)+1,) or offsets[0]!=0 or offsets[-1]!=len(src) or len(src)!=len(weights) or np.any(np.diff(offsets)<0) or np.any(src<0) or not np.isfinite(weights).all():
   raise ValueError("node source CSR invalid")
  if np.any(fields["parent_volume_node_row"] < -1) or not np.isfinite(fields["full86_weights"]).all() or not np.isfinite(fields["points_donor_local_m_f64"]).all():
   raise ValueError("node sidecars contain invalid values")
  if np.any(fields["full86_weights"] < -1e-10) or np.max(np.abs(fields["full86_weights"].sum(axis=1)-1))>1e-5:
   raise ValueError("full86 node weights invalid")
  # Existing-node sidecars are retained exactly under the explicit row map.
  for k in ("points_donor_local_m_f64","parent_volume_node_row","full86_weights","target_current_world_m"):
   if not np.array_equal(fields[k][base_rows],base[k]):raise ValueError(f"existing node sidecar changed: {k}")
  oldoff=np.asarray(base["refinement_node_source_offsets"],dtype=np.int64)
  oldsrc=np.asarray(base["refinement_node_source_rows"],dtype=np.int64)
  oldw=np.asarray(base["refinement_node_source_weights"],dtype=np.float64)
  for oldrow,newrow in enumerate(base_rows):
   a,b=oldoff[oldrow:oldrow+2];c,d=offsets[newrow:newrow+2]
   if not np.array_equal(src[c:d],oldsrc[a:b]) or not np.array_equal(weights[c:d],oldw[a:b]):
    raise ValueError(f"existing node source CSR changed at base row {oldrow}")
 pins=np.asarray(base["pinned_volume_node_rows"],dtype=np.int64)
 if pins.shape!=(149,) or np.any(pins<0) or np.any(pins>=n0) or not np.array_equal(p[pins if side is None else base_rows[pins]],base["points_current_world_m_f32"][pins]):
  raise ValueError("protected149 node coordinates/rows changed")
 if not np.array_equal(p.astype(np.float32),ref.astype(np.float32)):raise ValueError("current/reference F32 geometry differs")
 offsets=np.asarray(fields["refinement_node_source_offsets"],dtype=np.int64)
 src=np.asarray(fields["refinement_node_source_rows"],dtype=np.int64)
 weights=np.asarray(fields["refinement_node_source_weights"],dtype=np.float64)
 max_parent_node=int(np.max(base["refinement_node_source_rows"])) if len(base["refinement_node_source_rows"]) else -1
 if offsets.shape!=(len(p)+1,) or offsets[0]!=0 or offsets[-1]!=len(src) or len(src)!=len(weights) or np.any(np.diff(offsets)<1):
  raise ValueError("node source CSR has empty or invalid row domain")
 if np.any(src<0) or np.any(src>max_parent_node) or not np.isfinite(weights).all() or np.any(weights<0):
  raise ValueError("node source CSR rows/weights outside original330 node domain")
 rowsums=np.asarray([weights[offsets[i]:offsets[i+1]].sum() for i in range(len(p))])
 if np.max(np.abs(rowsums-1.0))>1e-8:raise ValueError("node source CSR weights do not sum to one")
 return p,ref,fields,base_rows

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--base-npz",type=Path,required=True);ap.add_argument("--geometry-npz",type=Path,required=True)
 ap.add_argument("--cell-ancestry-csv",type=Path,required=True);ap.add_argument("--parent-labels-csv",type=Path,required=True)
 ap.add_argument("--parent-nodes-csv",type=Path,required=True);ap.add_argument("--parent-tets-csv",type=Path,required=True)
 ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--node-sidecars-npz",type=Path)
 ap.add_argument("--boundary-sidecars-npz",type=Path);ap.add_argument("--compare-cli-dir",type=Path)
 ap.add_argument("--maximum-registration-distance-m",type=float,default=.0004);ap.add_argument("--receipt-json",type=Path,required=True)
 a=ap.parse_args()
 if not math.isfinite(a.maximum_registration_distance_m) or a.maximum_registration_distance_m<=0:raise ValueError("registration bound invalid")
 base=readnpz(a.base_npz);g=readnpz(a.geometry_npz);side=readnpz(a.node_sidecars_npz) if a.node_sidecars_npz else None
 bside=readnpz(a.boundary_sidecars_npz) if a.boundary_sidecars_npz else {}
 fields,maprows=readcsv(a.cell_ancestry_csv);pfields,parentrows=readcsv(a.parent_labels_csv)
 required={"current006_centroid_beta0","current006_centroid_beta1","current006_centroid_beta2","current006_centroid_beta3","registered_x_m","registered_y_m","registered_z_m","spliced_child_tet_row","material_index","current006_parent_tet_row","source330_parent_tet_row","source330_registration_beta0","source330_registration_beta1","source330_registration_beta2","source330_registration_beta3","residual_norm_m","child_volume_m3"}
 if not required.issubset(fields):raise ValueError(f"cell ancestry missing required existing fields {sorted(required-set(fields))}")
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
 p,ref,nodefields,base_node_rows=check_nodes(base,g,side)
 t=np.asarray(g["volume_tetrahedra"],dtype=np.int64);m=np.asarray(g["material_index"],dtype=np.int64)
 d=np.asarray(g["geometric_donor_body"],dtype=np.int64);r=np.asarray(g["anatomical_region_id"],dtype=np.int64)
 outside=np.asarray(g["outside_source_tet_rows"],dtype=np.int64)
 if t.ndim!=2 or t.shape[1]!=4 or m.shape!=(len(t),) or d.shape!=(len(t),) or r.shape!=(len(t),):raise ValueError("full-splice cell arrays malformed")
 if np.any(t<0) or np.any(t>=len(p)) or np.any(t[:,0]==t[:,1]):raise ValueError("full-splice tet node indices invalid")
 if len(np.unique(outside))!=len(outside) or np.any(outside<0) or np.any(outside>=len(base["volume_tetrahedra"])):raise ValueError("outside source cell rows invalid")
 if len(t)!=len(outside)+len(maprows):raise ValueError("regional cell ancestry count does not complement retained outside cells")
 remapped_base_tets=np.asarray(base["volume_tetrahedra"],dtype=np.int64)[outside].copy()
 remapped_base_tets=base_node_rows[remapped_base_tets]
 if not np.array_equal(t[:len(outside)],remapped_base_tets):raise ValueError("outside cell coordinates/indices are not exactly retained")
 if not np.array_equal(m[:len(outside)],base["material_index"][outside]) or not np.array_equal(d[:len(outside)],base["geometric_donor_body"][outside]) or not np.array_equal(r[:len(outside)],base["anatomical_region_id"][outside]):
  raise ValueError("outside material/donor/region labels changed")
 bychild={int(x["spliced_child_tet_row"]):x for x in maprows}
 newrows=np.arange(len(outside),len(t),dtype=np.int64)
 if len(bychild)!=len(maprows) or set(bychild)!=set(map(int,newrows)):raise ValueError("ancestry rows do not bijectively cover changed cells")
 oldvol=volume(base["points_current_world_m_f32"].astype(np.float64),base["volume_tetrahedra"])
 newvol=volume(p,t)
 if np.any(oldvol<=0) or np.any(newvol<=0):raise ValueError("nonpositive source or spliced F32 tet")
 newsource=np.empty(len(newrows),dtype=np.int64);newface=np.empty(len(newrows),dtype=np.int64);resids=[];current006_parent_checks=0
 base_tets=np.asarray(base["volume_tetrahedra"],dtype=np.int64);base_m=np.asarray(base["material_index"],dtype=np.int64);base_d=np.asarray(base["geometric_donor_body"],dtype=np.int64);base_r=np.asarray(base["anatomical_region_id"],dtype=np.int64)
 base_points=np.asarray(base["points_current_world_m_f32"],dtype=np.float32).astype(np.float64)
 for j,cr in enumerate(newrows):
  x=bychild[int(cr)];sr=int(x["source330_parent_tet_row"])
  if sr not in parent:raise ValueError(f"source330 parent row {sr} out of domain")
  label=parent[sr];lab=(int(label["material_index"]),int(label["geometric_donor_body"]),int(label["anatomical_region_id"]))
  if lab!=(int(m[cr]),int(d[cr]),int(r[cr])) or int(x["material_index"])!=int(m[cr]):
   raise ValueError(f"source330 ancestry semantic mismatch at changed child {cr}")
  oldrow=int(x["current006_parent_tet_row"])
  if oldrow<0 or oldrow>=len(base_tets) or (int(base_m[oldrow]),int(base_d[oldrow]),int(base_r[oldrow]))!=(int(m[cr]),int(d[cr]),int(r[cr])):raise ValueError(f"current006 ancestry semantic mismatch at child {cr}")
  if sr!=int(base["child_parent_tet_row"][oldrow]) or sr!=int(base["refinement_child_source_tet_rows"][oldrow]):raise ValueError(f"source330 ancestry is not the retained current006 composition at child {cr}")
  oldbeta=np.asarray([float(x[f"current006_centroid_beta{k}"]) for k in range(4)],dtype=np.float64)
  if not np.isfinite(oldbeta).all() or oldbeta.min() < -1e-8 or oldbeta.max()>1+1e-8 or abs(float(oldbeta.sum())-1)>1e-8:raise ValueError(f"current006 centroid barycentric invalid at child {cr}")
  centroid=p[t[cr]].astype(np.float64).mean(axis=0)
  registered=oldbeta @ base_points[base_tets[oldrow]]
  if np.linalg.norm(centroid-registered)>1e-8:raise ValueError(f"current006 ancestry endpoint does not reconstruct child centroid at row {cr}")
  current006_parent_checks+=1
  beta=np.asarray([float(x[f"source330_registration_beta{k}"]) for k in range(4)],dtype=np.float64)
  if not np.isfinite(beta).all() or beta.min() < -1e-8 or beta.max()>1+1e-8 or abs(float(beta.sum())-1)>1e-8:raise ValueError(f"source registration barycentric invalid at child {cr}")
  resid=float(x["residual_norm_m"]);resids.append(resid)
  if not math.isfinite(resid) or resid<0 or resid>a.maximum_registration_distance_m:raise ValueError(f"source registration residual outside bound at child {cr}")
  source_registered=beta @ source_points[source_tets[sr]]
  claimed_registered=np.asarray([float(x[k]) for k in ("registered_x_m","registered_y_m","registered_z_m")],dtype=np.float64)
  measured_resid=float(np.linalg.norm(centroid-source_registered))
  if not np.isfinite(claimed_registered).all() or np.linalg.norm(source_registered-claimed_registered)>1e-10:raise ValueError(f"source330 barycentric endpoint does not reconstruct registered point at child {cr}")
  if abs(measured_resid-resid)>1e-10 or measured_resid>a.maximum_registration_distance_m:raise ValueError(f"source330 actual registration residual does not match receipt or bound at child {cr}")
  cv=float(x["child_volume_m3"])
  if not math.isfinite(cv) or abs(cv-newvol[cr])/max(cv,newvol[cr])>1e-7:raise ValueError(f"cell ancestry volume mismatch at child {cr}")
  newsource[j]=sr;newface[j]=int(label["stable_source_face_id"])
 src=np.empty(len(t),dtype=np.int64);sf=np.empty(len(t),dtype=np.int64);kind=["retained_current006_child_map"]*len(t)
 src[:len(outside)]=np.asarray(base["child_parent_tet_row"],dtype=np.int64)[outside]
 sf[:len(outside)]=np.asarray(base["stable_source_face_id"],dtype=np.int64)[outside]
 src[newrows]=newsource;sf[newrows]=newface
 for row in newrows:kind[int(row)]="retet_current006_to_source330_composition"
 for row in range(len(outside)):
  if src[row] not in parent:raise ValueError(f"retained base child source row {src[row]} invalid")
  l=parent[int(src[row])]
  if (int(l["material_index"]),int(l["geometric_donor_body"]),int(l["anatomical_region_id"]))!=(int(m[row]),int(d[row]),int(r[row])):
   raise ValueError(f"retained outside child semantics mismatch at row {row}")
  if int(l["stable_source_face_id"])!=int(sf[row]):raise ValueError(f"retained outside stable source-face identity mismatch at row {row}")
 # Role surface arrays can be explicit from the new PLC; otherwise reuse the exact base
 # faces under the proven base-node map. Changed faces require matching fine mappings.
 roles={}
 changed=[]
 for role in ("O","I","D"):
  nm=f"coarse_boundary_faces_{role}"
  if nm in bside:faces=np.asarray(bside[nm],dtype=np.int64)
  elif nm in g:faces=np.asarray(g[nm],dtype=np.int64)
  else:faces=base[nm].copy() if np.array_equal(base_node_rows,np.arange(len(base_node_rows))) else base_node_rows[np.asarray(base[nm],dtype=np.int64)]
  roles[role]=faces
  basefaces=base[nm].copy() if np.array_equal(base_node_rows,np.arange(len(base_node_rows))) else base_node_rows[np.asarray(base[nm],dtype=np.int64)]
  if not np.array_equal(faces,basefaces):changed.append(role)
 if changed:
  for role in ("O","I","D"):
   for nm in (f"coarse_to_fine_face_{role}",f"coarse_to_fine_bary_{role}"):
    if nm not in bside and nm not in g:raise ValueError(f"changed O/I/D surfaces require explicit existing fine map {nm}")
 inc=incidence(t,m)
 other={};rolecounts={}
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
 outdir=a.output_dir;outdir.mkdir(parents=True,exist_ok=True)
 node=outdir/"mesh.node";ele=outdir/"mesh.ele";labels=outdir/"child.labels.csv";cellmap=outdir/"child-source330-map.csv"
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
 report={"schema":"regional-splice-native-child-inputs.v1","status":"INPUTS_PREPARED_SOURCE_ONLY",
  "physical_admission":False,"mass_fractions_emitted":False,"native_or_GPU_run":False,
  "scope":"generic full-splice geometry and existing source330 cell ancestry packaged for the native mass-transfer owner; no mass result or physical qualification",
  "inputs":{"base":{"path":str(a.base_npz),"sha256":sha(a.base_npz)},"geometry":{"path":str(a.geometry_npz),"sha256":sha(a.geometry_npz)},
            "cell_ancestry":{"path":str(a.cell_ancestry_csv),"sha256":sha(a.cell_ancestry_csv)},"source330_labels":{"path":str(a.parent_labels_csv),"sha256":sha(a.parent_labels_csv)},
            "source330_nodes":{"path":str(a.parent_nodes_csv),"sha256":sha(a.parent_nodes_csv)},
            "source330_tets":{"path":str(a.parent_tets_csv),"sha256":sha(a.parent_tets_csv)},
            "node_sidecars_sha256":sha(a.node_sidecars_npz) if a.node_sidecars_npz else None,"boundary_sidecars_sha256":sha(a.boundary_sidecars_npz) if a.boundary_sidecars_npz else None},
  "counts":{"nodes":len(p),"cells":len(t),"retained_outside_cells":len(outside),"changed_regional_cells":len(newrows),
            "changed_label_tuples":sorted({(int(m[i]),int(d[i]),int(r[i])) for i in newrows}),
            "protected149_rows":len(base["pinned_volume_node_rows"]),"retained_base_node_rows":len(base_node_rows),
            "cut_collar_faces_checked":len(cuts),"cut_collar_faces_preserved":len(cuts),
            "changed_child_registration_rows":len(resids),"current006_parent_centroid_checks":current006_parent_checks,"source330_endpoint_and_composition_checks":len(newrows),"max_registration_residual_m":float(max(resids))},
  "role_surfaces":{"counts":rolecounts,"changed_roles":changed,"I_material16_other_parent_rows_recomputed":len(other["I"])},
  "outputs":{str(x.name):{"path":str(x),"sha256":sha(x)} for x in (node,ele,labels,cellmap)},
  "identity_cli_comparison":comparison,"mass_measure":"not generated here; use existing conservative spatial mass-transfer owner",
  "source_cell_map":{"retained_cell_parents":len(outside),"changed_cell_parents":len(newrows),"source_measure_fractions_not_written":True}}
 a.receipt_json.parent.mkdir(parents=True,exist_ok=True);a.receipt_json.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps(report,sort_keys=True,separators=(",",":")))
if __name__=="__main__":main()
