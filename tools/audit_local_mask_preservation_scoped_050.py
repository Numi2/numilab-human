#!/usr/bin/env python3
"""Source-only preservation audit for existing local paired skin/fat replacement.
Uses the existing 427 material geometry owner and original330 source pins.
No volume mesh, mass reassignment or runtime admission is produced.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,time,shutil
from collections import Counter
import numpy as np
from scipy.spatial import ConvexHull
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
BASE=P/"reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz"
CAP=P/"native-physical-parent-mass-330/native-source-identity-control-003"
OWNER=P/"audit_coarse_material_bone_volume_427.py"
COMPILER=P/"d-boundary-retirement-501/export_refined_coarse_registered_volume_433_d_retirement.py"
BASESHA="888eb12450d0e69fb0ad3f5998159ab7a0a20d3e70b13c42ba416c61cbbede63"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def csvrows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def mesh_data(z,prefix=None):
 if prefix and prefix+"_points_world_m_f64" in z:
  return np.asarray(z[prefix+"_points_world_m_f64"],np.float64),np.asarray(z[prefix+"_faces"],np.int64)
 p=next(z[k] for k in ("points_world_m_f64","points_world_f32","points_world_m") if k in z)
 f=next(z[k] for k in ("faces_f64","faces") if k in z)
 return np.asarray(p,np.float64),np.asarray(f,np.int64)
def lattice_point(v):
 result=[]
 for x in np.asarray(v,np.float32):
  n,d=float(x).as_integer_ratio();result.append(n*((1<<149)//d))
 return tuple(result)
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def exact_point_on_triangle(point,triangle):
 a,b,c=map(lattice_point,triangle);p=lattice_point(point)
 n=cross(sub(b,a),sub(c,a))
 if n==(0,0,0) or dot(n,sub(p,a))!=0:return False
 return all(dot(n,cross(sub(y,x),sub(p,x)))>=0 for x,y in ((a,b),(b,c),(c,a)))
def protected_material_scope(base, protect, source_roles, child_for_parent, regional=False):
 """Select required protected points by the candidate's complete material domain.
 An explicit partial region cannot contain unrelated whole-body boundary points.
 This does not validate preservation in the retained outside cells.
 """
 t=np.asarray(base["volume_tetrahedra"],np.int64)
 m=np.asarray(base["material_index"],np.int64)
 d=np.asarray(base["geometric_donor_body"],np.int64)
 r=np.asarray(base["anatomical_region_id"],np.int64)
 if t.ndim!=2 or t.shape[1]!=4 or any(v.shape!=(len(t),) for v in (m,d,r)):
  raise ValueError("source regional cell domain is malformed")
 result={}
 for name,material,roles in (("skin",15,("O","I")),("fat",16,("I","D"))):
  selected=(m==material)&(d==23)&(r==23)
  if regional and not np.any(selected):raise ValueError("requested regional material has no source cells")
  used=set(map(int,np.unique(t[selected]))) if regional else None
  included=[];excluded=[]
  for node in sorted(protect):
   if source_roles[node][0] not in roles:continue
   if node not in child_for_parent:raise ValueError("protected source node has no baseline child")
   (included if used is None or child_for_parent[node] in used else excluded).append(int(node))
  result[name]={"required_source_nodes":included,"outside_regional_source_nodes":excluded,
                "source_cell_count":int(selected.sum()) if regional else None}
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--mask",required=True);ap.add_argument("--joint",required=True);ap.add_argument("--skin",required=True);ap.add_argument("--fat",required=True);ap.add_argument("--out",required=True)
 ap.add_argument("--regional-material-boundaries",action="store_true",help="candidate files contain only complete material15/16 donor23 region23 boundaries; check unrelated protected nodes after global splice")
 a=ap.parse_args();out=Path(a.out)
 if out.exists():raise ValueError("refusing existing output")
 for f in (BASE,CAP/"parent-layer-rows.csv",CAP/"compiled-parent-cell-mass.csv.labels.csv",CAP/"compiled-parent-cell-mass.csv.tets.csv",OWNER,COMPILER):
  if not f.exists():raise FileNotFoundError(f)
 if sha(BASE)!=BASESHA:raise ValueError("baseline pin changed")
 st=time.monotonic();out.mkdir(parents=True);sources=out/"authoring-source-chain";sources.mkdir()
 pinned_inputs=(BASE,Path(a.mask),Path(a.joint),Path(a.skin),Path(a.fat),OWNER,COMPILER,Path(__file__))
 input_hashes={str(f):sha(f) for f in pinned_inputs}
 for f in (OWNER,COMPILER,Path(__file__)):shutil.copy2(f,sources/f.name)
 owner=load_module("pinned_geometry_427",sources/OWNER.name);compiler=load_module("d_retirement_compiler",sources/COMPILER.name)
 b=arrays(BASE);p=np.asarray(b["points_current_world_m_f32"],np.float32).astype(np.float64);t=np.asarray(b["volume_tetrahedra"],np.int64)
 parentmap=np.asarray(b["parent_volume_node_row"],np.int64)
 source_tets=[tuple(int(r[f"parent_node{k}"]) for k in range(4)) for r in csvrows(CAP/"compiled-parent-cell-mass.csv.tets.csv")]
 labels=csvrows(CAP/"compiled-parent-cell-mass.csv.labels.csv")
 source_roles={int(r["volume_node_row"]):(r["role"],int(r["role_row"])) for r in csvrows(CAP/"parent-layer-rows.csv")}
 if len(source_roles)!=7233:raise ValueError("source330 role inventory mismatch")
 protect,summary=compiler.derive_protected_parent_nodes(source_tets,labels,source_roles,csvrows(CAP/"parent-deep-attachments.csv"),sha(CAP/"parent-deep-attachments.csv"))
 child_for_parent={int(n):i for i,n in enumerate(parentmap) if n>=0}
 if any(n not in child_for_parent for n in protect):raise ValueError("baseline protected source node missing")
 maskp,maskf=mesh_data(arrays(a.mask),"mask");jp,jf=mesh_data(arrays(a.joint),"joint");mask=owner.solid(maskp,maskf)
 if mask.status()!=owner.mf.Error.NoError:raise ValueError("mask is not a closed valid solid")
 hull=ConvexHull(maskp);normal=hull.equations[:,:3];offset=hull.equations[:,3]
 # Broad phase only: exact closed-solid intersections below resolve every surviving cell.
 tp=p[t];blo=tp.min(1);bhi=tp.max(1);lo=maskp.min(0);hi=maskp.max(0)
 active=np.flatnonzero(np.all(bhi>=lo,axis=1)&np.all(blo<=hi,axis=1))
 hit=[];zero=[];invalid=[]
 tetfaces=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
 for ci in active:
  x=tp[ci]
  if np.any(np.all(x@normal.T+offset>0,axis=0)):continue
  tet=owner.solid(x,tetfaces);z=mask^tet
  if z.status()!=owner.mf.Error.NoError:invalid.append({"tet":int(ci),"status":str(z.status())});continue
  v=float(z.volume())
  row={"tet":int(ci),"material":int(b["material_index"][ci]),"donor":int(b["geometric_donor_body"][ci]),"region":int(b["anatomical_region_id"][ci]),"intersection_m3":v}
  if v>0:hit.append(row)
  elif v<0:invalid.append(row)
  else:zero.append(int(ci))
 counts=Counter((r["material"],r["donor"],r["region"]) for r in hit)
 support_other=[r for r in hit if (r["donor"],r["region"])!=(23,23)]
 scope=protected_material_scope(b,protect,source_roles,child_for_parent,a.regional_material_boundaries)
 protected=[]
 for role,path in (("skin",a.skin),("fat",a.fat)):
  pp,ff=mesh_data(arrays(path));used=np.unique(ff)
  face_points=pp.astype(np.float32)[ff];face_min=face_points.min(1);face_max=face_points.max(1)
  bits={tuple(v) for v in pp[used].astype(np.float32).view(np.uint32)}
  relevant=[]
  for n in scope[role]["required_source_nodes"]:
   ci=child_for_parent[n];xyz=p[ci].astype(np.float32);same=tuple(xyz.view(np.uint32)) in bits
   candidates=np.flatnonzero(np.all(face_min<=xyz,axis=1)&np.all(face_max>=xyz,axis=1)) if not same else []
   on_boundary=same or any(exact_point_on_triangle(xyz,face_points[fi]) for fi in candidates)
   relevant.append({"source_node":n,"current006_node":ci,"unchanged_vertex_present":same,"role":source_roles[n][0],"exact_f32_boundary_point_present":bool(on_boundary),"source_xyz_f32":xyz.astype(float).tolist(),"mask_max_plane_distance_m":float(np.max(normal@xyz.astype(float)+offset))})
  protected.append({"material":role,"rows":relevant,"missing":sum(not r["unchanged_vertex_present"] for r in relevant),"missing_exact_boundary_points":sum(not r["exact_f32_boundary_point_present"] for r in relevant)})
 if any(sha(f)!=input_hashes[str(f)] for f in pinned_inputs):raise ValueError("input changed during preservation audit")
 report={"status":"SOURCE_ONLY_LOCAL_MASK_PRESERVATION_DIAGNOSTIC","runtime_created":0,"accepted_physical_steps":0,"input_sha256":input_hashes,"protected_source_summary":summary,"candidate_protected_scope":{"mode":"material15_16_donor23_region23" if a.regional_material_boundaries else "whole_body","materials":scope,"outside_global_preservation":"NOT_ESTABLISHED_BY_PARTIAL_CANDIDATE"},"protected_vertex_inventory":protected,"mask_group_hits":[{"material":k[0],"donor":k[1],"region":k[2],"cells":v} for k,v in sorted(counts.items())],"other_group_cell_witnesses":support_other,"Boolean_invalid":invalid,"group23_only_positive_volume_support":not support_other and not invalid,"positive_volume_mask_cells":hit,"elapsed_s":time.monotonic()-st,"limits":["Mask intersections use the existing 427 Boolean owner; zero volumes do not supersede exact emitted-surface tests.","Vertex presence alone does not establish a valid attachment, unchanged outside-cell topology or physical admission.","No regional capacity can be assigned from global volume deltas unless group support and final source mass flow both pass."]}
 (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"groups":report["mask_group_hits"],"other_group_cells":len(support_other),"protected_missing":sum(x["missing"] for x in protected),"protected_missing_exact_boundary_points":sum(x["missing_exact_boundary_points"] for x in protected),"invalid":len(invalid),"elapsed_s":report["elapsed_s"]}))
if __name__=="__main__":main()
