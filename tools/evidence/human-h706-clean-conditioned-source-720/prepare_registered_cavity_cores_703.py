"""Prepare existing TetGen hole-core NPZs using authenticated current rims.
Construction caps identify cavity interiors only; no physical surface is capped.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
HELP=P/"reduced-physical-boundary-400/attempt-046-fine-role-simplification/run_joint_exact_audit_003.py"
BUILD=P/"volume-worker/native-domain-217/build_native_domain_217.py"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def close_with_registered_cap(points,faces,cap_points,cap_faces,helper,maximum_rim_error_m=2e-8):
 points=np.asarray(points,np.float32);faces=np.asarray(faces,np.int64)
 cp=np.asarray(cap_points,np.float32);cf=np.asarray(cap_faces,np.int64)
 s=helper.mesh_structure(points,faces)
 if s["zero_area_f32_widened_rows"] or s["nonmanifold_edges"] or s["same_direction_edges"] or s["duplicate_face_count"]:raise ValueError("source structure invalid")
 boundary=np.unique(np.asarray(s["boundary_directed_edges"],np.int64))
 used=np.unique(cf)
 if len(boundary)!=len(used):raise ValueError("cap/current boundary cardinality differs")
 if np.any(cf<0) or np.any(cf>=len(cp)):raise ValueError("cap vertex outside source domain")
 delta=cp[used,None,:].astype(np.float64)-points[boundary][None,:,:].astype(np.float64)
 d2=np.einsum("ijk,ijk->ij",delta,delta);nearest=d2.argmin(axis=1)
 if len(set(map(int,nearest)))!=len(used):raise ValueError("cap/rim map is not bijective")
 error=np.sqrt(d2[np.arange(len(used)),nearest])
 if np.any(error>maximum_rim_error_m):raise ValueError("cap/rim separation exceeds native 20nm identity tolerance")
 mapping=np.full(len(cp),-1,np.int64);mapping[used]=boundary[nearest]
 caps=mapping[cf]
 target={tuple(x) for x in s["boundary_directed_edges"]}
 def cap_boundary(f):return {tuple(x) for x in helper.mesh_structure(points,f)["boundary_directed_edges"]}
 cb=cap_boundary(caps)
 if cb==target:caps=caps[:,[0,2,1]];flipped=True
 elif cb=={(b,a) for a,b in target}:flipped=False
 else:raise ValueError("mapped cap boundary is not the exact directed source collar")
 core=np.vstack((faces,caps));cs=helper.mesh_structure(points,core)
 if cs["boundary_edges"] or cs["zero_area_f32_widened_rows"] or cs["nonmanifold_edges"] or cs["same_direction_edges"] or cs["duplicate_face_count"]:raise ValueError("construction core is not closed/oriented/nondegenerate")
 tri=points[core].astype(np.float64);origin=points.astype(np.float64).mean(axis=0)
 volume=float(np.sum(np.einsum("ij,ij->i",tri[:,0]-origin,np.cross(tri[:,1]-origin,tri[:,2]-origin)))/6.)
 if not np.isfinite(volume) or volume==0.:raise ValueError("core signed volume invalid")
 global_flip=volume<0
 if global_flip:core=core[:,[0,2,1]]
 return core,{"used_cap_source_vertices":used.tolist(),"mapped_current_vertices":mapping[used].tolist(),"maximum_rim_error_m":float(error.max(initial=0.)),"cap_winding_reversed":flipped,"core_global_winding_reversed":global_flip,"positive_signed_volume_m3":abs(volume),"source_faces":len(faces),"construction_cap_faces":len(caps),"points_bitwise_unchanged":True}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--surface",type=Path,required=True);ap.add_argument("--surface-sha256",required=True);ap.add_argument("--cap",type=Path,required=True);ap.add_argument("--cap-sha256",required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 if sha(a.surface)!=a.surface_sha256 or sha(a.cap)!=a.cap_sha256:raise ValueError("source pin mismatch")
 if sha(HELP)!="b202f43adf921cea0faeb30f32d921f0979516ca2e5655d9da51c72856906d0a" or sha(BUILD)!="bd266b63967f8120b95f2df817704defe9608dec011eed3013f510517c597b66":raise ValueError("owner pin mismatch")
 h=module("joint003_cavity_core",HELP);b=module("owner217_cavity_core",BUILD);pred=h.load_owner()
 d=h.load_npz_once(a.surface);c=h.load_npz_once(a.cap)
 pk=next(k for k in ("points_world_m_f32","points_world_m") if k in d);p=np.asarray(d[pk],np.float32);f=np.asarray(d["faces"],np.int64)
 cf=c.get("faces",c.get("source_faces"));core,report=close_with_registered_cap(p,f,c["points_world_m"],cf,h)
 # Entire original geometry is retained; check only new cap pairs against it.
 n=len(f);caprec,_=pred.float32_surface_records(p.astype(np.float64),core[n:],source_seam_groups=())
 caprec=[(r[0],r[1],r[2],n+j,r[4]) for j,r in enumerate(caprec)]
 bodyrec,_=pred.float32_surface_records(p.astype(np.float64),core[:n],source_seam_groups=())
 pairs=list(pred._query_surface_aabb(caprec,pred._prepare_surface_aabb(bodyrec)))
 res=pred._audit_record_pairs(pairs,same_surface=True)
 capself=pred._audit_pair(caprec,caprec,same_surface=True)
 report.update(cap_vs_source_exact_pairs=int(res["count"]),cap_self_exact_pairs=int(capself["count"]),cap_vs_source_aabb_pairs=len(pairs))
 if report["cap_vs_source_exact_pairs"] or report["cap_self_exact_pairs"]:raise ValueError("construction cap introduces unexplained intersections: "+json.dumps(report))
 mf=b._manifold(p,core)
 components=mf.decompose();positive=[x for x in components if float(x.volume())>1e-15]
 holes=[]
 for comp in positive:
  point=b._interior_point(comp)
  if not b._point_inside(mf,point):raise ValueError("positive component seed lies outside whole nested cavity core")
  holes.append(point)
 if not holes:raise ValueError("no positive cavity component")
 a.output.mkdir(parents=True)
 out=a.output/"cavity-core.npz"
 np.savez_compressed(out,points_world_m=p,faces=core,source_faces=core[:n],temporary_construction_cap_faces=core[n:],temporary_construction_only=np.asarray([True]),hole_points_world_m=np.asarray(holes,np.float64),parent_asset_path=np.asarray([str(a.surface),str(a.cap)]),parent_asset_sha256=np.asarray([a.surface_sha256,a.cap_sha256]))
 # Exercise the existing owner rather than a parallel hole-point implementation.
 owner_holes=b._core_holes([out])
 if not np.array_equal(np.asarray(owner_holes),np.asarray(holes)):raise ValueError("owner hole extraction parity failed")
 report.update(status="PASS_CONSTRUCTION_CORE_AND_OWNER_HOLE_EXTRACTION_ONLY",core_path=str(out),core_sha256=sha(out),surface_path=str(a.surface),surface_sha256=a.surface_sha256,cap_source_path=str(a.cap),cap_source_sha256=a.cap_sha256,positive_components=len(positive),component_count=len(components),negative_component_count=sum(float(x.volume())<0 for x in components),hole_points_world_m=np.asarray(holes).tolist(),physical_admission=False,runtime=0,accepted_steps=0,limitations=["Construction caps are not anatomical skin closure.","Not TetGen cells, mass transfer, native admission or motion qualification."])
 (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({k:report[k] for k in ("status","core_sha256","positive_components","component_count","maximum_rim_error_m")}),flush=True)
if __name__=="__main__":main()
