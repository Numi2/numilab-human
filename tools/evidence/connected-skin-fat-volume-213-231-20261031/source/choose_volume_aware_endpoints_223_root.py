"""Try alternate existing endpoint representatives for D-only contraction clusters.
Numerical and face-shape proposal only: exact intersections/provenance remain
mandatory before any promotion. No new coordinates, force or state ownership.
"""
from pathlib import Path
from collections import Counter
import numpy as np,json,argparse,time,importlib.util
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location("base209",H/"transfer_surface_contractions_to_volume_209_root.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def face_measure(p,b):
 q=p[b].astype(float);n=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);area=np.linalg.norm(n,axis=1)
 edge=np.maximum.reduce([np.linalg.norm(q[:,i]-q[:,j],axis=1) for i,j in ((0,1),(1,2),(2,0))])
 return area/np.maximum(edge**2,1e-300),n
def run(a):
 a.output.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v["points_world_m"];t=v["tetrahedra"].copy();b=v["desired_boundary_faces"].copy();mp=v["input_node_quotient"].copy()
 lookup={m.key(x):i for i,x in enumerate(p)}
 with np.load(a.d_surface) as z:ds=z["points_world_m"];df=z["faces"]
 with np.load(a.i_surface) as z:ii=z["points_world_m"];iff=z["faces"]
 dnodes={lookup[m.key(x)] for x in ds[np.unique(df)]};inodes={lookup[m.key(x)] for x in ii[np.unique(iff)]}
 eligible=dnodes-inodes
 flat=t.ravel();order=np.argsort(flat,kind="stable");off=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 sort=np.argsort(mp,kind="stable");counts=np.bincount(mp,minlength=len(p));coff=np.r_[0,np.cumsum(counts)]
 det,q=m.m.measurements(p,t);initialneg=int(sum(det<=0));initialbad=int(sum(~m.m.passes(det,q)))
 candidates=[]
 for root in np.unique(t[det<=0]):
  if root not in eligible or counts[root]<2:continue
  rows=np.unique(order[off[root]:off[root+1]]//4)
  candidates.append((int(root),rows))
 candidates.sort(key=lambda c:(-int(sum(det[c[1]]<=0)),c[0]))
 used=set(map(int,np.unique(t)));ops=[];rej=Counter();tries=0
 for root,rows in candidates:
  old=t[rows].copy();od,oq=m.m.measurements(p,old);og=m.m.passes(od,oq);oldneg=od<=0
  if not np.any(oldneg):continue
  br=np.flatnonzero(np.any(b==root,axis=1));ob=b[br].copy();fb,fn=face_measure(p,ob);fg=fb>=1e-5
  best=None;members=sort[coff[root]:coff[root+1]]
  for alt in members:
   alt=int(alt)
   if alt==root:continue
   if alt in used:rej["alternate_still_active"]+=1;continue
   nt=np.where(old==root,alt,old);nd,nq=m.m.measurements(p,nt);ng=m.m.passes(nd,nq);nneg=nd<=0;tries+=1
   if np.any(~oldneg&nneg) or sum(nneg)>=sum(oldneg):rej["nonpositive_not_reduced"]+=1;continue
   if np.any(og&~ng) or sum(~ng)>sum(~og):rej["volume_quality_regression"]+=1;continue
   nb=np.where(ob==root,alt,ob);nf,nn=face_measure(p,nb)
   if np.any(nf<=0) or np.any(fg&(nf<1e-5)):rej["boundary_quality_regression"]+=1;continue
   dots=np.einsum("ij,ij->i",fn,nn)
   if np.any(dots<=0):rej["boundary_normal_flip"]+=1;continue
   # Exact signs for the modified cells; existing nonpositive rows may remain.
   nodes=np.unique(nt);ik={int(v):i for i,v in enumerate(nodes)};ip=m.m.integer_points(p[nodes])
   signs=np.asarray([np.sign(m.m.exact_det(ip,[ik[int(x)] for x in tet])) for tet in nt],np.int8)
   if np.any((nd>0)&(signs<=0)) or np.any((~oldneg)&(signs<=0)):rej["exact_F32_sign"]+=1;continue
   score=(int(sum(nneg)),int(sum(~ng)),-float(nq.min()))
   if best is None or score<best[0]:best=(score,alt,nt,nb,nd,nq)
  if best is None:continue
  score,alt,nt,nb,nd,nq=best;t[rows]=nt;b[br]=nb;used.remove(root);used.add(alt);mp[mp==root]=alt
  ops.append({"retired_current_node":root,"selected_original_node":alt,"affected_cell_rows":rows.tolist(),"affected_desired_face_rows":br.tolist(),"old_nonpositive":int(sum(oldneg)),"new_nonpositive":int(sum(nd<=0)),"old_quality_failures":int(sum(~og)),"new_quality_failures":int(sum(~m.m.passes(nd,nq))),"old_coordinate_world_m":p[root].tolist(),"new_coordinate_world_m":p[alt].tolist(),"displacement_m":float(np.linalg.norm(p[root].astype(float)-p[alt]))})
 det,q=m.m.measurements(p,t);cb=m.boundary(t);ak={tuple(sorted(map(int,f))) for f in cb};dk={tuple(sorted(map(int,f))) for f in b}
 out=a.output/"alternate-endpoint-volume-and-boundary-proposal.npz"
 np.savez_compressed(out,points_world_m=p,tetrahedra=t,material_indices=v["material_indices"],desired_boundary_faces=b,combinatorial_boundary_faces=cb,input_node_quotient=mp,desired_missing_faces=np.asarray([f for f in b if tuple(sorted(map(int,f))) not in ak],np.int64).reshape(-1,3),current_extra_boundary_faces=np.asarray([f for f in cb if tuple(sorted(map(int,f))) not in dk],np.int64).reshape(-1,3),source_input_node_quotient=v["input_node_quotient"],changed_desired_face_rows=np.unique(np.concatenate([np.asarray(o["affected_desired_face_rows"],np.int64) for o in ops])) if ops else np.empty(0,np.int64))
 r={"status":"NUMERICAL_ENDPOINT_ALTERNATIVES_REQUIRE_EXACT_GEOMETRY_AND_ANCESTRY","physical_admission":False,"native_steps":0,"input":str(a.input),"input_sha256":m.sha(a.input),"script_sha256":m.sha(Path(__file__)),"D_surface":str(a.d_surface),"D_sha256":m.sha(a.d_surface),"I_surface":str(a.i_surface),"I_sha256":m.sha(a.i_surface),"output":str(out),"output_sha256":m.sha(out),"clusters_considered":len(candidates),"trials":tries,"passing_operations":len(ops),"nonpositive_before":initialneg,"nonpositive_after":int(sum(det<=0)),"quality_failures_before":initialbad,"quality_failures_after":int(sum(~m.m.passes(det,q))),"rejections":dict(rej),"elapsed_s":time.monotonic()-tic,"operations":ops}
 (a.output/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({k:v for k,v in r.items() if k!="operations"},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True);ap.add_argument("--d-surface",type=Path,required=True);ap.add_argument("--i-surface",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);run(ap.parse_args())
