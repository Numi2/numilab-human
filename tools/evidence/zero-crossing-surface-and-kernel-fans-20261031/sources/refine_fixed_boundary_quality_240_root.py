"""Fixed-boundary local quality refinement through the existing TetGen owner.
Compiler construction only; accepted coordinates, mass and native state stay untouched.
"""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import argparse,importlib.util,sys,time,json,subprocess,numpy as np
H=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
top=load("top218for240",H/"remesh_complete_surface_cavities_218_root.py")
m=top.m
dom=load("dom217for240",H/"native-domain-217-source-004/build_native_domain_217.py")
OWNER=H.parent/"deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py"
sys.path.insert(0,str(OWNER.parent.parent))
exact=__import__("numilab_human.cardiac_cavity_intersections",fromlist=["_audit_pair"])
TETGEN=H/"attribute-propagation-062/hull-attribute-fix-001/tetgen-hull-attribute-fix"
def solve(p,old,mat,case,seconds=2):
 b=m.boundary(old);vertices=np.unique(b);ix=np.full(len(p),-1,np.int64);ix[vertices]=np.arange(len(vertices));fixed=p[vertices];lb=ix[b]
 if len({m.key(x) for x in fixed})!=len(fixed):raise ValueError("duplicate F32 boundary")
 if not top.closed(lb)[0]:raise ValueError("nonclosed cavity")
 topo=top.manifold_topology(lb)
 if topo["bad_vertex_links"] or topo["boundary_components"]!=1:raise ValueError("nonmanifold cavity")
 records,_=exact.float32_surface_records(fixed,lb)
 if exact._audit_pair(records,records,same_surface=True)["count"]:raise ValueError("self-crossing cavity shell")
 prepared=exact.prepare_float32_point_location_meters(fixed,lb)
 det,qual=m.m.measurements(p,old);seed=None
 for row in np.argsort(det)[::-1][:min(30,len(old))]:
  q=tuple(sum(Fraction(float(v)) for v in p[old[row],axis])/4 for axis in range(3))
  if exact.point_location_prepared_indexed(q,prepared)["location"]=="inside":seed=np.array([float(v) for v in q]);break
 if seed is None:raise ValueError("no exact interior seed")
 edges=np.vstack([lb[:,[0,1]],lb[:,[1,2]],lb[:,[2,0]]]);length=np.linalg.norm(fixed[edges[:,0]].astype(float)-fixed[edges[:,1]],axis=1)
 diag=np.linalg.norm(np.ptp(fixed.astype(float),axis=0));tol=.1*float(length.min())/float(diag)
 case.mkdir();poly=case/"cavity.poly";dom.write_poly(poly,fixed,lb,np.full(len(lb),100,np.int32),[],[(seed,int(mat))])
 argv=[str(TETGEN),"-pYYAq5S64O2T%.17g"%tol,poly.name]
 pr=subprocess.run(argv,cwd=case,capture_output=True,text=True,timeout=seconds)
 (case/"stdout.log").write_text(pr.stdout);(case/"stderr.log").write_text(pr.stderr)
 if pr.returncode:raise ValueError("TetGen return "+str(pr.returncode))
 nn=top.readtable(case/"cavity.1.node");ee=top.readtable(case/"cavity.1.ele");op=nn[:,1:4].astype(np.float32);nid=nn[:,0].astype(np.int64);nt=np.searchsorted(nid,ee[:,1:5].astype(np.int64))
 if not np.array_equal(nid[nt],ee[:,1:5].astype(np.int64)) or ee.shape[1]<6 or not np.all(ee[:,5]==mat):raise ValueError("identity/material")
 nd,nq=m.m.measurements(op,nt)
 if np.any(nd<=0):raise ValueError("F32 nonpositive")
 oldbad=int(sum(~m.m.passes(det,qual)));newbad=int(sum(~m.m.passes(nd,nq)))
 if newbad>=oldbad:raise ValueError("no strict total native-defect descent")
 lookup={m.key(x):i for i,x in enumerate(fixed)};omap=np.array([lookup.get(m.key(x),-1) for x in op],np.int64);nb=m.boundary(nt)
 if np.any(omap[nb]<0) or not np.array_equal(m.boundary_identity(omap[nb]),m.boundary_identity(lb)):raise ValueError("fixed boundary")
 for er in np.flatnonzero(omap<0):
  if exact.point_location_prepared_indexed(tuple(Fraction(float(v)) for v in op[er]),prepared)["location"]!="inside":raise ValueError("Steiner not exactly inside")
 ip=m.m.integer_points(op);ds=[m.m.exact_det(ip,tet) for tet in nt]
 if min(ds)<=0 or sum(ds)!=top.exact_volume(fixed,lb):raise ValueError("exact sign/volume")
 fs=nt[:,m.m.PAT].reshape(-1,3);_,u=np.unique(np.sort(fs,axis=1),axis=0,return_index=True);rr,_=exact.float32_surface_records(op,fs[u])
 if exact._audit_pair(rr,rr,same_surface=True)["count"]:raise ValueError("internal face crossing")
 return op,nt,omap,vertices,{"old_bad":oldbad,"new_bad":newbad,"old_cells":len(old),"new_cells":len(nt),"interior_points":int(sum(omap<0)),"exact_F32_positive":True,"exact_oriented_boundary":True,"exact_volume":True,"exact_internal_face_crossings":0,"argv":argv}
def fixture(out):
 out.mkdir(exist_ok=False,parents=True)
 p=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01],[.003,.003,1e-9]],np.float32)
 b=m.boundary(np.array([[0,1,2,3]],np.int64));t=np.c_[np.full(4,4),b]
 assert np.all(m.m.measurements(p,t)[0]>0)
 op,nt,omap,vertices,r=solve(p,t,15,out/"actual-owner")
 assert r["new_bad"]<r["old_bad"] and np.all(m.m.passes(*m.m.measurements(op,nt)))
 r.update(status="PASS_FIXED_BOUNDARY_REFINEMENT_CONTROL",physical_admission=False,native_steps=0)
 (out/"report.json").write_text(json.dumps(r,indent=2)+"\n");return r
def run(a):
 tic=time.monotonic();a.output.mkdir(exist_ok=False,parents=True)
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v["points_world_m"];t=v["tetrahedra"];mat=v["material_indices"];d,q=m.m.measurements(p,t);g=m.m.passes(d,q)
 if np.any(d<=0):raise ValueError("positive input required")
 flat=t.ravel();order=np.argsort(flat,kind="stable");off=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 hits=np.bincount(t[~g].ravel(),minlength=len(p));degree=np.diff(off);nodes=sorted(map(int,np.flatnonzero(hits)),key=lambda n:(-float(hits[n])/max(1,int(degree[n])),-int(hits[n]),n))
 used=set();removed=[];added=[];newmats=[];newpoints=[];ops=[];rejections=Counter();attempts=0;seen=set();globalkeys={m.key(x) for x in p}
 for node in nodes:
  if time.monotonic()-tic>a.seconds or attempts>=a.maximum_attempts:break
  rows=np.unique(order[off[node]:off[node+1]]//4);ot=t[rows];vs=set(map(int,ot.ravel()));key=tuple(rows.tolist())
  if key in seen:continue
  seen.add(key)
  if vs&used or len(vs)>a.maximum_points or len(np.unique(mat[rows]))!=1:continue
  attempts+=1;case=a.output/("trial-%04d"%attempts)
  try:
   op,nt,omap,vertices,rr=solve(p,ot,int(mat[rows[0]]),case,a.component_seconds)
   gm=np.full(len(op),-1,np.int64);mask=omap>=0;gm[mask]=vertices[omap[mask]]
   extras=np.flatnonzero(~mask)
   if any(m.key(op[x]) in globalkeys for x in extras):raise ValueError("new point coincides with another existing point")
   for x in extras:
    gm[x]=len(p)+len(newpoints);newpoints.append(op[x]);globalkeys.add(m.key(op[x]))
   gt=gm[nt];used.update(vs);removed.extend(rows.tolist());added.append(gt);newmats.append(np.full(len(gt),int(mat[rows[0]]),np.int32))
   rr.update(node=node,old_cell_rows=rows.tolist(),new_global_point_rows=gm[extras].tolist(),source_ancestry_rule="rebind strictly interior nodes against retained material region; mass allocation remains mandatory")
   np.savez_compressed(case/"replacement.npz",old_cell_rows=rows,new_tetrahedra=gt,new_global_point_rows=gm[extras],new_interior_points_world_m=op[extras],material_indices=newmats[-1])
   (case/"report.json").write_text(json.dumps(rr,indent=2)+"\n");ops.append(rr)
  except Exception as e:
   reason=type(e).__name__+":"+str(e);rejections[reason]+=1
   if case.exists():(case/"failure.json").write_text(json.dumps({"node":node,"old_cell_rows":rows.tolist(),"failure":reason},indent=2)+"\n")
 keep=np.ones(len(t),bool);keep[removed]=False;pp=np.vstack([p,np.array(newpoints,np.float32).reshape(-1,3)]);tt=np.vstack([t[keep],*(added or [np.empty((0,4),np.int64)])]);mm=np.concatenate([mat[keep],*newmats]);bd=m.boundary(tt)
 assert np.array_equal(m.boundary_identity(bd),m.boundary_identity(m.boundary(t)))
 nd,nq=m.m.measurements(pp,tt);assert np.all(nd>0)
 assert int(sum(~m.m.passes(nd,nq)))==int(sum(~g))-sum(o["old_bad"]-o["new_bad"] for o in ops)
 assert len(np.unique(np.sort(tt,axis=1),axis=0))==len(tt)
 out=a.output/"fixed-boundary-refined-volume-proposal.npz"
 np.savez_compressed(out,points_world_m=pp,tetrahedra=tt,material_indices=mm,boundary_faces=bd,desired_boundary_faces=v.get("desired_boundary_faces",bd),solver_infeasible_cavity_sha256=v.get("solver_infeasible_cavity_sha256",np.empty((0,32),np.uint8)))
 r={"status":"PASS_FIXED_BOUNDARY_NATIVE_DEFECT_REFINEMENT" if ops else "NO_FIXED_BOUNDARY_REFINEMENT_PROGRESS","physical_admission":False,"native_steps":0,"input":str(a.input),"input_sha256":m.sha(a.input),"script_sha256":m.sha(Path(__file__)),"owners":{str(x):m.sha(x) for x in (TETGEN,OWNER,Path(top.__file__))},"output":str(out),"output_sha256":m.sha(out),"attempts":attempts,"passing_operations":len(ops),"bad_before":int(sum(~g)),"bad_after":int(sum(~m.m.passes(nd,nq))),"new_interior_points":len(newpoints),"output_cells":len(tt),"nonpositive":int(sum(nd<=0)),"oriented_boundary_bit_identical":True,"original_points_bit_identical":True,"operations":ops,"rejections":dict(rejections),"elapsed_s":time.monotonic()-tic}
 (a.output/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({k:x for k,x in r.items() if k!="operations"},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--input",type=Path);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seconds",type=float,default=45);ap.add_argument("--maximum-attempts",type=int,default=100);ap.add_argument("--maximum-points",type=int,default=60);ap.add_argument("--component-seconds",type=float,default=2);a=ap.parse_args()
 if a.fixture:print(json.dumps(fixture(a.output),indent=2))
 else:run(a)
