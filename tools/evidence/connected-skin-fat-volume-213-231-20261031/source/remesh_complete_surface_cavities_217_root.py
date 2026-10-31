"""Connected local cavity remeshing of complete skin/fat surface quotients.
Reuses existing TetGen, PLC writer and exact geometry owner. No runtime admission.
"""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import numpy as np,json,time,argparse,importlib.util,sys,subprocess
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
H=Path(__file__).resolve().parent;P=H.parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load("cavity209",H/"transfer_surface_contractions_to_volume_209_root.py")
base=load("quality213",H/"quality_star_cavity_213_root.py")
def sortedkeys(f):return {tuple(sorted(map(int,x))) for x in f}
def closed(b):
 ed=np.vstack([b[:,[0,1]],b[:,[1,2]],b[:,[2,0]]]);se=np.sort(ed,axis=1);order=np.lexsort((se[:,1],se[:,0]));ss=se[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];ct=np.diff(np.r_[starts,len(ss)])
 signs=np.where(ed[:,0]<ed[:,1],1,-1);sm=np.add.reduceat(signs[order],starts)
 return bool(np.all(ct==2) and np.all(sm==0)),int(sum((ct!=2)|(sm!=0)))
def prepare_once(p,t,desired,bb,missing,extra,rings,additional_seeds=()):
 d,q=m.m.measurements(p,t);dirty=np.unique(np.r_[t[d<=0].ravel(),missing.ravel(),extra.ravel(),np.asarray(additional_seeds,np.int64)])
 mask=np.zeros(len(p),bool);mask[dirty]=True;selected=np.flatnonzero(np.any(mask[t],axis=1))
 neigh=base.adjacency(t);active=np.zeros(len(t),bool);active[selected]=True
 for _ in range(rings):
  ns=neigh[active].ravel();active[ns[ns>=0]]=True
 selected=np.flatnonzero(active)
 # Vertex-connected grouping conservatively combines interacting cavities.
 st=t[selected];flat=st.ravel();order=np.argsort(flat,kind="stable");sf=flat[order]
 starts=np.r_[0,1+np.flatnonzero(sf[1:]!=sf[:-1])];counts=np.diff(np.r_[starts,len(sf)])
 x=[];y=[]
 for a,n in zip(starts,counts):
  cells=np.unique(order[a:a+n]//4)
  if len(cells)>1:x.extend([int(cells[0])]*(len(cells)-1));y.extend(cells[1:].tolist())
 graph=coo_matrix((np.ones(len(x),bool),(x,y)),shape=(len(selected),len(selected))).tocsr()
 nc,labels=connected_components(graph,directed=False)
 nodecomp=defaultdict(set)
 for r,l in zip(st,labels):
  for v in r:nodecomp[int(v)].add(int(l))
 assign=defaultdict(list)
 for f in missing:
  cs=set.union(*(nodecomp.get(int(v),set()) for v in f))
  if len(cs)!=1:raise ValueError("new desired facet crosses unmerged cavity components")
  assign[next(iter(cs))].append(f)
 extrakeys=sortedkeys(extra);components=[]
 for ci in range(nc):
  rows=selected[labels==ci];oldb=m.boundary(t[rows])
  b=np.asarray([f for f in oldb if tuple(sorted(map(int,f))) not in extrakeys],np.int64).reshape(-1,3)
  newf=np.asarray(assign[ci],np.int64).reshape(-1,3);b=np.vstack([b,newf])
  ok,nfail=closed(b);grows=np.unique(b);idx=np.full(len(p),-1,np.int64);idx[grows]=np.arange(len(grows))
  components.append({"id":ci,"rows":rows,"global_points":grows,"faces_local":idx[b],"closed":ok,"edge_failures":nfail,"old_bad":int(sum(~m.m.passes(d[rows],q[rows]))),"old_nonpositive":int(sum(d[rows]<=0)),"old_cells":len(rows),"new_desired_faces":len(newf)})
 return components,{"selected_cells":len(selected),"dirty_nodes":len(dirty),"components":nc,"rings":rings}
def bad_edge_nodes(faces):
 ed=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);se=np.sort(ed,axis=1);order=np.lexsort((se[:,1],se[:,0]));ss=se[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];ct=np.diff(np.r_[starts,len(ss)])
 signs=np.where(ed[:,0]<ed[:,1],1,-1);sm=np.add.reduceat(signs[order],starts);bad=(ct!=2)|(sm!=0)
 return np.unique(ss[starts[bad]])
def prepare(p,t,desired,bb,missing,extra,rings):
 seeds=set();history=[]
 for iteration in range(12):
  comps,report=prepare_once(p,t,desired,bb,missing,extra,rings,sorted(seeds))
  failed=[c for c in comps if not c["closed"]]
  history.append({"iteration":iteration,"selected_cells":report["selected_cells"],"components":len(comps),"unclosed_components":len(failed),"bad_boundary_edges":sum(c["edge_failures"] for c in failed)})
  if not failed:break
  new=set()
  for c in failed:
   f=c["global_points"][c["faces_local"]];new.update(map(int,bad_edge_nodes(f)))
  if new<=seeds or report["selected_cells"]>50000:break
  seeds.update(new)
 report["closure_history"]=history;report["additional_witness_seed_nodes"]=len(seeds)
 return comps,report
def readtable(p):
 with open(p) as f:n=int(f.readline().split()[0])
 a=np.loadtxt(p,skiprows=1,max_rows=n)
 return np.atleast_2d(a)
def exact_volume(p,faces):
 # Oriented shell determinant sum on the exact Float32 integer lattice.
 ip=m.m.integer_points(p);v=0
 for f in faces:
  a,b,c=[ip[int(x)] for x in f]
  v+=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
 return v
def fixture():
 p=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]],np.float32)
 t=np.array([[0,1,2,3]],np.int64);bb=m.boundary(t);assert closed(bb)==(True,0)
 # One reversed desired facet fails global orientation.
 bad=bb.copy();bad[0,[0,1]]=bad[0,[1,0]];assert not closed(bad)[0]
 assert len(bad_edge_nodes(bb))==0 and len(bad_edge_nodes(bad))>0
 ip=m.m.integer_points(p);assert exact_volume(p,bb)==m.m.exact_det(ip,t[0])
 return {"status":"PASS_CONNECTED_CAVITY_PREPARATION_CONTROLS","controls":["oriented_closed_boundary","reversed_face_rejected","exact_surface_cell_volume"],"native_steps":0,"physical_admission":False}
def run(a):
 a.output.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v["points_world_m"];t=v["tetrahedra"];mat=v["material_indices"];desired=v["desired_boundary_faces"];bb=v["combinatorial_boundary_faces"] if "combinatorial_boundary_faces" in v else v["boundary_faces"]
 ak=sortedkeys(bb);dk=sortedkeys(desired)
 missing=v.get("desired_missing_faces",np.asarray([f for f in desired if tuple(sorted(map(int,f))) not in ak],np.int64).reshape(-1,3))
 extra=v.get("current_extra_boundary_faces",np.asarray([f for f in bb if tuple(sorted(map(int,f))) not in dk],np.int64).reshape(-1,3))
 comps,prep=prepare(p,t,desired,bb,missing,extra,a.rings)
 report={"status":"CONNECTED_CAVITY_PREPARATION","physical_admission":False,"native_steps":0,"input":str(a.input),"input_sha256":m.sha(a.input),"script_sha256":m.sha(Path(__file__)),"preparation":prep,"components":[]}
 def save():
  (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 for c in comps:
  report["components"].append({k:(x.tolist() if isinstance(x,np.ndarray) else x) for k,x in c.items() if k not in ("rows","global_points","faces_local")})
 save()
 arrays={};po=[0];fo=[0];co=[0]
 for c in comps:po.append(po[-1]+len(c["global_points"]));fo.append(fo[-1]+len(c["faces_local"]));co.append(co[-1]+len(c["rows"]))
 np.savez_compressed(a.output/"connected-cavity-shells.npz",point_offsets=po,face_offsets=fo,cell_offsets=co,global_point_rows=np.concatenate([c["global_points"] for c in comps]),local_faces=np.concatenate([c["faces_local"] for c in comps]),input_cell_rows=np.concatenate([c["rows"] for c in comps]))
 if a.prepare_only:print(json.dumps(report,indent=2));return
 owner=P/"deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py"
 sys.path.insert(0,str(owner.parent.parent));exact=__import__("numilab_human.cardiac_cavity_intersections",fromlist=["_audit_pair"])
 dom=load("plc_domain004",H/"native-domain-217-source-004/build_native_domain_217.py")
 tetgen=H/"attribute-propagation-062/hull-attribute-fix-001/tetgen-hull-attribute-fix"
 report["owners"]={str(x):m.sha(x) for x in (owner,tetgen,H/"native-domain-217-source-004/build_native_domain_217.py")}
 newpoints=[];replacements=[];removed=[];newmats=[];global_lookup={m.key(x):i for i,x in enumerate(p)}
 for ci,c in enumerate(comps):
  rr=report["components"][ci];case=a.output/("component-%04d"%ci);case.mkdir()
  try:
   if time.monotonic()-tic>a.seconds:rr["status"]="NOT_RUN_GLOBAL_BUDGET";save();continue
   if not c["closed"]:raise ValueError("restored shell not closed oriented")
   grows=c["global_points"];lf=c["faces_local"];fixed=p[grows];rows=c["rows"]
   if len(np.unique(mat[rows]))!=1:raise ValueError("mixed material cavity")
   vol=exact_volume(fixed,lf)
   if vol<=0:raise ValueError("restored shell volume not positive")
   records,_=exact.float32_surface_records(fixed,lf);sg=exact._audit_pair(records,records,same_surface=True)
   rr["exact_shell_self_pairs"]=int(sg["count"])
   if sg["count"]:raise ValueError("restored shell self-intersection")
   prepared=exact.prepare_float32_point_location_meters(fixed,lf)
   dd,qq=m.m.measurements(p,t[rows]);candidate_rows=rows[np.argsort(dd)[::-1]];seed=None
   for cr in candidate_rows[:min(200,len(rows))]:
    if dd[np.flatnonzero(rows==cr)[0]]<=0:continue
    q=tuple(sum(Fraction(float(v)) for v in p[t[cr],axis])/4 for axis in range(3))
    location=exact.point_location_prepared_indexed(q,prepared)
    if location["location"]=="inside":seed=np.asarray([float(v) for v in q]);break
   if seed is None:raise ValueError("no exact interior region seed in restored cavity")
   edges=np.vstack([lf[:,[0,1]],lf[:,[1,2]],lf[:,[2,0]]]);length=np.linalg.norm(fixed[edges[:,0]].astype(float)-fixed[edges[:,1]],axis=1)
   diag=float(np.linalg.norm(np.ptp(fixed.astype(float),axis=0)));tol=.1*float(length.min())/diag
   poly=case/"cavity.poly";dom.write_poly(poly,fixed,lf,np.full(len(lf),100,np.int32),[],[(seed,int(mat[rows[0]]))])
   arg=("-pYYAq5S128O2T%.17g"%tol) if a.refine else ("-pYYAS0O0/0/0T%.17g"%tol);rr["argv"]=[str(tetgen),arg,poly.name];rr["poly_sha256"]=m.sha(poly);rr["status"]="TETGEN_RUNNING";save()
   pr=subprocess.run([str(tetgen),arg,poly.name],cwd=case,capture_output=True,text=True,timeout=a.component_seconds)
   (case/"tetgen.stdout.log").write_text(pr.stdout);(case/"tetgen.stderr.log").write_text(pr.stderr);rr["return_code"]=pr.returncode
   if pr.returncode:raise ValueError("TetGen failed")
   nn=readtable(case/"cavity.1.node");ee=readtable(case/"cavity.1.ele");op=nn[:,1:4].astype(np.float32);nid=nn[:,0].astype(np.int64);ix=np.searchsorted(nid,ee[:,1:5].astype(np.int64));ot=ix.astype(np.int64)
   if not np.array_equal(nid[ix],ee[:,1:5].astype(np.int64)):raise ValueError("TetGen node identity")
   if ee.shape[1]<6 or not np.all(ee[:,5]==mat[rows[0]]):raise ValueError("TetGen material ownership")
   od,oq=m.m.measurements(op,ot)
   if np.any(od<=0):raise ValueError("TetGen F32 nonpositive")
   ob=m.boundary(ot);lookup={m.key(x):i for i,x in enumerate(fixed)}
   # New nodes may be strictly interior; no boundary Steiner replacement is admitted.
   omap=np.asarray([lookup.get(m.key(x),-1) for x in op],np.int64)
   if np.any(omap[ob]<0):raise ValueError("new point is on fixed boundary")
   if not np.array_equal(m.boundary_identity(omap[ob]),m.boundary_identity(lf)):raise ValueError("fixed oriented boundary differs")
   extras=np.flatnonzero(omap<0)
   for er in extras:
    x=tuple(Fraction(float(v)) for v in op[er])
    if exact.point_location_prepared_indexed(x,prepared)["location"]!="inside":raise ValueError("new Steiner point is not exactly inside restored cavity")
   ip=m.m.integer_points(op);eds=[m.m.exact_det(ip,x) for x in ot]
   if any(x<=0 for x in eds) or sum(eds)!=vol:raise ValueError("exact F32 signed volume mismatch")
   newbad=int(sum(~m.m.passes(od,oq)))
   if newbad>c["old_bad"]:raise ValueError("native quality failure count increased")
   gm=np.full(len(op),-1,np.int64);fixedmask=omap>=0;gm[fixedmask]=grows[omap[fixedmask]]
   added=[]
   for er in extras:
    k=m.key(op[er]);found=global_lookup.get(k)
    if found is None:
     found=len(p)+len(newpoints);newpoints.append(op[er]);global_lookup[k]=found;added.append((found,op[er].copy()))
    gm[er]=found
   gt=gm[ot];removed.extend(rows.tolist());replacements.append(gt);newmats.extend([int(mat[rows[0]])]*len(gt))
   np.savez_compressed(case/"replacement.npz",tetrahedra=gt,parent_cell_rows=rows,global_point_rows=grows,desired_shell_faces_global=grows[lf],new_global_point_rows=np.asarray([x[0] for x in added],np.int64),new_interior_points_world_m=np.asarray([x[1] for x in added],np.float32).reshape(-1,3))
   rr["new_interior_points"]=len(extras);rr["new_interior_global_rows"]=[int(gm[x]) for x in extras]
   rr.update(status="PASS_CONNECTED_FIXED_SURFACE_CAVITY",new_cells=len(gt),new_bad=newbad,all_F32_positive=True,oriented_boundary_exact=True,exact_new_shell_volume=True,replacement_sha256=m.sha(case/"replacement.npz"))
  except subprocess.TimeoutExpired as e:
   (case/"tetgen.timeout.log").write_text(str(e));rr["status"]="REJECT_TETGEN_TIMEOUT"
  except Exception as e:rr.update(status="REJECT",error=type(e).__name__+": "+str(e))
  save()
 keep=np.ones(len(t),bool);keep[removed]=False
 tt=np.vstack([t[keep],*(replacements or [np.empty((0,4),np.int64)])]);mm=np.r_[mat[keep],np.asarray(newmats,np.int32)]
 pp=np.vstack([p,np.asarray(newpoints,np.float32).reshape(-1,3)]);od,oq=m.m.measurements(pp,tt);bound=m.boundary(tt)
 output=a.output/"connected-fixed-surface-volume-proposal.npz"
 np.savez_compressed(output,points_world_m=pp,tetrahedra=tt,material_indices=mm,boundary_faces=bound,desired_boundary_faces=desired,retained_input_cell_rows=np.flatnonzero(keep))
 matches=np.array_equal(m.boundary_identity(bound),m.boundary_identity(desired))
 report.update(status="CONNECTED_CAVITY_BATCH_PROPOSAL",output=str(output),output_sha256=m.sha(output),passing_components=len(replacements),remaining_nonpositive_cells=int(sum(od<=0)),remaining_quality_failures=int(sum(~m.m.passes(od,oq))),complete_boundary_matches_desired=bool(matches),elapsed_s=time.monotonic()-tic)
 save();print(json.dumps({k:v for k,v in report.items() if k!="components"},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--input",type=Path);ap.add_argument("--output",type=Path);ap.add_argument("--rings",type=int,default=1);ap.add_argument("--prepare-only",action="store_true");ap.add_argument("--refine",action="store_true");ap.add_argument("--seconds",type=float,default=120);ap.add_argument("--component-seconds",type=float,default=10);a=ap.parse_args()
 if a.fixture:print(json.dumps(fixture()))
 else:run(a)
