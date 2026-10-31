"""Replay independent fixed-coordinate quality cavities onto a repaired boundary.
Only exactly matching material/cell tuples can be replaced. Conflicting source
cavities are retained as skipped witnesses. No physical admission or mass mapping.
"""
from pathlib import Path
import argparse,json,time,importlib.util,numpy as np
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("joined209",H/"transfer_surface_contractions_to_volume_209_root.py");m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def key(t,mat):
 return np.sort(np.asarray(t,np.int64)).astype("<i8",copy=False).tobytes()+np.asarray([mat],"<i4").tobytes()
def replacements(stage):
 return sorted(stage.glob("solve-*/replacement.npz"))
def run(a):
 tic=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 with np.load(a.base) as z:v={k:z[k] for k in z.files}
 p=v["points_world_m"];bt=v["tetrahedra"];bm=v["material_indices"]
 dd,qq=m.m.measurements(p,bt);bg=m.m.passes(dd,qq)
 if np.any(dd<=0):raise ValueError("positive base required")
 lookup={key(t,mat):i for i,(t,mat) in enumerate(zip(bt,bm))}
 if len(lookup)!=len(bt):raise ValueError("duplicate base cell identity")
 added=[];am=[];deleted=[];total=len(bt);applied=[];skipped=[];stages=[]
 caches=[v.get("solver_infeasible_cavity_sha256",np.empty((0,32),np.uint8))]
 for stage in a.stage:
  sr=json.loads((stage/"report.json").read_text());source=Path(sr["input"])
  candidate=Path(sr["output"])
  if m.sha(candidate)!=sr["output_sha256"]:raise ValueError("stage output pin")
  with np.load(candidate) as zz:caches.append(zz["solver_infeasible_cavity_sha256"] if "solver_infeasible_cavity_sha256" in zz else np.empty((0,32),np.uint8))
  if m.sha(source)!=sr["input_sha256"]:raise ValueError("stage input pin")
  with np.load(source) as z:spoints=z["points_world_m"];st=z["tetrahedra"];sm=z["material_indices"]
  if p.shape!=spoints.shape or not np.array_equal(p.view(np.uint32),spoints.view(np.uint32)):raise ValueError("exact point domain mismatch")
  cases=replacements(stage)
  if len(cases)!=sr["passing_operations"]:raise ValueError("stage replacement count mismatch")
  stages.append({"report":str(stage/"report.json"),"sha256":m.sha(stage/"report.json"),"input":str(source),"input_sha256":m.sha(source),"operations":len(cases)})
  for case in cases:
   with np.load(case) as z:oldrows=z["old_cell_rows"];new=z["new_tetrahedra"];nm=z["material_indices"]
   old=st[oldrows];om=sm[oldrows]
   if len(np.unique(om))!=1 or np.any(nm!=om[0]):raise ValueError("material mismatch")
   oks=[key(t,mat) for t,mat in zip(old,om)];nks=[key(t,mat) for t,mat in zip(new,nm)]
   absent=[k for k in oks if k not in lookup]
   if absent:
    skipped.append({"case":str(case),"sha256":m.sha(case),"reason":"source cavity no longer exists in repaired base","missing_cells":len(absent)});continue
   existing=[k for k in nks if k in lookup and k not in oks]
   if existing or len(set(nks))!=len(nks):
    skipped.append({"case":str(case),"sha256":m.sha(case),"reason":"replacement cell already exists outside source cavity"});continue
   ok,why=m.m.certify(p,old,new)
   if not ok:raise ValueError("exact source cavity "+why)
   od,oq=m.m.measurements(p,old);nd,nq=m.m.measurements(p,new)
   before=int(sum(~m.m.passes(od,oq)));after=int(sum(~m.m.passes(nd,nq)))
   if np.any(nd<=0) or after>=before:raise ValueError("quality descent missing")
   ids=[lookup.pop(k) for k in oks];deleted.extend(ids)
   for j,k in enumerate(nks):lookup[k]=total+j
   total+=len(new);added.append(new);am.append(nm)
   applied.append({"case":str(case),"sha256":m.sha(case),"old_cells":len(old),"new_cells":len(new),"native_bad_before":before,"native_bad_after":after})
 tt=np.vstack([bt,*added]);mm=np.concatenate([bm,*am]);keep=np.ones(len(tt),bool);keep[deleted]=False;tt=tt[keep];mm=mm[keep]
 nd,nq=m.m.measurements(p,tt);bad=int(sum(~m.m.passes(nd,nq)))
 assert np.all(nd>0) and bad==int(sum(~bg))-sum(o["native_bad_before"]-o["native_bad_after"] for o in applied)
 assert len(np.unique(np.c_[np.sort(tt,axis=1),mm],axis=0))==len(tt)
 ob=m.boundary(tt);bb=m.boundary(bt);assert np.array_equal(m.boundary_identity(bb),m.boundary_identity(ob))
 out=a.output/"joined-fixed-boundary-quality-volume-proposal.npz"
 for cc in caches:
  if cc.dtype!=np.uint8 or cc.ndim!=2 or cc.shape[1]!=32:raise ValueError("cache shape")
 cache=np.unique(np.vstack(caches),axis=0)
 np.savez_compressed(out,solver_infeasible_cavity_sha256=cache,points_world_m=p,tetrahedra=tt,material_indices=mm,boundary_faces=ob,desired_boundary_faces=v.get("desired_boundary_faces",bb))
 desired=v.get("desired_boundary_faces",bb);ak={tuple(sorted(map(int,f)))for f in ob};dk={tuple(sorted(map(int,f)))for f in desired}
 r={"status":"PASS_EXACT_FIXED_BOUNDARY_QUALITY_PATCH_JOIN","physical_admission":False,"native_steps":0,"base":str(a.base),"base_sha256":m.sha(a.base),"script_sha256":m.sha(Path(__file__)),"stages":stages,"output":str(out),"output_sha256":m.sha(out),"input_cells":len(bt),"output_cells":len(tt),"nonpositive_cells":int(sum(nd<=0)),"native_bad_before":int(sum(~bg)),"native_bad_after":bad,"applied_operations":len(applied),"retained_solver_infeasible_cache_entries":len(cache),"skipped_operations":len(skipped),"oriented_boundary_unchanged":True,"points_bit_identical":True,"exact_F32_local_cavity_volumes":True,"desired_missing_faces":len(dk-ak),"current_extra_boundary_faces":len(ak-dk),"applied":applied,"skipped":skipped,"elapsed_s":time.monotonic()-tic}
 (a.output/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({k:v for k,v in r.items() if k not in ("applied","skipped")},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--base",type=Path,required=True);ap.add_argument("--stage",type=Path,action="append",required=True);ap.add_argument("--output",type=Path,required=True);run(ap.parse_args())
