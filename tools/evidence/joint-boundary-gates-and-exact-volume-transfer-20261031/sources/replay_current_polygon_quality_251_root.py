"""Replay existing polygon flips with complete-domain boundary/cache checks.
No change to 198/202 cavity mathematics or native admission thresholds.
"""
from pathlib import Path
import argparse,json,importlib.util,sys,numpy as np,hashlib
H=Path(__file__).resolve().parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
op=load("polygon202for251",H/"fixed_boundary_volume_polygon_flip_202_root.py")
top=load("top218for251",H/"remesh_complete_surface_cavities_218_root.py")
OWNER=H.parent/"deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py"
sys.path.insert(0,str(OWNER.parent.parent));exact=__import__("numilab_human.cardiac_cavity_intersections",fromlist=["_audit_pair"])
def run(a):
 with np.load(a.input_npz) as z:original={k:z[k] for k in z.files}
 op.run(a)
 out=a.output/"fixed-boundary-volume-proposal.npz"
 with np.load(out) as z:v={k:z[k] for k in z.files}
 p=v["points_world_m"];t=v["tetrahedra"]
 assert np.array_equal(p.view(np.uint32),original["points_world_m"].view(np.uint32))
 ib=top.m.boundary(original["tetrahedra"]);ob=top.m.boundary(t)
 assert np.array_equal(top.m.boundary_identity(ib),top.m.boundary_identity(ob))
 assert np.array_equal(top.m.boundary_identity(ob),top.m.boundary_identity(original.get("desired_boundary_faces",ib)))
 assert len(np.unique(np.sort(t,axis=1),axis=0))==len(t)
 d,q=op.m.measurements(p,t);assert np.all(d>0)
 new=v["replacement_tetrahedra"];faces=new[:,op.m.PAT].reshape(-1,3)
 if len(faces):
  _,idx=np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
  records,_=exact.float32_surface_records(p,faces[idx]);gate=exact._audit_pair(records,records,same_surface=True);assert gate["count"]==0,gate
 v["boundary_faces"]=ob;v["desired_boundary_faces"]=original.get("desired_boundary_faces",ib)
 for name in ("solver_infeasible_cavity_sha256","kernel_fan_no_progress_sha256"):
  if name in original:v[name]=original[name]
 np.savez_compressed(out,**v)
 report=a.output/"report.json";r=json.loads(report.read_text())
 r.update(full_oriented_boundary_replay=True,complete_current_target_boundary_replay=True,replacement_internal_face_crossings=0,duplicate_cells=0,nonpositive=0,cache_metadata_preserved=True,
  output_sha256=op.sha(out),wrapper_source_sha256=op.sha(Path(__file__)),exact_owner_sha256=op.sha(OWNER),
  native_absolute_volume_failures=int(sum(d/6<=1e-18)),native_normalized_determinant_failures=int(sum(q<1e-5)))
 report.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({k:v for k,v in r.items() if k!="operations"}))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--input-npz",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seconds",type=float,default=45);ap.add_argument("--maximum-degree",type=int,default=12);ap.add_argument("--maximum-operations",type=int,default=10000);a=ap.parse_args();run(a)
