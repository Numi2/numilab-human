"""Carry a validated physical-volume candidate to later retained endpoint maps.
Keeps extra interior nodes, same material, exact Float32 point prefix and all
source outputs. A proposal cannot be admitted by this compiler helper.
"""
from pathlib import Path
import numpy as np,json,time,argparse,importlib.util
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location("m209",H/"transfer_surface_contractions_to_volume_209_root.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def run(a):
 a.output.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 with np.load(a.quotient) as z:q={k:z[k] for k in z.files}
 p=v["points_world_m"];qp=q["points_world_m"]
 if len(p)<len(qp) or not np.array_equal(p[:len(qp)].view(np.uint32),qp.view(np.uint32)):raise ValueError("exact Float32 prefix mismatch")
 mp=np.arange(len(p));mp[:len(qp)]=q["input_node_quotient"];t=v["tetrahedra"];tt=mp[t];srt=np.sort(tt,axis=1);keep=~np.any(srt[:,1:]==srt[:,:-1],axis=1);changed=np.any(t!=tt,axis=1)&keep
 od,oq=m.m.measurements(p,t);nd,nq=m.m.measurements(p,tt[keep]);oldgood=m.m.passes(od,oq)[keep];newgood=m.m.passes(nd,nq)
 if np.any((od[keep]>0)&(nd<=0)):raise ValueError("new nonpositive cell")
 if np.any(oldgood&~newgood) or sum(~newgood)>sum(~m.m.passes(od,oq)):raise ValueError("quality regression")
 # Certify changed signs independently on exact lattice.
 rows=tt[changed];nodes=np.unique(rows);il={int(n):i for i,n in enumerate(nodes)};ip=m.m.integer_points(p[nodes])
 if any(m.m.exact_det(ip,[il[int(n)] for n in tet])<=0 for tet in rows):raise ValueError("exact changed-cell nonpositive")
 b=m.boundary(tt[keep]);expected=mp[m.boundary(t)];ss=np.sort(expected,axis=1);expected=expected[~np.any(ss[:,1:]==ss[:,:-1],axis=1)]
 if not np.array_equal(m.boundary_identity(b),m.boundary_identity(expected)):raise ValueError("quotient boundary mismatch")
 desired=q["desired_boundary_faces"];ak={tuple(sorted(map(int,f))) for f in b};dk={tuple(sorted(map(int,f))) for f in desired}
 out=a.output/"rebased-physical-volume-proposal.npz"
 np.savez_compressed(out,points_world_m=p,tetrahedra=tt[keep],material_indices=v["material_indices"][keep],boundary_faces=b,combinatorial_boundary_faces=b,desired_boundary_faces=desired,desired_missing_faces=np.asarray([f for f in desired if tuple(sorted(map(int,f))) not in ak],np.int64).reshape(-1,3),current_extra_boundary_faces=np.asarray([f for f in b if tuple(sorted(map(int,f))) not in dk],np.int64).reshape(-1,3),retained_input_cell_rows=np.flatnonzero(keep),input_node_quotient=mp)
 r={"status":"ENDPOINT_MAP_REBASE_PROPOSAL","physical_admission":False,"native_steps":0,"input":str(a.input),"input_sha256":m.sha(a.input),"quotient":str(a.quotient),"quotient_sha256":m.sha(a.quotient),"script_sha256":m.sha(Path(__file__)),"output":str(out),"output_sha256":m.sha(out),"input_cells":len(t),"output_cells":int(sum(keep)),"removed_degenerate_cells":int(sum(~keep)),"changed_retained_cells":int(sum(changed)),"nonpositive_before":int(sum(od<=0)),"nonpositive_after":int(sum(nd<=0)),"quality_failures_before":int(sum(~m.m.passes(od,oq))),"quality_failures_after":int(sum(~newgood)),"boundary_quotient_exact":True,"complete_boundary_matches_desired":bool(np.array_equal(m.boundary_identity(b),m.boundary_identity(desired))),"elapsed_s":time.monotonic()-tic}
 (a.output/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True);ap.add_argument("--quotient",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);run(ap.parse_args())
