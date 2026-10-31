"""Run one bounded strict-quality MILP only after oriented-facet coverage passes."""
from pathlib import Path
import hashlib, importlib.util, itertools, json, math, re, subprocess, sys, time
import numpy as np
H=Path(__file__).resolve().parents[1]
P=H.parent
ROOT=H/"complete-surface-cavity-223-root/run-001"
SHELL=ROOT/"component4-dp-cluster-11face-shell.npz"
SHELL_SHA="8c297fe0279603d0ee96042c3d1342e64db6e003b6bf87e994820861560c1d07"
SOLVER=H/"complete-surface-cavity-220-root/solve_oriented_cavity_milp_220.py"
QUALITY=H/"fixed_boundary_volume_flip_198_root.py"
MAX_POINTS=76;MAX_CANDIDATES=1_300_000;SECONDS=20;MATERIAL=15
OUT=ROOT/"milp-run-001"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
if sha(SHELL)!=SHELL_SHA:raise SystemExit("shell input hash mismatch")
if sha(QUALITY)!="650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0":raise SystemExit("native quality owner changed")
solver=load("solver220_component4_223",SOLVER)
with np.load(SHELL,allow_pickle=False) as z: shell={k:z[k] for k in z.files}
p=shell["points_world_m"];b=shell["faces"]
if len(p)>MAX_POINTS or math.comb(len(p),4)>MAX_CANDIDATES:raise SystemExit("bounded candidate cap exceeded")
if len(p)!=73 or len(b)!=138 or len(shell["desired_dp_faces_global"])!=11:raise SystemExit("shell dimensions or DP coverage changed")
OUT.mkdir(exist_ok=False)
vm_stat=subprocess.run(["vm_stat"],capture_output=True,text=True,check=True).stdout
page_match=re.search(r"page size of (\d+) bytes",vm_stat)
free_match=re.search(r"Pages free:\s+(\d+)",vm_stat)
if not page_match or not free_match:raise SystemExit("cannot read host memory before MILP")
free_bytes=int(page_match.group(1))*int(free_match.group(1))
if free_bytes<2*1024**3:
 report={"schema":"numi.skin.component4-dp-cluster-milp-223.v1","status":"NOT_RUN_RESOURCE_FLOOR","physical_admission":False,"native_steps":0,"shell_path":str(SHELL),"shell_sha256":sha(SHELL),"max_points":MAX_POINTS,"max_candidate_tetrahedra":MAX_CANDIDATES,"candidate_count_upper_bound":math.comb(len(p),4),"required_free_bytes":2*1024**3,"measured_free_bytes":free_bytes,"reason":"less than 2 GiB free host pages at lock acquisition"}
 (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2),flush=True)
 raise SystemExit(0)
coverage={"status":"NOT_REACHED","candidate_count_upper_bound":math.comb(len(p),4),"boundary_faces":len(b),"covered_oriented_boundary_faces":0,"uncovered_faces":[]}
real_milp=solver.milp
class CoverageFailure(Exception):pass
def guarded_milp(c,*,integrality,bounds,constraints,options):
 A=constraints.A.tocsc()
 n=len(p)
 tris=np.asarray(list(itertools.combinations(range(n),3)),np.int64)
 tc=solver.codes(tris,n)
 br=np.searchsorted(tc,solver.codes(b,n))
 if len(np.unique(br))!=len(br):raise CoverageFailure("duplicate boundary face rows")
 expected=solver.signs(b)
 target_by_row=np.zeros(A.shape[0],np.int8);target_by_row[br]=expected.astype(np.int8)
 covered=set()
 for start in range(0,A.nnz,1_000_000):
  stop=min(A.nnz,start+1_000_000)
  rows=A.indices[start:stop];wanted=target_by_row[rows]
  valid=(wanted!=0)&(A.data[start:stop]==wanted)
  covered.update(map(int,np.unique(rows[valid])))
 uncovered=[{"boundary_face_row":i,"local_face":b[i].tolist(),"required_sign":int(expected[i])} for i,row in enumerate(br) if int(row) not in covered]
 coverage.update(status="PASS" if not uncovered else "REJECT_UNCOVERABLE_ORIENTED_BOUNDARY_FACES",candidate_tetrahedra_after_native_quality=int(A.shape[1]),constraint_face_rows=int(A.shape[0]),covered_oriented_boundary_faces=int(len(b)-len(uncovered)),uncovered_faces=uncovered)
 if uncovered:raise CoverageFailure("oriented boundary face has no native-quality-admissible incident tet")
 return real_milp(c,integrality=integrality,bounds=bounds,constraints=constraints,options=options)
solver.milp=guarded_milp
start=time.monotonic();result=None;selected=None
try:
 result,selected=solver.solve(p,b,seconds=SECONDS,max_points=MAX_POINTS,material=MATERIAL)
except CoverageFailure as e:
 result={"status":"REJECT_PRE_MILP_BOUNDARY_COVERAGE","reason":str(e),"physical_admission":False,"runtime":0}
except Exception as e:
 result={"status":"CAVITY_SOLVE_FAILED","reason":type(e).__name__+": "+str(e),"physical_admission":False,"runtime":0}
finally:solver.milp=real_milp
report={"schema":"numi.skin.component4-dp-cluster-milp-223.v1","status":result.get("status","UNKNOWN"),"physical_admission":False,"native_steps":0,"shell_path":str(SHELL),"shell_sha256":sha(SHELL),"solver_path":str(SOLVER),"solver_sha256":sha(SOLVER),"native_quality_owner_path":str(QUALITY),"native_quality_owner_sha256":sha(QUALITY),"max_points":MAX_POINTS,"max_candidate_tetrahedra":MAX_CANDIDATES,"time_limit_seconds":SECONDS,"material_index":MATERIAL,"pre_milp_oriented_boundary_coverage":coverage,"solver_result":result,"elapsed_wall_s":time.monotonic()-start}
if selected is not None:
 global_tets=shell["global_point_rows"][selected]
 prop=OUT/"component4-dp-cluster-11face-milp-proposal.npz"
 np.savez_compressed(prop,points_world_m=p,tetrahedra=selected,global_tetrahedra=global_tets,material_indices=np.full(len(selected),MATERIAL,np.int32),boundary_faces=b,global_point_rows=shell["global_point_rows"],source_input_tet_rows=shell["source_input_tet_rows"],source_217_output_tet_rows=shell["source_217_output_tet_rows"],source_222_output_tet_rows=shell["source_222_output_tet_rows"],desired_dp_faces_global=shell["desired_dp_faces_global"])
 report["proposal_path"]=str(prop);report["proposal_sha256"]=sha(prop);report["proposed_tetrahedra"]=len(selected)
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
