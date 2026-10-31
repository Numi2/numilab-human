"""One bounded strict-native-quality MILP for the remaining 5-face Skin DP cluster."""
from pathlib import Path
import hashlib, importlib.util, itertools, json, math, re, subprocess, sys, time
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
ROOT=H/"complete-surface-cavity-224-root/run-004-dp-clusters-shells"
SHELL=ROOT/"component4-dp-cluster-5face-shell.npz"
SHELL_SHA="677c43027b3b2b695e8ade1b11c3d2b601aa22f1087ea84c7a6b066ea75a8868"
SHELL_REPORT=ROOT/"report.json"
SOLVER=H/"complete-surface-cavity-220-root/solve_oriented_cavity_milp_220.py"
QUALITY=H/"fixed_boundary_volume_flip_198_root.py"
QUALITY_SHA="650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"
MAX_POINTS=90
MAX_CANDIDATES=2_300_000
SECONDS=20
MATERIAL=15
OUT=ROOT/"cluster5-milp-run-001"
def sha(p): return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 return module
if sha(SHELL)!=SHELL_SHA: raise SystemExit("shell hash mismatch")
if sha(QUALITY)!=QUALITY_SHA: raise SystemExit("quality owner hash mismatch")
if OUT.exists(): raise SystemExit("refusing to overwrite "+str(OUT))
shell_report=json.loads(SHELL_REPORT.read_text())
cluster=shell_report["clusters"][0]
if cluster.get("status")!="PASS_CLOSED_DP_CLUSTER_SHELL_PREPARATION" or cluster.get("shell_sha256")!=SHELL_SHA or cluster.get("target_face_count")!=5: raise SystemExit("shell preflight pin/status mismatch")
solver=load("solver220_cluster5_224",SOLVER)
with np.load(SHELL,allow_pickle=False) as z: shell={k:z[k] for k in z.files}
p=shell["points_world_m"];b=shell["faces"]
upper=math.comb(len(p),4)
if len(p)>MAX_POINTS or upper>MAX_CANDIDATES: raise SystemExit("bounded candidate cap exceeded")
if len(p)!=87 or len(b)!=166 or len(shell["desired_dp_faces_global"])!=5: raise SystemExit("shell dimensions or DP coverage changed")
OUT.mkdir(exist_ok=False)
vm_stat=subprocess.run(["vm_stat"],capture_output=True,text=True,check=True).stdout
page_match=re.search(r"page size of (\d+) bytes",vm_stat);free_match=re.search(r"Pages free:\s+(\d+)",vm_stat)
if not page_match or not free_match: raise SystemExit("cannot read host memory before MILP")
free_bytes=int(page_match.group(1))*int(free_match.group(1))
coverage={"status":"NOT_REACHED","candidate_count_upper_bound":upper,"boundary_faces":len(b),"covered_oriented_boundary_faces":0,"uncovered_faces":[]}
if free_bytes<2*1024**3:
 report={"schema":"numi.skin.component4-dp-cluster-5face-milp-224.v1","status":"NOT_RUN_RESOURCE_FLOOR","physical_admission":False,"native_steps":0,"shell_path":str(SHELL),"shell_sha256":sha(SHELL),"max_points":MAX_POINTS,"max_candidate_tetrahedra":MAX_CANDIDATES,"candidate_count_upper_bound":upper,"required_free_bytes":2*1024**3,"measured_free_bytes":free_bytes,"reason":"less than 2 GiB free host pages at lock acquisition"}
 (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2),flush=True);raise SystemExit(0)
real_milp=solver.milp
class CoverageFailure(Exception): pass
def guarded_milp(c,*,integrality,bounds,constraints,options):
 A=constraints.A.tocsc();n=len(p)
 tris=np.asarray(list(itertools.combinations(range(n),3)),np.int64)
 triangle_codes=solver.codes(tris,n);boundary_rows=np.searchsorted(triangle_codes,solver.codes(b,n))
 if len(np.unique(boundary_rows))!=len(boundary_rows): raise CoverageFailure("duplicate boundary face rows")
 expected=solver.signs(b);target=np.zeros(A.shape[0],np.int8);target[boundary_rows]=expected.astype(np.int8)
 covered=set()
 for start in range(0,A.nnz,1_000_000):
  stop=min(A.nnz,start+1_000_000);rows=A.indices[start:stop];wanted=target[rows]
  valid=(wanted!=0)&(A.data[start:stop]==wanted)
  covered.update(map(int,np.unique(rows[valid])))
 uncovered=[{"boundary_face_row":i,"local_face":b[i].tolist(),"required_sign":int(expected[i])} for i,row in enumerate(boundary_rows) if int(row) not in covered]
 coverage.update(status="PASS" if not uncovered else "REJECT_UNCOVERABLE_ORIENTED_BOUNDARY_FACES",candidate_tetrahedra_after_native_quality=int(A.shape[1]),constraint_face_rows=int(A.shape[0]),covered_oriented_boundary_faces=int(len(b)-len(uncovered)),uncovered_faces=uncovered)
 if uncovered: raise CoverageFailure("oriented boundary face has no native-quality-admissible incident tet")
 return real_milp(c,integrality=integrality,bounds=bounds,constraints=constraints,options=options)
solver.milp=guarded_milp
start=time.monotonic();result=None;selected=None
try:
 result,selected=solver.solve(p,b,seconds=SECONDS,max_points=MAX_POINTS,material=MATERIAL)
except CoverageFailure as e:
 result={"status":"REJECT_PRE_MILP_BOUNDARY_COVERAGE","reason":str(e),"physical_admission":False,"runtime":0}
except Exception as e:
 result={"status":"CAVITY_SOLVE_FAILED","reason":type(e).__name__+": "+str(e),"physical_admission":False,"runtime":0}
finally: solver.milp=real_milp
report={"schema":"numi.skin.component4-dp-cluster-5face-milp-224.v1","status":result.get("status","UNKNOWN"),"physical_admission":False,"native_steps":0,"shell_path":str(SHELL),"shell_sha256":sha(SHELL),"shell_report_sha256":sha(SHELL_REPORT),"solver_path":str(SOLVER),"solver_sha256":sha(SOLVER),"native_quality_owner_path":str(QUALITY),"native_quality_owner_sha256":sha(QUALITY),"max_points":MAX_POINTS,"max_candidate_tetrahedra":MAX_CANDIDATES,"candidate_count_upper_bound":upper,"free_bytes_at_lock_acquisition":free_bytes,"time_limit_seconds":SECONDS,"material_index":MATERIAL,"pre_milp_oriented_boundary_coverage":coverage,"solver_result":result,"elapsed_wall_s":time.monotonic()-start}
if selected is not None:
 global_tets=shell["global_point_rows"][selected]
 proposal=OUT/"component4-dp-cluster-5face-milp-proposal.npz"
 np.savez_compressed(proposal,points_world_m=p,tetrahedra=selected,global_tetrahedra=global_tets,material_indices=np.full(len(selected),MATERIAL,np.int32),boundary_faces=b,global_point_rows=shell["global_point_rows"],source_input_tet_rows=shell["source_input_tet_rows"],source_217_output_tet_rows=shell["source_217_output_tet_rows"],source_222_output_tet_rows=shell["source_222_output_tet_rows"],desired_dp_faces_global=shell["desired_dp_faces_global"])
 report["proposal_path"]=str(proposal);report["proposal_sha256"]=sha(proposal);report["proposed_tetrahedra"]=int(len(selected))
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2),flush=True)
