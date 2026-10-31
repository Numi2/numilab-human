"""Emit a role-D diagnostic from a connected physical-volume proposal.
Original volume coordinate identities map to original D point rows; inferred
current positions are emitted explicitly. Typed ancestry is intentionally pending.
"""
from pathlib import Path
import numpy as np,json,argparse,hashlib,time
def sha(p):return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def key(x):return np.asarray(x,np.float32).view(np.uint32).tobytes()
def run(a):
 a.output.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
 with np.load(a.original_volume) as z:original=z["points_world_m"]
 with np.load(a.volume) as z:p=z["points_world_m"];b=z["desired_boundary_faces"]
 with np.load(a.d_baseline) as z:dp=z["points_world_m"].astype(np.float32);df=z["faces"]
 with np.load(a.i_baseline) as z:nI=len(z["faces"])
 if len(b)!=nI+len(df)+90:raise ValueError("desired boundary role counts do not match authenticated D/I and90wall facets")
 if len(original)>len(p):raise ValueError("source volume prefix missing")
 look={key(x):i for i,x in enumerate(dp)}
 if len(look)!=len(dp):raise ValueError("ambiguous D coordinate identity")
 faces=b[nI:nI+len(df)][:,[0,2,1]]
 active=np.unique(faces);mp={int(j):look[key(original[j])] for j in active}
 outp=dp.copy()
 for j,k in mp.items():outp[k]=p[j]
 outf=np.asarray([[mp[int(j)] for j in f] for f in faces],np.int64)
 moved=np.flatnonzero(np.any(outp.view(np.uint32)!=dp.view(np.uint32),axis=1))
 changed=np.flatnonzero(np.any(outf!=df,axis=1)|np.any(np.isin(outf,moved),axis=1)|np.any(np.isin(df,moved),axis=1))
 candidate=a.output/"candidate-D-physical-position-diagnostic.npz"
 np.savez_compressed(candidate,points_world_m=outp,points_world_m_f32=outp,faces=outf,point_output_to_parent_rows=np.arange(len(dp),dtype=np.int64),face_output_to_parent_rows=np.arange(len(df),dtype=np.int64),changed_parent_face_rows=changed,changed_point_rows=moved,parent_candidate_sha256=np.array([sha(a.d_baseline)]),source_volume_sha256=np.array([sha(a.volume)]))
 r={"status":"ROLE_D_POSITION_DIAGNOSTIC_TYPED_ANCESTRY_PENDING","physical_admission":False,"native_steps":0,"script_sha256":sha(Path(__file__)),"inputs":{str(p):sha(p) for p in (a.original_volume,a.volume,a.d_baseline,a.i_baseline)},"candidate":str(candidate),"candidate_sha256":sha(candidate),"changed_point_rows":moved.tolist(),"changed_face_rows":changed.tolist(),"changed_points":len(moved),"changed_faces":len(changed),"maximum_point_displacement_m":float(np.linalg.norm(outp.astype(float)-dp.astype(float),axis=1).max()),"elapsed_s":time.monotonic()-tic}
 (a.output/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({k:v for k,v in r.items() if k not in ("changed_point_rows","changed_face_rows")},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--original-volume",type=Path,required=True);ap.add_argument("--volume",type=Path,required=True);ap.add_argument("--d-baseline",type=Path,required=True);ap.add_argument("--i-baseline",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);run(ap.parse_args())
