"""Necessary boundary-face bound for the existing native tet quality gate.
For longest boundary edge L and apex altitude h, max edge E obeys
E^2 >= max(L^2, h^2+L^2/4). Thus h/E^3 <= sqrt(3)/(2 L^2).
Since det6 = |cross(face)| h, qtet <= sqrt(3)/2 qface.
An upper bound below the existing gate proves no apex can qualify that face.
This is a necessary test, not sufficient mesh or physics admission.
"""
from pathlib import Path
import numpy as np,json,argparse,hashlib
def sha(p):return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def bound(p,f):
 v=p[f].astype(float);cross=np.linalg.norm(np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=1)
 length2=np.maximum.reduce([np.sum((v[:,i]-v[:,j])**2,axis=1) for i,j in ((0,1),(0,2),(1,2))])
 q=cross/length2
 return q,np.sqrt(3)/2*q
def fixture():
 # The longest-edge upper bound is approached by a thin isosceles face.
 p=np.array([[-.5,0,0],[.5,0,0],[0,1.05e-5,0]],np.float32)
 q,u=bound(p,np.array([[0,1,2]]))
 assert q[0]>1e-5 and u[0]<1e-5
 rng=np.random.default_rng(256)
 apex=rng.normal(size=(10000,3));h=np.abs(apex[:,2]);E=np.maximum(1.,np.linalg.norm(apex[:,None,:]-p.astype(float)[None,:,:],axis=2).max(1))
 actual=float(np.linalg.norm(np.cross(p[1].astype(float)-p[0],p[2].astype(float)-p[0])))*h/E**3
 assert np.all(actual<=u[0]*(1+1e-12))
 near=np.array([0,0,np.sqrt(3)/2]);Em=max(1.,float(np.linalg.norm(near[None,:]-p.astype(float),axis=1).max()))
 assert abs(float(q[0])*near[2]/Em**3-float(u[0]))<1e-14
 return {"status":"PASS_NATIVE_FACE_NECESSARY_BOUND_CONTROL","surface_q_passes_but_no_native_apex_possible":True,"random_apex_checks":10000,"sharp_longest_edge_bound":True}
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,action="append",required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 reports=[]
 for f in a.input:
  with np.load(f) as z:p=z["points_world_m"];faces=z["faces"]
  q,u=bound(p,faces);bad=np.flatnonzero(u<1e-5*(1-1e-12))
  reports.append({"path":str(f),"sha256":sha(f),"faces":len(faces),"surface_q_below_1e5":int(sum(q<1e-5)),"proven_no_native_apex_faces":len(bad),"minimum_tet_q_upper_bound":float(u.min()),"witnesses":[{"face_row":int(i),"face":faces[i].tolist(),"surface_q":float(q[i]),"tet_q_upper_bound":float(u[i])} for i in bad]})
 r={"status":"PASS_NECESSARY_BOUND_CENSUS","physical_admission":False,"native_steps":0,"source_sha256":sha(Path(__file__)),"control":fixture(),"surfaces":reports}
 (a.output/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
