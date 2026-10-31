#!/usr/bin/env python3
"""Necessary joint native volume/shape bound; does not change either native gate.
If face area2 is A, longest edge L, and apex altitude h, then
det6=A*h and E²>=max(L²,h²+L²/4).
Subject to det6>=dmin, the best envelope occurs at
h=max(dmin/A, sqrt(3)*L/2).
A failed upper bound proves impossibility; a pass is not mesh admission.
"""
from pathlib import Path
from fractions import Fraction
import numpy as np,argparse,json,hashlib,importlib.util,sys,time
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('owner198for272',H/'fixed_boundary_volume_flip_198_root.py');M=importlib.util.module_from_spec(s);sys.modules[s.name]=M;s.loader.exec_module(M)
# These permissive bounds are <= the existing native/CPU cutoffs.
GATE=float(np.float32(1e-5));GNUM,GDEN=GATE.as_integer_ratio()
DMIN=Fraction(6,10**18);D2=DMIN**2*2**894;DA,DB=D2.numerator,D2.denominator
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def score(points,faces):
 xyz=np.asarray(points,np.float32)[np.asarray(faces,np.int64)].astype(float)
 a=xyz[:,1]-xyz[:,0];b=xyz[:,2]-xyz[:,0];c=xyz[:,2]-xyz[:,1]
 area=np.linalg.norm(np.cross(a,b),axis=1);L2=np.maximum.reduce([np.sum(x*x,axis=1) for x in (a,b,c)])
 h=np.maximum(np.divide(float(DMIN),area,out=np.full_like(area,np.inf),where=area>0),np.sqrt(.75*L2))
 E2=np.maximum(L2,h*h+.25*L2)
 out=np.divide(area*h,E2**1.5,out=np.zeros_like(area),where=np.isfinite(h)&(E2>0))
 return np.nan_to_num(out,nan=0.,posinf=0.,neginf=0.)
def exact_terms(ip,face):
 a,b,c=[ip[int(i)] for i in face];u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)];w=[c[i]-b[i] for i in range(3)]
 n=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);n2=sum(x*x for x in n);L2=max(sum(x*x for x in d) for d in (u,v,w))
 normalized_lhs=3*n2*GDEN*GDEN;normalized_rhs=4*L2*L2*GNUM*GNUM
 if normalized_lhs<normalized_rhs:return 'normalized_shape',normalized_rhs-normalized_lhs
 if 4*DA>3*DB*n2*L2:
  lhs=64*DA*DB*DB*n2**3*GDEN*GDEN;rhs=GNUM*GNUM*(4*DA+DB*n2*L2)**3
  if lhs<rhs:return 'joint_volume_and_shape',rhs-lhs
 return None,None
def fixture():
 p=np.array([[0,0,0],[1e-7,0,0],[.5e-7,.8660254e-7,0]],np.float32);f=np.array([[0,1,2]],np.int64)
 ip=M.integer_points(p);reason,gap=exact_terms(ip,f[0]);assert reason=='joint_volume_and_shape'
 A=np.linalg.norm(np.cross(p[1].astype(float)-p[0],p[2].astype(float)-p[0]));L=max(np.linalg.norm(p[i].astype(float)-p[j]) for i,j in ((0,1),(0,2),(1,2)))
 assert np.sqrt(3)/2*A/L**2>GATE and score(p,f)[0]<GATE
 # Enforce the minimum volume on 10k random apexes; the envelope bounds each.
 rng=np.random.default_rng(272);apex=rng.normal(size=(10000,3))*1e-3;apex[:,2]=np.maximum(abs(apex[:,2]),float(DMIN)/A)
 edge=np.linalg.norm(apex[:,None,:]-p.astype(float)[None,:,:],axis=2).max(1);E=np.maximum(edge,L);q=A*apex[:,2]/E**3
 assert np.all(q<=score(p,f)[0]*(1+1e-12))
 p*=100;assert exact_terms(M.integer_points(p),f[0])[0] is None
 return {'status':'PASS_JOINT_VOLUME_SHAPE_COUNTEREXAMPLE_AND_ENVELOPE','normalized_bound_pass_does_not_imply_joint_pass':True,'random_minimum_volume_apexes_checked':10000,'rescaled_feasible_necessary_gate':True}
def run(a):
 tic=time.monotonic();a.output.mkdir(parents=True,exist_ok=False);control=fixture();results=[]
 for path in a.input:
  with np.load(path) as z:p=np.asarray(z['points_world_m'],np.float32);f=np.asarray(z['faces'],np.int64)
  ip=M.integer_points(p);w=[];types={}
  for i,face in enumerate(f):
   reason,gap=exact_terms(ip,face)
   if reason:w.append({'face_row':i,'face':face.tolist(),'reason':reason,'strict_integer_margin':str(gap)});types[reason]=types.get(reason,0)+1
  results.append({'path':str(path),'sha256':sha(path),'faces':len(f),'all_faces_exactly_checked':True,'proven_no_joint_native_apex_faces':len(w),'counts':types,'witnesses':w})
 report={'status':'PASS_EXACT_JOINT_NATIVE_APEX_NECESSARY_BOUND_CENSUS','physical_admission':False,'native_steps':0,'source_sha256':sha(Path(__file__)),'native_shape_gate_permissive_rational':[GNUM,GDEN],'minimum_det6_permissive_rational':[DMIN.numerator,DMIN.denominator],'control':control,'surfaces':results,'elapsed_s':time.monotonic()-tic}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({**report,'surfaces':[{k:v for k,v in r.items() if k!='witnesses'} for r in results]},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,action='append',required=True);ap.add_argument('--output',type=Path,required=True);run(ap.parse_args())
