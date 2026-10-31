#!/usr/bin/env python3
"""Exact Float32-lattice necessary apex bound. Native thresholds unchanged."""
from pathlib import Path
import numpy as np,argparse,json,hashlib,importlib.util,sys,time
H=Path(__file__).resolve().parent
p=H/'fixed_boundary_volume_flip_198_root.py';s=importlib.util.spec_from_file_location('owner198for263',p);M=importlib.util.module_from_spec(s);sys.modules[s.name]=M;s.loader.exec_module(M)
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def bound_terms(points,face,gate):
 a,b,c=[points[int(i)] for i in face]
 u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)];w=[c[i]-b[i] for i in range(3)]
 n=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
 n2=sum(x*x for x in n);L2=max(sum(x*x for x in d) for d in (u,v,w));gn,gd=gate
 return 3*n2*gd*gd,4*L2*L2*gn*gn
def fixture():
 g=float(np.float32(1e-5)).as_integer_ratio()
 p=np.asarray([[-.5,0,0],[.5,0,0],[0,1.05e-5,0]],np.float32);lo,hi=bound_terms(M.integer_points(p),[0,1,2],g);assert lo<hi
 p[2,1]=1.3e-5;lo,hi=bound_terms(M.integer_points(p),[0,1,2],g);assert lo>hi
 return {'status':'PASS_EXACT_GATE_AND_THIN_FACE_CONTROLS','native_gate_float32_hex':float(np.float32(1e-5)).hex(),'old_surface_gate_can_pass_impossible_face':True}
def main(a):
 tic=time.monotonic();a.output.mkdir(parents=True,exist_ok=False);control=fixture();surfaces=[];g=float(np.float32(1e-5)).as_integer_ratio()
 for path in a.input:
  with np.load(path) as z:p=np.asarray(z['points_world_m'],np.float32);f=np.asarray(z['faces'],np.int64)
  ip=M.integer_points(p);w=[]
  for i,tri in enumerate(f):
   lo,hi=bound_terms(ip,tri,g)
   if lo<hi:w.append({'face_row':i,'face':tri.tolist(),'strict_integer_margin':str(hi-lo)})
  surfaces.append({'path':str(path),'sha256':sha(path),'faces':len(f),'all_faces_checked_exactly':True,'proven_no_native_apex_faces':len(w),'witnesses':w})
 r={'status':'PASS_EXACT_NATIVE_APEX_NECESSARY_BOUND_CENSUS','physical_admission':False,'native_steps':0,'source_sha256':sha(Path(__file__)),'integer_lattice_owner_sha256':sha(Path(M.__file__)),'native_gate_rational':list(g),'control':control,'surfaces':surfaces,'elapsed_s':time.monotonic()-tic}
 (a.output/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({**r,'surfaces':[{k:v for k,v in z.items() if k!='witnesses'} for z in surfaces]},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,action='append',required=True);ap.add_argument('--output',type=Path,required=True);main(ap.parse_args())
