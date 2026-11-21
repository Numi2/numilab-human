"""Compare the actual native constitutive kernel with independent double-precision nominal stress."""
from pathlib import Path
import json,subprocess,sys,tempfile
import numpy as np
from scipy.spatial.transform import Rotation
from material_context import sha
def run(engine,lib,out):
    tags=['undeformed','translation','rigid_rotation','tiny_dilation','simple_shear','biaxial_stretch']
    L=.02;x=np.vstack((np.zeros(3),L*np.eye(3)));H=np.zeros((6,3,3));H[2]=Rotation.from_rotvec([.1,-.2,.15]).as_matrix()-np.eye(3)
    H[3]=np.diag([2e-8,1e-8,-1e-8]);H[4,0,1]=.11;H[5]=np.diag([.08,.03,-.04]);du=np.zeros((6,4,4),'f4');du[:,:,:3]=x@H.transpose(0,2,1);du[1,:,:3]=[.2,-.1,.12]
    t=np.arange(24,dtype='u4').reshape(-1,4);inverse=np.zeros((6,3,4),'f4');inverse[:,:,:3]=np.eye(3)/L;volume=np.full(6,L**3/6,'f4')
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'fixture'
        with path.open('wb') as f:
            for a in (np.array([6],'u4'),t,inverse,du,volume):a.tofile(f)
        from io import StringIO
        actual=np.loadtxt(StringIO(subprocess.check_output([engine,lib,path],text=True)))
    record={}
    for i,name in enumerate(tags):
        dd=du[i,1:,:3].astype(float)-du[i,:1,:3].astype(float);h=(dd/L).T;F=np.eye(3)+h;J=np.linalg.det(F);ln=np.log(J)
        P=12000*F+(120000*ln-12000)*np.linalg.inv(F).T
        grads=np.vstack((-np.ones(3),np.eye(3)))/L;force=float(volume[i])*grads@P.T;observed=actual[i,:12].reshape(4,3)
        err=np.max(np.abs(observed-force));scale=np.max(np.abs(force));tol=(2e-11 if i in (0,1,3) else 2e-6)+3e-5*scale
        if err>tol or abs(actual[i,12]-J)>4e-6:raise ValueError((name,err,tol,J,actual[i,12]))
        record[name]=dict(max_force_error_N=float(err),peak_reference_force_N=float(scale),determinant_error=abs(J-actual[i,12]),net_force_N=float(np.linalg.norm(observed.sum(0))))
    out.write_text(json.dumps(dict(kernel='nm_pelvic_bulk_corners',fixture_source_sha256=sha(Path(__file__)),metallib_sha256=sha(lib),cases=record),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
