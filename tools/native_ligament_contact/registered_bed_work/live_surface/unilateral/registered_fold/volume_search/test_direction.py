"""Exercise the production Metal search kernel and independently recompute physical trial volumes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation
from material_context import sha

def test(root,shader=None,mode=None):
    rng=np.random.default_rng(607213)
    N=3072
    orientation=Rotation.random(N,random_state=rng).as_matrix()
    reference=orientation@np.broadcast_to(np.array([[.02,.004,.001],[0,.014,.002],[0,0,.003]]),(N,3,3))
    X=np.zeros((N,4,3));X[:,1:]=reference.transpose(0,2,1)
    X+=rng.uniform(-1.,1.,(N,1,3));X=X.astype('f4').astype(float)
    D=(X[:,1:]-X[:,:1]).transpose(0,2,1)
    inv=np.linalg.inv(D).astype('f4')
    F=Rotation.from_rotvec(rng.normal(0,.08,(N,3))).as_matrix()
    F*=rng.uniform(.92,1.08,(N,1,1))
    Y=X.copy();Y[:,1:]+=((F-np.eye(3))@D).transpose(0,2,1)
    u=(Y-X).astype('f4')
    kind=np.arange(N)%3
    # Trial targets alternate compression, dilation and an interior rigid-like change.
    floor=.35*1.41864 if mode=='release' else .35
    target=np.choose(kind,[.45 if mode=='release' else .2,2.7,1.03])
    a=target/np.linalg.det(F)
    H=F.copy()
    H[:,:,2]*=a[:,None]
    v=np.zeros_like(X);v[:,1:]=((H-F)@D).transpose(0,2,1)/.002
    v+=rng.uniform(-.02,.02,(N,1,3));v=v.astype('f4')
    mass=rng.uniform(.0001,.009,(N,4)).astype('f4')
    fixed=np.arange(N)%7==0;reference_mode=np.arange(N)%2
    if mode=='shared':
        # Preserve each tetrahedron's edges while translating its reference to
        # the same Float32 point. Refresh the inverse using the emitted edges.
        X[1::2]+=X[::2,:1]-X[1::2,:1];X=X.astype('f4').astype(float)
        X[1::2,0]=X[::2,0];u[1::2,0]=u[::2,0];v[1::2,0]=v[::2,0];mass[1::2,0]=mass[::2,0]
        fixed[:]=False;reference_mode=(np.arange(N)//2)%2
        inv=np.linalg.inv((X[:,1:]-X[:,:1]).transpose(0,2,1)).astype('f4')
    data=np.column_stack([X.reshape(N,-1),u.reshape(N,-1),v.reshape(N,-1),mass,
        inv.reshape(N,-1),fixed,reference_mode]).astype('<f4')
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);data.tofile(tmp/'in')
        if shader is None:shader=root/'build/NumiMatter.metallib'
        proc=subprocess.run([root/'test-kernel',shader,tmp/'in',tmp/'out']+([mode] if mode else []),capture_output=True,text=True)
        if proc.returncode:raise RuntimeError((proc.returncode,proc.stderr))
        out=np.fromfile(tmp/'out','<f4').reshape(N,4,4)
    if mode=='shared' and not np.array_equal(out[:,0,:3],v[:,0]):
        raise ValueError('Shared Newton direction was overwritten')
    adjust=out[:,:,:3].astype(float)-v
    def volumes(dir):
        y=X+u+.002*dir
        return np.linalg.det((y[:,1:]-y[:,:1]).transpose(0,2,1)@inv)
    J=volumes(out[:,:,:3]);J0=volumes(v)
    projected=(kind!=2)&~fixed
    if not (np.isfinite(out).all() and (out[:,:,3]==13).all() and np.array_equal(out[~projected,:,:3],v[~projected])
            and np.all((J[projected]>floor)&(J[projected]<2.5))):raise ValueError(('Volume or identity regression',np.where(projected&((J<=floor)|(J>=2.5)))[0][:8]))
    impulse=mass[:,:,None]*adjust
    relative=np.linalg.norm(impulse.sum(1),axis=1)/np.maximum(1e-24,np.linalg.norm(impulse,axis=2).sum(1))
    if relative.max()>2e-5:raise ValueError(('Unbalanced mass metric adjustment',float(relative.max()),int(relative.argmax()),impulse[relative.argmax()].tolist(),v[relative.argmax()].tolist(),adjust[relative.argmax()].tolist()))
    return dict(device=proc.stdout.strip(),kernel_library=sha(shader),cases=N,shared_pairs=N//2 if mode=='shared' else 0,changed=int(projected.sum()),unchanged=int((~projected).sum()),
        corrected_J_limits=[float(J[projected].min()),float(J[projected].max())],
        maximum_relative_linear_momentum_change=float(relative.max()),unconstrained_input_volume_limits=[float(J0.min()),float(J0.max())])
if __name__=='__main__':print(json.dumps(test(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve() if len(sys.argv)>2 else None,sys.argv[3] if len(sys.argv)>3 else None),indent=2))
