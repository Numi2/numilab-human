"""Polish rounded material vertices using exact emitted cell measures and all known surface witnesses."""
import json,sys,ctypes
from pathlib import Path
import numpy as np
from certify_transported_material import measure
from compile_connected_joint import digest
def run(job,h,lib,source,out):
    domain=h/'closed-native/native';x=np.loadtxt(domain/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype(float);s=np.load(source/'candidate.npz');y=x.copy();y[s['nodes']]=s['positions']
    e=np.loadtxt(domain/'mesh.ele',skiprows=1,dtype='i4');t=e[:,1:5]-1;m=e[:,5];mass=np.fromfile(domain/'cell-mass.f64','<f8')
    z=np.load(job/'registered-boundary/registered-boundary.npz');f=np.ascontiguousarray(z['D_node_joint'][z['D_face_nodes']],'u4')
    w=np.array(json.loads((job/'connected-layer-crossings/reference.json').read_bytes())['witnesses']['2-2'],dtype='u4')
    arg=np.ctypeslib.ndpointer(flags='C_CONTIGUOUS');fn=ctypes.CDLL(str(lib)).batch;fn.argtypes=[arg,arg,arg,ctypes.c_uint,arg];fn.restype=None
    lock=np.unique(np.r_[z['O_node_joint'],z['I_node_joint'],np.load(h/'cortical-registration/registered-cortical-facets.npz')['cell_side_material_bank_nodes'][:,-3:].ravel()])
    def qual(X,T):
        d,l=measure(X,T);return d/np.maximum(6e-18,1e-5*l)
    qi=qual(y,t);bad=np.flatnonzero(qi<1);history=[];print('start',bad,qi[bad],flush=True)
    for it in range(14):
        if not len(bad):break
        best=None
        for node in np.setdiff1d(np.unique(t[bad]),lock):
            rows=np.flatnonzero((t==node).any(1));tet=t[rows];p=y[node].copy();unit=np.maximum(np.abs(np.spacing(p.astype('f4'))),1e-11);q=qual(y,tet);base=np.maximum(1.00008-q,0.).sum()
            for axis in range(3):
                for k in [-256,-64,-16,-4,-1,1,4,16,64,256]:
                    y[node,axis]=float(np.float32(p[axis]+k*unit[axis]));new=qual(y,tet);value=base-float(np.maximum(1.00008-new,0.).sum())
                    if value>1e-10 and not np.any((new<1)&(q>=1)):
                        hit=np.zeros(len(w),'u1');fn(y,f,w,len(hit),hit)
                        if not hit.any() and (best is None or (value,-abs(k))>best[:2]):best=(value,-abs(k),node,axis,y[node,axis])
                    y[node,axis]=p[axis]
        if best is None:break
        gain,_,node,axis,value=best;y[node,axis]=value;rows=np.flatnonzero((t==node).any(1));qi[rows]=qual(y,t[rows]);bad=np.flatnonzero(qi<1);history.append((int(node),int(axis),gain,len(bad)))
    ids=np.flatnonzero(np.any(x!=y,axis=1));rows=np.flatnonzero(np.isin(t,ids).any(1));d,l=measure(y,t[rows]);rho=mass[rows]*6/d;ok=(rho>=np.where(m[rows]==15,900.,850.)*(1-1e-10))&(rho<=np.where(m[rows]==15,1100.,950.)*(1+1e-10))
    out.mkdir();np.savez_compressed(out/'candidate.npz',nodes=ids,positions=y[ids].astype('f4'));r=dict(polish=history,invalid_cells=bad.tolist(),updated_cells=rows.tolist(),density_failures=rows[~ok].tolist(),max_move_um=float(np.linalg.norm(y[ids]-x[ids],axis=1).max()*1e6),minimum_quality_ratio=float(qi.min()),candidate=digest(out/'candidate.npz'));(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print('bad',bad,'densities',sum(~ok),history,flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
