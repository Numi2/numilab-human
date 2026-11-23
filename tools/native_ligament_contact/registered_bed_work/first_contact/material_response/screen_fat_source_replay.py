"""Replay the pre-fold source displacement on the actual emitted volume with fixed topology."""
import ctypes,sys,json
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from certify_transported_material import measure

def run(job,h,library,out):
    current=h/'closed-native/native'
    z=np.load(job/'registered-boundary/registered-boundary.npz')
    w=np.array(json.loads((job/'connected-layer-crossings/reference.json').read_bytes())['witnesses']['2-2'],dtype='u4')
    f=np.ascontiguousarray(z['D_node_joint'][z['D_face_nodes']],dtype='u4')
    e=np.loadtxt(current/'mesh.ele',skiprows=1,dtype='i4')[:,1:5]-1
    x=np.loadtxt(current/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8')
    c=np.load(h/'needle-conditioned/domain.npz');source=np.load(h.parent/'positive-material/domain.npz')
    mapper=np.full(len(c['x']),-1,dtype='i4');child=int(np.load(current/'material-source-indices.npz')['child_begin']);mapper[c['t']]=e[child:]
    assert np.array_equal(mapper[c['t']],e[child:]);rev={v:i for i,v in enumerate(mapper) if v>=0}
    v=np.unique(f[np.unique(w)]);local=np.array([rev[i] for i in v]);delta=source['x'][local].astype(float)-x[v]
    keep=(delta!=0).any(1);v=v[keep];local=local[keep];delta=delta[keep]
    touched=np.flatnonzero(np.isin(e,v).any(1));t=e[touched]
    d0,q0=measure(x,t);mass=np.fromfile(current/'cell-mass.f64','<f8')[touched];mat=np.loadtxt(current/'mesh.ele',skiprows=1,usecols=5,dtype='i4')[touched];lo=np.where(mat==15,900.,850.);hi=np.where(mat==15,1100.,950.)
    arg=np.ctypeslib.ndpointer(flags='C_CONTIGUOUS');fn=ctypes.CDLL(str(library)).batch;fn.argtypes=[arg,arg,arg,ctypes.c_uint,arg];fn.restype=None
    cortical=np.unique(c['facets'][c['markers']==116]);table=[]
    for group,allowed in [('free',~np.isin(local,cortical)),('all',np.ones(len(v),bool))]:
        for frac in np.linspace(0,1,21):
            y=x.copy();y[v[allowed]]=(x[v[allowed]]+frac*delta[allowed]).astype('f4');hit=np.zeros(len(w),'u1');fn(y,f,w,len(hit),hit)
            det,length=measure(y,t);rho=6*mass/det;valid=(det>6e-18)&(det>=length*1e-5);dense=(rho>=lo*(1-1e-10))&(rho<=hi*(1+1e-10));table.append(dict(group=group,fraction=round(float(frac),3),crossings=int(hit.sum()),inverted=int((det<=0).sum()),quality_fail=int((~valid).sum()),density_fail=int((~dense).sum())))
    out.mkdir();np.savez_compressed(out/'source-displacements.npz',nodes=v,local=local,delta=delta,incident=touched)
    s=dict(cortical=local[np.isin(local,cortical)].tolist(),local=local.tolist(),displacement_um=np.linalg.norm(delta,axis=1).__mul__(1e6).tolist(),trials=table);(out/'gate.json').write_text(json.dumps(s,indent=2)+'\n')
    print([(i['group'],i['fraction'],i['crossings'],i['inverted'],i['quality_fail'],i['density_fail']) for i in table if i['fraction'] in (0,.25,.5,.75,1)])
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
