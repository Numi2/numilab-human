"""Track current deep-fat witnesses across registered shoulder cavity construction stages."""
import ctypes,sys,json
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from certify_transported_material import measure

def run(job,h,library,out):
    current=h/'closed-native/native';z=np.load(job/'registered-boundary/registered-boundary.npz')
    witnesses=np.array(json.loads((job/'connected-layer-crossings/reference.json').read_bytes())['witnesses']['2-2'],dtype='u4')
    f=np.ascontiguousarray(z['D_node_joint'][z['D_face_nodes']],dtype='u4')
    e=np.loadtxt(current/'mesh.ele',skiprows=1,dtype='i4')[:,1:5]-1
    x=np.loadtxt(current/'mesh.node',skiprows=1,usecols=(1,2,3));c=np.load(h/'needle-conditioned/domain.npz');mask=np.full(len(c['x']),-1,dtype='i4')
    child=int(np.load(current/'material-source-indices.npz')['child_begin'])
    mask[c['t']]=e[child:]
    assert np.array_equal(mask[c['t']],e[child:]) and np.array_equal(x[mask[c['t']]].astype('f4'),c['x'][c['t']].astype('f4'))
    active=np.unique(f[np.unique(witnesses)]);inv={v:i for i,v in enumerate(mask) if v>=0};keys=np.array([inv[v] for v in active])
    arg=np.ctypeslib.ndpointer(flags='C_CONTIGUOUS');fn=ctypes.CDLL(str(library)).batch;fn.argtypes=[arg,arg,arg,ctypes.c_uint,arg];fn.restype=None
    stages=[];out.mkdir()
    for name in ['../positive-material','../material-quality','../regional-material','../expanded-material','../connected-cortex-material','collapsed-links','clear-stars','joined-field','loaded-patch','elastic-patch','needle-ring','needle-conditioned']:
        n=h/name/'domain.npz';s=np.load(n);v=s['x'];test=x.copy()
        sub=np.load(h/'needle-ring/subdivision.npz')['nodes'];frac=json.loads((h/'needle-ring/result.json').read_bytes())['fraction']
        if len(v)<len(c['x']):
            v=list(v)
            for a,b,i in sub:
                assert i==len(v);v.append((v[a]+frac*(v[b]-v[a])).astype('f4'))
            v=np.array(v)
        test[mask[mask>=0]]=v[mask>=0];hit=np.zeros(len(witnesses),'u1');fn(test,f,witnesses,len(hit),hit)
        stages.append(dict(stage=name,mesh=digest(n),hits=int(sum(hit)),hits_by_pair=hit.tolist(),source_displacement_um=np.round(np.linalg.norm(test[active]-x[active],axis=1)*1e6,4).tolist()))
    result=dict(emitted_nodes=active.tolist(),patch_nodes=keys.tolist(),witness_pairs=witnesses.tolist(),stages=stages)
    (out/'history.json').write_text(json.dumps(result,indent=2)+'\n');print([(v['stage'],v['hits']) for v in stages]);print('witness local IDs',keys)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
