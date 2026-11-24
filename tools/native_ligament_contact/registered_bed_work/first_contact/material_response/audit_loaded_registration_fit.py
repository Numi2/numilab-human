"""Use the native cell-interval checker and exact boundary predicates on each registration proposal."""
import ctypes,json,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from layer_contacts import run as check_layers

def run(job,trial,lib,layers):
    mat=job/'interval-certified-scene/material'
    x,y=[np.loadtxt(mat/k,skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8') for k in ('mesh.node','current.node')]
    g=np.ascontiguousarray(np.loadtxt(mat/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1)
    u=np.load(trial/'candidate-positions.npz');ids=u['nodes'];y[ids]=u['positions']
    q=json.loads((job/'executable-fat-pose/result.json').read_bytes())
    if digest(lib)!=q['interval_checker'] or q['updated']!=digest(mat/'current.node'):raise ValueError('Unknown checker or parent')
    fn=ctypes.CDLL(str(lib)).scanCellIntervals;f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    fn.argtypes=[f,f,ctypes.c_size_t,i,ctypes.c_size_t,f];values=np.empty((len(g),3))
    if fn(x,y,len(x),g,len(g),values)!=0:raise ValueError('Cell check failed')
    margins=np.minimum(values[:,0]-values[:,1]-float(np.float32(.35)),2.5-values[:,0]-values[:,1])
    fail=np.flatnonzero((~np.isfinite(margins))|(margins<=0))
    report=dict(cells=len(g),moved=len(ids),failed_intervals=fail.tolist(),minimum_margin=float(np.min(margins)),checker=digest(lib),layer_checker=digest(layers),fit=digest(trial/'candidate-positions.npz'))
    (trial/'interval-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    z=np.load(job/'registered-boundary/registered-boundary.npz')
    check_layers(y,[z[k+'_node_joint'][z[k+'_face_nodes']] for k in 'OID'],trial/'loaded-layers.json',layers)
    print(report,json.loads((trial/'loaded-layers.json').read_bytes())['pair_counts'],flush=True)
if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
