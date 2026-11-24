"""Replay native Float32 material determinant bounds for a registration-only change."""
import argparse,ctypes,json
from pathlib import Path
import numpy as np
from compile_connected_joint import digest

def run(volume,fit,checker,receipt):
    src=np.loadtxt(volume/'mesh.node',skiprows=1,usecols=(1,2,3))
    dst=np.loadtxt(fit/'current.node',skiprows=1,usecols=(1,2,3))
    t=np.ascontiguousarray(np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype=np.int32)-1)
    sha=digest(checker)
    if sha!=json.loads(receipt.read_bytes())['checker']:raise ValueError('Wrong native checker')
    arr=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    ids=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    fn=ctypes.CDLL(str(checker)).scanCellIntervals
    fn.argtypes=[arr,arr,ctypes.c_size_t,ids,ctypes.c_size_t,arr]
    vals=np.empty((len(t),3))
    if fn(src,dst,len(src),t,len(t),vals):raise ValueError('Native interval measurement failed')
    margin=np.minimum(vals[:,0]-vals[:,1]-float(np.float32(.35)),2.5-vals[:,0]-vals[:,1])
    failing=np.flatnonzero(~np.isfinite(margin)|(margin<=0))
    (fit/'interval.json').write_text(json.dumps(dict(checker=sha,input=digest(fit/'current.node'),elements=len(t),failed=failing.tolist(),minimum_margin=float(np.min(margin))),indent=2)+'\n')
    if len(failing):raise ValueError(f'{len(failing)} inadmissible native intervals')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('volume','fit','checker','receipt'):p.add_argument(n,type=Path)
    run(**vars(p.parse_args()))
