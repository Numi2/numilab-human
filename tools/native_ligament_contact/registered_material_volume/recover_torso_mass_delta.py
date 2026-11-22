"""Save or restore a sparse, byte-verified difference of two existing NumPy role-flow packets."""
import argparse
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha
def recover(base,delta,dst):
    with np.load(delta) as d,np.load(base) as old:
        assert sha(base)==str(d['base_sha256']);result={}
        for k in old.files:
            a=old[k].copy();a[d[k+'_idx']]=d[k+'_values'];a=np.concatenate([a,d[k+'_tail']]);result[k]=a
        np.savez_compressed(dst,**result);assert sha(dst)==str(d['recovered_sha256'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('base',type=Path);ap.add_argument('delta',type=Path);ap.add_argument('recovered',type=Path);ap.add_argument('--encode-from',type=Path)
    a=ap.parse_args()
    if a.encode_from:
        d=dict(base_sha256=sha(a.base),recovered_sha256=sha(a.encode_from))
        with np.load(a.base) as old,np.load(a.encode_from) as current:
            assert old.files==current.files
            for k in old.files:
                b=old[k];c=current[k];assert b.ndim==c.ndim==1 and len(c)>=len(b) and b.dtype==c.dtype
                ids=np.flatnonzero(b.view('u'+str(b.dtype.itemsize))!=c[:len(b)].view('u'+str(b.dtype.itemsize))).astype('i4')
                d[k+'_idx']=ids;d[k+'_values']=c[ids];d[k+'_tail']=c[len(b):]
        np.savez_compressed(a.delta,**d)
    recover(a.base,a.delta,a.recovered)
if __name__=='__main__':main()
