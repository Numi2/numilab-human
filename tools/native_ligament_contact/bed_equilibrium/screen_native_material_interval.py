import ctypes,json,sys
from pathlib import Path
import numpy as np
import re
cfg,pose,base,variant,log,out=map(Path,sys.argv[1:])
env=json.loads(cfg.read_text())['environment'];k='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
def table(path,typ):
    return np.loadtxt(path,skiprows=1,dtype=typ)
rs=table(env[k+'TETGEN_NODE'],float);cs=table(pose,float);el=table(env[k+'TETGEN_ELEMENTS'],np.int32)
assert np.array_equal(rs[:,0],cs[:,0]) and np.array_equal(el[:,0],np.arange(1,len(el)+1))
ref=np.ascontiguousarray(rs[:,1:]);q=np.ascontiguousarray(cs[:,1:]);t=np.ascontiguousarray(el[:,1:5]-1);del rs,cs
w=re.findall(r'cell=(\d+) material=(\d+) center=(\S+) radius=(\S+)',log.read_text())
matched=np.array([int(a[0]) for a in w])
record={'reference_matched':False}
for label,lib in [('baseline',base),('exact',variant)]:
    api=ctypes.CDLL(str(lib.resolve())).scanCellIntervals
    fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    api.argtypes=[fp,fp,ctypes.c_size_t,ip,ctypes.c_size_t,fp]
    val=np.empty((len(t),3));assert api(ref,q,len(q),t,len(t),val)==0
    bad=np.flatnonzero((val[:,0]-val[:,1]<float(np.float32(.35)))|(val[:,0]+val[:,1]>2.5)|~np.isfinite(val).all(1))
    if label=='baseline':
        ok=np.array_equal(bad,matched) and np.allclose(val[bad,:2],np.array([[float(i[2]),float(i[3])] for i in w]),atol=2e-8,rtol=2e-8)
        record['reference_matched']=bool(ok)
    record[label]={'failures':len(bad),'skin':int((el[bad,5]==15).sum()),'fat':int((el[bad,5]==16).sum()),
                  'worst_lower_margin':float((val[:,0]-val[:,1]-float(np.float32(.35))).min()),'worst_upper_margin':float((2.5-val[:,0]-val[:,1]).min()),
                  'witness_ids':bad[:30].tolist()}
    if label=='exact' and len(bad)>0: np.savez_compressed(out.with_suffix('.npz'),ids=bad,nodes=t[bad],intervals=val[bad],points=q[t[bad]])
out.write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
if not record['reference_matched']:raise ValueError('raw cells do not reproduce native reference')
