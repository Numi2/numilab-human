"""Strict independent reader for the existing FSSCARR01 packet."""
from pathlib import Path
import hashlib,struct
import numpy as np
FIELDS=(('source_parent_tet_rows','<u4',None),('global_source330_tet_rows','<u4',None),('parent_offsets','<u4',None),('carrier_weights','<f8',None),('carrier_barycentric','<f8',None),('carrier_target_child_rows','<u4',None),('carrier_target_distance_m','<f8',None))
def load_packet(path, npz_path, expected_global_rows):
    data=Path(path).read_bytes()
    if data[:9]!=b'FSSCARR01':raise ValueError('source carrier magic')
    np_,nc=struct.unpack_from('<QQ',data,9)
    if np_!=len(expected_global_rows) or nc>4000000:raise ValueError('source carrier row domain')
    n=[np_,np_,np_+1,nc,nc*4,nc,nc];cursor=25;arrays={}
    for (name,dtype,_),count in zip(FIELDS,n):
        size=count*np.dtype(dtype).itemsize
        if cursor+size>len(data):raise ValueError('truncated source carrier packet')
        a=np.frombuffer(data,dtype=dtype,count=count,offset=cursor).copy()
        if name=='carrier_barycentric':a=a.reshape(nc,4)
        arrays[name]=a;cursor+=size
    if cursor!=len(data):raise ValueError('source carrier packet trailing bytes')
    with np.load(npz_path,allow_pickle=False) as z:
        if set(z.files)!=set(arrays):raise ValueError('source carrier NPZ field domain')
        for name,a in arrays.items():
            if a.dtype!=z[name].dtype or not np.array_equal(a,z[name]):
                raise ValueError('source carrier binary/NPZ mismatch '+name)
    if not np.array_equal(arrays['source_parent_tet_rows'],np.arange(np_,dtype=np.uint32)):
        raise ValueError('source carrier parent ordering')
    if not np.array_equal(arrays['global_source330_tet_rows'],expected_global_rows):
        raise ValueError('source carrier global source330 identity')
    offsets=arrays['parent_offsets'].astype(np.int64);w=arrays['carrier_weights'];b=arrays['carrier_barycentric'];dist=arrays['carrier_target_distance_m']
    if offsets[0]!=0 or offsets[-1]!=nc or np.any(np.diff(offsets)<0):
        raise ValueError('source carrier CSR')
    if not np.isfinite(w).all() or np.any(w<=0) or not np.isfinite(b).all() or np.any(b<0) or np.any(b>1):
        raise ValueError('invalid source carrier quadrature')
    if not np.isfinite(dist).all() or np.any(dist<0) or np.any(dist>.019353):
        raise ValueError('invalid source carrier witness distance')
    if np.any(np.abs(b.sum(axis=1)-1)>1e-12):raise ValueError('source carrier barycentric closure')
    for p in np.flatnonzero(np.diff(offsets)):
        a,e=offsets[p:p+2]
        if abs(w[a:e].sum()-1)>1e-12 or np.max(np.abs(w[a:e]@b[a:e]-.25))>1e-12:
            raise ValueError('source carrier original tetrahedron mass/first moments')
    return arrays,hashlib.sha256(data).hexdigest()
def relative_close(actual,expected,what,rtol=1e-10):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(what+' shape or finite')
    if np.any(np.abs(a-b)>rtol*np.abs(b)):raise ValueError(what+' relative closure')
    return float(np.max(np.abs(a-b),initial=0))
