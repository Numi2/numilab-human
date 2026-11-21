import re
import numpy as np

def old_missing(q,status,table):
    mask=status==5
    if not np.all((status==0)|mask): raise ValueError('Nonconverged cardiac coordinates')
    for row in table.read_text().splitlines():
        terms=re.findall(r'(-?0x[0-9a-f.]+p[+-]\d+)f', row)
        if not terms:continue
        a,b,lo,hi=np.array([float.fromhex(v) for v in terms]).reshape(4,8)
        p=int(a[7]);a=a[:7];b=b[:7];ids=np.flatnonzero(mask)
        s=(q[ids,p]-a[p])/(b[p]-a[p]);err=q[ids]-a-s[:,None]*(b-a);cs=np.arange(7)!=p
        passed=(s>=0)&(s<=1)&np.all((err[:,cs]>=lo[:7][cs]+.000001)&(err[:,cs]<=hi[:7][cs]-.000001),axis=1)
        mask[ids[passed]]=False
    return np.flatnonzero(mask)

def tube(points):
    # Endpoints taken from actual extrema, partitioning on the widest channel.
    # Extend the pivot by two representable floats for stable inward membership.
    axis=int(np.argmax(np.ptp(points,axis=0)))
    a=points[np.argmin(points[:,axis])].copy();b=points[np.argmax(points[:,axis])].copy()
    d=b-a
    if d[axis]<.00001:
        d=np.zeros(7);d[axis]=.00004;a[axis]-=.00002;b=a+d
    else:
        a-=d*.003;b+=d*.003
    a=a.astype('f4').astype(float);b=b.astype('f4').astype(float)
    t=(points[:,axis]-a[axis])/(b[axis]-a[axis]);errs=points-a-t[:,None]*(b-a)
    lo=np.nextafter((errs.min(0)-.000004).astype('f4'),np.float32(-np.inf))
    hi=np.nextafter((errs.max(0)+.000004).astype('f4'),np.float32(np.inf))
    lo[axis]=hi[axis]=0
    return np.array([a,b,lo,hi],dtype='f4'),axis
