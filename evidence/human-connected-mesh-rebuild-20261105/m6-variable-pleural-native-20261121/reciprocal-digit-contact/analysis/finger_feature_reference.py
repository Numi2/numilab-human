"""Find closest material features of opposing registered phalanges before contact."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
def dot(x,y):return np.einsum('...i,...i->...',x,y)
def edges(f):return np.unique(np.sort(f[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1),axis=0)
def closest(a,fa,b,fb):
    best=(np.inf,None)
    def select(d,valid,kind,ids1,ids2,s,t,vec):
        nonlocal best
        vv=np.where(valid,d,np.inf);i,j=np.unravel_index(np.argmin(vv),vv.shape)
        if vv[i,j]<best[0]:
            best=(float(vv[i,j]),dict(kind=kind,first=np.atleast_1d(ids1[i]).tolist(),second=np.atleast_1d(ids2[j]).tolist(),
                s=float(s[i,j]) if s is not None else None,t=float(t[i,j]) if t is not None else None,ab=vec[i,j].tolist()))
    for rev,(v,o,fo) in enumerate(((a,b,fb),(b,a,fa))):
        tri=o[fo];e=tri[:,1]-tri[:,0];f=tri[:,2]-tri[:,0];n=np.cross(e,f);n2=dot(n,n)
        if np.any(n2==0):raise ValueError('Degenerate cortex triangle')
        h=v[:,None]-tri[None,:,0];s=dot(np.cross(h,f),n)/n2;t=dot(np.cross(e,h),n)/n2
        vnormal=dot(h,n)/n2;delta=vnormal[:,:,None]*n
        select(dot(delta,delta),(s>=0)&(t>=0)&(s+t<=1),('VF','FV')[rev],np.arange(len(v)),fo,s,t,delta*([1,-1][rev]))
        ed=edges(fo);p=o[ed[:,0]];axis=o[ed[:,1]]-p
        h=v[:,None]-p;s=np.clip(dot(h,axis)/dot(axis,axis),0,1);delta=h-s[:,:,None]*axis
        select(dot(delta,delta),np.ones_like(s,dtype=bool),('VE','EV')[rev],np.arange(len(v)),ed,s,None,delta*([1,-1][rev]))
    ea,eb=edges(fa),edges(fb)
    p=a[ea[:,0]];q=b[eb[:,0]];u=a[ea[:,1]]-p;v=b[eb[:,1]]-q
    aa=dot(u,u)[:,None];cc=dot(v,v)[None];bb=u@v.T;w=p[:,None]-q[None];dd=dot(w,u[:,None]);ee=dot(w,v[None])
    cross=np.cross(u[:,None],v[None]);den=dot(cross,cross);valid=den>1e-12*aa*cc
    den=np.where(valid,den,1.);s=(bb*ee-cc*dd)/den;t=(aa*ee-bb*dd)/den
    delta=w+s[:,:,None]*u[:,None]-t[:,:,None]*v[None]
    select(dot(delta,delta),valid&(s>=0)&(s<=1)&(t>=0)&(t<=1),'EE',ea,eb,s,t,delta)
    return dict(distance_m=float(np.sqrt(best[0])),**best[1])
def replay(case,out):
    r=json.loads((case/'step-0.receipt.json').read_text());poses={x['body_index']:x for x in r['accepted_body_poses']}
    data=np.load(case/'initial-cortical.npz');members=json.loads((case/'initial-cortical.json').read_text())['member_by_stable_id']
    frame={};ids={}
    for name,body,_,_ in members.values():
        p=poses[body];frame[name]=Rotation.from_quat(p['quaternion_xyzw']).inv().apply(data[name+'_v'].astype(float)-p['position_m']);ids[name]=body
    captured={0:{b:(np.asarray(poses[b]['position_m']),np.asarray(poses[b]['quaternion_xyzw'])) for b in ids.values()}}
    for row in csv.DictReader((case/'resting-body-motion-partition.csv').open()):
        b=int(row['body_index']);k=int(row['accepted_step'])
        if b in ids.values():
            p=[float(row['root_'+a+'_m'])+float(row['com_relative_root_'+a+'_m']) for a in 'xyz']
            q=[float(row['orientation_'+a]) for a in 'xyzw'];captured.setdefault(k,{})[b]=(np.array(p),np.array(q))
    reports={}
    for k in sorted(set(captured)|set(range(256,353,4))):
        l=max(x for x in captured if x<=k);h=min(x for x in captured if x>=k);t=(k-l)/(h-l) if h>l else 0
        def pos(name):
            b=ids[name];p,q=captured[l][b];p1,q1=captured[h][b];rot=Rotation.from_quat(q) if l==h else Slerp([0,1],Rotation.from_quat([q,q1]))(t)
            return rot.apply(frame[name])+p+t*(p1-p)
        row={'interpolated':l!=h}
        for side in ['r','l']:
            first='4proxph_'+side;second='5proxph_'+side;row[side]=closest(pos(first),data[first+'_f'],pos(second),data[second+'_f'])
        reports[k]=row
    out.write_text(json.dumps(reports,indent=2)+'\n')
if __name__=='__main__':replay(*map(Path,sys.argv[1:]))
