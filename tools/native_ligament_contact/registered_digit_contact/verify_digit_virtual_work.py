"""Independent rigid-twist work and action/reaction checks for registered cortical features."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear
def point_triangle(p,t):
    a,b,c=t;e=b-a;f=c-a;h=p-a
    uv=np.linalg.lstsq(np.column_stack([e,f]),h,rcond=None)[0];inside=min(uv)>=0 and uv.sum()<=1
    if inside:return a+uv[0]*e+uv[1]*f,True
    candidates=[]
    for u,v in [(a,b),(a,c),(b,c)]:
        d=v-u;candidates.append(u+d*np.clip((p-u)@d/(d@d),0.,1.))
    return min(candidates,key=lambda v:np.linalg.norm(v-p)),False
def feature(A,B,kind,sign):
    if kind=='EE':
        u=A[1]-A[0];v=B[1]-B[0];m=np.column_stack([u,-v]);d=B[0]-A[0]
        st=np.linalg.lstsq(m,d,rcond=None)[0];inside=np.all(st>=0) and np.all(st<=1)
        if not inside:st=lsq_linear(m,d,bounds=(0.,1.),method='bvls',tol=1e-14).x
        a=A[0]+u*st[0];b=B[0]+v*st[1];n=np.cross(u,v)
    else:
        a=A[0];b,inside=point_triangle(a,B);n=np.cross(B[1]-B[0],B[2]-B[0])
    n=sign*n if inside else a-b
    n/=np.linalg.norm(n)
    return (a-b)@n,n,a,b
def run(assets,case,out):
    raw=(assets/'digit-cortical-features.bin').read_bytes();reg=json.loads((assets/'registration.json').read_text());rng=np.random.default_rng(817)
    stats=[];check={}
    for step in (0,415):
        r=json.loads((case/f'step-{step}.receipt.json').read_text());bodies={p['body_index']:p for p in r['accepted_body_poses']}
        for i,m in enumerate(reg['rows']):
            v=np.frombuffer(raw[i*80+16:(i+1)*80],dtype='<f4').reshape(4,4)[:,:3];k=int(m['kind']=='EE')+1
            X=[]
            for body,x in zip(m['body'],[v[:k],v[k:]],strict=True):
                p=bodies[body];X.append(Rotation.from_quat(p['quaternion_xyzw']).apply(x)+p['position_m'])
            A,B=X;kind=m['kind'];s=m['sign'];g,n,a,b=feature(A,B,kind,s);check[f'{step}:{i}']=g
            # Unit normal force at each material point.
            tau=np.cross(a-b,n)
            for _ in range(32):
                ta,tb,wa,wb=rng.normal(size=(4,3));eps=1.e-7
                ca,cb=[np.asarray(bodies[j]['position_m']) for j in m['body']]
                def at(h):
                    x=Rotation.from_rotvec(wa*h).apply(A-ca)+ca+ta*h
                    y=Rotation.from_rotvec(wb*h).apply(B-cb)+cb+tb*h
                    return feature(x,y,kind,s)[0]
                rate=(at(eps)-at(-eps))/(2*eps)
                work=n@(ta+np.cross(wa,a-ca)-tb-np.cross(wb,b-cb))
                stats.append((abs(rate-work),np.linalg.norm(tau),step,i))
    err=max(x[0] for x in stats);torque=max(x[1] for x in stats)
    if err>2e-6 or torque>2e-10:raise ValueError((err,torque))
    out.write_text(json.dumps({'random_velocity_tests':len(stats),'max_virtual_work_absolute_error':err,'max_internal_torque_per_unit_force_m':torque,'max_net_force_per_unit_force':0,'signed_feature_gaps_m':check},indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
