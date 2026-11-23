"""Test whether the native joint ranges and equality constraints can recover the existing hand registration at rest."""
from pathlib import Path
import json,struct,sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from audit_fiber_load_bounds import read_pose
from compile_connected_joint import digest

def run(h,job,out):
    out.mkdir();r=h.parent;current=read_pose(r/'native-ligament-material/stderr.log')
    raw=(r/'supported/source-bound.nhrigid').read_bytes();launch=json.loads((job/'launch.json').read_text());args=launch['argv'];eq=Path(args[args.index('--joint-equality-payload')+1]);payload=eq.read_bytes()
    if len(payload)!=80+96*51 or payload[:5]!=b'NHEQ1':raise ValueError('Mismatched equality layout')
    equal=[struct.unpack_from('<4I8f',payload,80+i*96) for i in range(51)]
    old={v['body_index']:v['current'] for v in json.loads((job/'native-initial-bodies.json').read_text())['all_bodies']}
    now=np.array(current['bodyPoses']);q0=np.array(current['poseQ']);head=224+157*160;dof=head+156*144
    joints=[struct.unpack_from('<8I28f',raw,head+i*144) for i in range(156)]
    props=[struct.unpack_from('<8I8f',raw,dof+i*64) for i in range(128)]
    bounds={u[2]:u[8:10] for u in props if u[5]&4};results=[]
    for side,begin,hand in [('l',78,95),('r',28,45)]:
        chain=[];b=hand
        while b>=begin:
            j=joints[b-1];assert j[1]==b and j[2]==0 and j[5]==1
            R=Rotation.from_quat(now[j[0],3:]);pivot=now[j[0],:3]+R.apply(j[20:23])
            axis=(R*Rotation.from_quat(j[28:32])).apply(j[8:11]);assert abs(np.linalg.norm(axis)-1)<1e-5
            chain.append((j[4],axis,pivot,b));b=j[0]
        chain=chain[::-1];qs=[c[0] for c in chain];dependent={v[0] for v in equal};ids=np.array([i for i,axis,pivot,body in chain if i not in dependent and body>=hand-3])
        def projected(x):
            q=q0.copy();q[ids]=x
            for e in equal:
                y,_,m,_,y0,m0,*coeff=e
                dx=0 if m==0xffffffff else q[m]-m0
                q[y]=y0+np.polynomial.polynomial.polyval(dx,coeff[:5])
            return q
        if np.max(abs(projected(q0[ids])[qs]-q0[qs]))>2e-6:raise ValueError('Rest not on compiled joint manifold')
        def fk(x):
            q=projected(x);Q=np.eye(3);p=np.zeros(3);states={}
            for qi,axis,piv,body in chain:
                A=Rotation.from_rotvec(axis*(q[qi]-q0[qi])).as_matrix()
                p+=Q@(piv-A@piv);Q=Q@A
                states[body]=(Q@now[body,:3]+p,Rotation.from_matrix(Q)*Rotation.from_quat(now[body,3:]))
            return q,states
        def err(x):
            _,st=fk(x);p,Q=st[hand];desired=old[hand]
            return np.r_[(p-desired['position_world_m'])*1000.,(Q*Rotation.from_quat(desired['orientation_world_xyzw']).inv()).as_rotvec()*95.]
        lower,upper=np.asarray([bounds[i] for i in ids]).T
        x0=np.clip(q0[ids],lower+1e-7,upper-1e-7);best=None;rng=np.random.default_rng(71)
        for i in range(8):
            start=x0 if i==0 else np.clip(x0+rng.normal(0,.2,len(ids)),lower+1.e-7,upper-1.e-7)
            # Secondary objective selects the smallest change from the old input pose.
            sol=least_squares(lambda x:np.r_[err(x)[:3]*.00005,err(x)[3:],.0005*(x-x0)],start,bounds=(lower,upper),max_nfev=350,ftol=1e-12,gtol=1.e-10,xtol=1.e-12)
            if best is None or sol.cost<best.cost:best=sol
            if np.linalg.norm(err(best.x)[3:])<1e-5:break
        q,st=fk(best.x);err6=err(best.x);record=dict(side=side,hand=hand,initial_position_error_mm=float(np.linalg.norm(err(x0)[:3])),final_position_error_mm=float(np.linalg.norm(err6[:3])),final_orientation_error_deg=float(np.rad2deg(np.linalg.norm(err6[3:])/95.)),q={str(k):float(q[k]) for k in qs if k not in dependent},q_offsets_rad={str(k):float(q[k]-q0[k]) for k in ids},equality_max_limit_excess_rad=max([0.,*[max(bounds[k][0]-q[k],q[k]-bounds[k][1]) for k in qs]]),body_poses={str(k):[*p,*Q.as_quat()] for k,(p,Q) in st.items()})
        results.append(record);print(side,{k:record[k] for k in ('initial_position_error_mm','final_position_error_mm','final_orientation_error_deg','equality_max_limit_excess_rad')},flush=True)
    (out/'result.json').write_text(json.dumps(dict(source_pose=digest(job/'native-initial-bodies.json'),current_capture=digest(r/'native-ligament-material/stderr.log'),source_rigid=digest(r/'supported/source-bound.nhrigid'),joint_equalities=digest(eq),arms=results),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
