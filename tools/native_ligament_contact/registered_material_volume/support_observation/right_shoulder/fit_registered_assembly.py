"""Fit six registered muscles together, using signed material volumes and fixed native tissue boundaries."""
from itertools import combinations
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation
from manifold3d import Manifold,Mesh64,Error,OpType
from repair_registered_surface import volume,signed_volume_gradient
from assemble_material import frames
from measure_assembly_contacts import canonical
from inspect_shoulder_source import nearest
from fit_scapular_material import forces
from prepare_material_interface import solid
from prepare_infraspinatus import dets
from prepare_shoulder import MOVING,LABELS,read
from compile_connected_joint import digest


def layer_surface(j,org,lo,hi):
    from build_carpal_cavity import boundary
    v=j/'conditioned/witness-material/volume';p=(np.loadtxt(j/'posterior-interface-082/native-pair/physical-accepted.node',skiprows=1,usecols=(1,2,3))-org)*1000
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=raw[:,:4]-1
    keep=[]
    for ids in np.array_split(np.arange(len(t)),48):
        x=p[t[ids]]
        keep.extend(ids[np.all((x.max(1)>=lo-12)&(x.min(1)<=hi+12),axis=1)])
    ids=np.array(keep)
    def split(q):
        cells=t[q];f=boundary(cells);ix,r=np.unique(f,return_inverse=True)
        s=Manifold(Mesh64(np.ascontiguousarray(p[ix]),np.ascontiguousarray(r.reshape(-1,3),dtype='u8')))
        if s.status()==Error.NoError and s.volume()<0:
            s=Manifold(Mesh64(np.ascontiguousarray(p[ix]),np.ascontiguousarray(r.reshape(-1,3)[:,[0,2,1]],dtype='u8')))
        expected=dets(p,cells).sum()/6
        if s.status()==Error.NoError and abs(s.volume()-expected)<max(1e-8,expected*1e-9):return [s]
        if len(q)<2:raise ValueError('Unresolved physical layer')
        c=p[cells].mean(1);axis=np.ptp(c,0).argmax()
        return [s for a in np.array_split(np.argsort(c[:,axis],kind='stable'),2)for s in split(q[a])]
    result={}
    for role in (15,16):
        q=ids[raw[ids,4]==role];parts=split(q)
        s=Manifold.batch_boolean(parts,OpType.Add) if len(parts)>1 else parts[0]
        if s.status()!=Error.NoError or abs(s.volume()-sum(a.volume()for a in parts))>1e-5:raise ValueError('Incomplete material union')
        result[0,role]=(s,p[t[q]].min((0,1)),p[t[q]].max((0,1)))
    return result


def run(j,reg,cert,mat,lib,out,steps):
    out.mkdir(exist_ok=True);case=j/'posterior-interface-082/native-pair';pose=frames(case)[-1]
    a=read(case/'launch.json')['argv'];tendon=Path(a[a.index('--tendon-payload')+1]);ends=read(tendon.with_suffix('.manifest.json'))['endpoints']
    org=np.array(read(reg/'registration.json')['origin']);p0=[];F=[];T=[];N=[];solver=[];pins=[];goal=[]
    for sid in MOVING:
        z=np.load(cert/f'bank-{sid}.npz');m=np.load(mat/f'bank-{sid}.npz')
        x=(z['accepted']-org)*1000;t=z['t'];f=m['f'];ref=dets(x,t);goal.append(volume(x,f))
        fixed=set()
        for e in ends:
            if e.get('muscle')!=LABELS[sid]:continue
            b=pose[e['body_index']];sites=[e['resolved_local_point_m']]+list((e.get('surface') or {}).get('node_local_points_m',[]))
            world=Rotation.from_quat(b['quaternion_xyzw']).apply(sites)+b['position_m']
            _,_,idx=nearest((np.ascontiguousarray(x/1000+org),np.ascontiguousarray(f,'i4')),world,lib)
            fixed.update(f[idx].ravel().tolist())
        if not fixed:raise ValueError('Unbound tendon footprint')
        inv=np.linalg.inv(x[t[:,1:]]-x[t[:,:1]])
        g=np.concatenate((-inv.sum(2,keepdims=True),inv),axis=2).transpose(0,2,1)
        ii,jj=np.broadcast_arrays(t[:,:,None],t[:,None,:]);E=(g@g.transpose(0,2,1))*(ref/6)[:,None,None]
        mass=np.bincount(t.ravel(),weights=np.repeat(ref/24,4),minlength=len(x))
        ell=2*np.median(np.linalg.norm(x[f[:,1]]-x[f[:,0]],axis=1))
        K=coo_matrix((E.ravel(),(ii.ravel(),jj.ravel())),shape=(len(x),len(x))).tocsc()+diags(mass/ell**2)
        free=np.setdiff1d(np.arange(len(x)),sorted(fixed));solver.append((free,splu(K[free][:,free].tocsc())))
        pins.append(sorted(fixed));p0.append(x);T.append(t);F.append(f);N.append(len(x))
    start=np.cumsum([0]+N);ref=[dets(x,t)for x,t in zip(p0,T)]
    data=np.load(reg/'neighbors.npz');fixed={}
    pending_fixed=[]
    for name in data.files:
        if not name.endswith('-v'):continue
        k,b=map(int,name[:-2].split('-'))
        if k==0:continue
        x=data[name];f=data[name[:-1]+'f'];lo=x.min(0);hi=x.max(0)
        if not any(np.all((xx.max(0)+10>=lo)&(xx.min(0)-10<=hi))for xx in p0):continue
        s,r=canonical(x,f)
        if s is None:
            # Outside the exact opening AABBs. Keep the bounds as a hard
            # exclusion for every line-search candidate until a solid exists.
            if any(np.all((xx.max(0)>=lo)&(xx.min(0)<=hi))for xx in p0):raise ValueError(f'Unresolved overlapping fixed neighbor {(k,b)}')
            pending_fixed.append(((k,b),lo,hi));continue
        fixed[k,b]=(s,lo,hi)
    fixed.update(layer_surface(j,org,np.vstack(p0).min(0),np.vstack(p0).max(0)))
    def measure(P,grad=False):
        obs=[];G=[];tags=[];bodies=[solid(x,f,str(s))[:2]for x,f,s in zip(P,F,MOVING)]
        lo=[p.min(0) for p in P];hi=[p.max(0) for p in P]
        for i,(s,tag) in enumerate(bodies):
            obslist=[((51005,MOVING[b]),m,lo[b],hi[b],b)for b,(m,_)in enumerate(bodies) if b>i]+[(k,m,l,h,None)for k,(m,l,h)in fixed.items()]
            for key,m,l,h,b in obslist:
                if np.any(hi[i]<l)or np.any(lo[i]>h):continue
                obj=s^m
                if obj.status()!=Error.NoError:raise ValueError('Invalid contact intersection')
                v=float(obj.volume())
                if v< -1e-7:raise ValueError('Negative contact volume')
                if v<=1e-8:continue
                tags.append((MOVING[i],*key));obs.append(v)
                if grad:
                    a=np.zeros((start[-1],3))
                    a[start[i]:start[i+1]]=forces(P[i],F[i],obj,tag)
                    if b is not None:a[start[b]:start[b+1]]=forces(P[b],F[b],obj,bodies[b][1])
                    G.append(a)
        for i in range(len(MOVING)):
            obs.append(abs(volume(P[i],F[i])-goal[i]));tags.append((MOVING[i],-1,-1))
            if grad:
                a=np.zeros((start[-1],3));a[start[i]:start[i+1]]=signed_volume_gradient(P[i],F[i]);G.append(a)
        return np.asarray(obs),tags,np.asarray(G)
    P=[x.copy()for x in p0];history=[];first=None
    for step in range(steps+1):
        obs,tags,G=measure(P,True);contacts=len(obs)-len(MOVING)
        values={str(tag):float(v) for tag,v in zip(tags[:contacts],obs[:contacts])}
        if first is None:
            first=values
            for checked in read(Path('layers.json'))['rows']:
                if checked['stage']!='after':continue
                actual=values.get(str((checked['muscle'],0,checked['material'])),0.)
                if abs(actual-checked['mm3'])>max(.001,1e-6*checked['mm3']):raise ValueError('Regional union differs from physical-cell intersection')
        report=dict(index=step,sum_pair_mm3=float(obs[:contacts].sum()),maximum_pair_mm3=float(obs[:contacts].max()),volume_error_mm3=obs[contacts:].tolist(),contacts=values)
        history.append(report);print(step,round(report['sum_pair_mm3'],2),round(report['maximum_pair_mm3'],2),flush=True)
        if step==steps:break
        base_rhs=np.r_[.15*obs[:contacts],[volume(x,f)-g for x,f,g in zip(P,F,goal)]]
        types=[(0,None)]*contacts+[(None,None)]*len(MOVING)
        rhs_rows=base_rhs.copy();rows=G;selected=set()
        for working in range(5):
            H=np.zeros_like(rows)
            for i,(free,K)in enumerate(solver):
                q=rows[:,start[i]+free].transpose(1,0,2)
                H[:,start[i]+free]=K.solve(q.reshape(len(free),-1)).reshape(q.shape).transpose(1,0,2)
            A=rows.reshape(len(rows),-1)@H.reshape(len(rows),-1).T
            scale=np.sqrt(np.maximum(A.diagonal(),1e-18));Q=A/(scale[:,None]*scale);rhs=rhs_rows/scale
            sol=minimize(lambda v:(.5*v@Q@v-rhs@v,Q@v-rhs),np.zeros(len(rows)),jac=True,method='L-BFGS-B',
                         bounds=types,options={'maxiter':600,'ftol':1e-13,'gtol':1e-8})
            delta=-np.einsum('a,aij->ij',sol.x/scale,H)
            peak=np.linalg.norm(delta,axis=1).max();delta*=min(1.,1.0/max(peak,1e-20))
            block=[];brhs=[]
            if working==4:break
            for i,(p,t,d) in enumerate(zip(P,T,ref)):
                future=dets(p+delta[start[i]:start[i+1]],t)/d
                tests=np.flatnonzero((future<.42)|(future>2.45))
                severity=np.maximum(.42-future[tests],future[tests]-2.45)
                for k in tests[np.argsort(-severity,kind='stable')[:12]]:
                    side=1 if future[k]>2.45 else -1
                    if (i,k,side)in selected:continue
                    selected.add((i,k,side))
                    verts=p[t[k]];e=verts[1:]-verts[0];jval=np.linalg.det(e)/d[k]
                    g=np.cross(e[[1,2,0]],e[[2,0,1]])/d[k]
                    row=np.zeros((start[-1],3))
                    row[t[k]+start[i]]=side*np.vstack((-g.sum(0),g))
                    block.append(row);brhs.append(.42-jval if side<0 else jval-2.45)
            if not block:break
            rows=np.concatenate((rows,block));rhs_rows=np.r_[rhs_rows,brhs];types.extend([(0,None)]*len(block))
        report['active_material_bounds']=len(selected)
        for bt in range(12):
            fac=2.**-bt;trial=[]
            for i,p in enumerate(P):
                y=p+fac*delta[start[i]:start[i+1]];free,K=solver[i]
                # Scalar restoration of the original closed material volume.
                for _ in range(3):
                    gv=signed_volume_gradient(y,F[i]);dvol=volume(y,F[i])-goal[i]
                    if abs(dvol)<1e-8:break
                    u=K.solve(gv[free]);den=np.sum(gv[free]*u)
                    if den<=0:raise ValueError('No free volume correction')
                    y[free]-=(dvol/den)*u
                trial.append(y)
            if any(np.any((v<.4)|(v>2.5))for x,t,d in zip(trial,T,ref) for v in [dets(x,t)/d]):continue
            if any(np.all((y.max(0)>=lo)&(y.min(0)<=hi)) for _,lo,hi in pending_fixed for y in trial):continue
            new,ids,_=measure(trial);vv={str(q):z for q,z in zip(ids[:-len(MOVING)],new[:-len(MOVING)])}
            # Reject new occupied tissue and increasing existing overlaps beyond Boolean roundoff.
            if abs(new[-len(MOVING):]).max()>1e-5 or sum(vv.values())>=sum(values.values()) or any(v>values.get(k,0)+1e-4 for k,v in vv.items()):continue
            report['alpha']=fac;P=trial;break
        else:
            report['stopped']='No cell-valid and volume-conservative joint step improves occupancy without increasing another crossing';break
    for i,(y,x)in enumerate(zip(P,p0)):
        np.savez_compressed(out/f'{MOVING[i]}.npz',x=y/1000+org,t=T[i],f=F[i],fixed_nodes=pins[i],displacement_m=(y-x)/1000)
    result=dict(source=digest(cert/'result.json'),native=digest(case/'launch.json'),layer_checks=digest(Path('layers.json')),history=history,
                peak_displacements_mm=[float(np.linalg.norm(x-y,axis=1).max())for x,y in zip(P,p0)],
                lowest_J=[float((dets(y,t)/d).min())for y,t,d in zip(P,T,ref)])
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:7]),int(sys.argv[7]))
