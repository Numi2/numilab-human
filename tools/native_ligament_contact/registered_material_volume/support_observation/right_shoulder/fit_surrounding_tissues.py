"""Fit the connected registered material against same-frame neighboring solids with cell constraints."""
import json,mmap,runpy,sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import factorized
from scipy.optimize import minimize
from prepare_material_interface import solid
from fit_scapular_material import forces
from prepare_infraspinatus import dets
from compile_connected_joint import digest


def determinants(x,t,ref):
    a,b,c=(x[t[:,1:]]-x[t[:,:1]]).transpose(1,0,2)
    g=np.stack((np.cross(b,c),np.cross(c,a),np.cross(a,b)),axis=1)/ref[:,None,None]
    return np.sum(a*g[:,0],axis=1),np.concatenate((-g.sum(1)[:,None],g),axis=1)


def run(case,record,reader,witness,out):
    out.mkdir(exist_ok=True)
    z=np.load(record);f=z['boundary'];t=z['tets'];world=z['accepted'];origin=world.mean(0);y=(world-origin)*1000;x=y.copy()
    d0=dets(x,t);ref=dets(z['ref']*1000,t);V0=d0.sum()/6
    rc=next((case/'native-traces/accepted-geometry').glob('*.receipt.json'));d=json.loads(rc.read_bytes())
    history=json.loads(witness.read_bytes());assert history['capture']==digest(rc)
    poses={r['body_index']:r for r in d['accepted_body_poses']}
    # Include the complete neighboring bounds of the first fit. Further nearby
    # surfaces enter the working set as soon as the moving AABB approaches them.
    domain={}
    with Path(d['accepted_pack_path']).open('rb')as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)as buf:
        for key,_,v,tri in runpy.run_path(str(reader))['primitives'](buf,poses):
            if key==(51005,83) or key[0]==51007:continue
            if np.any(v.max(0)<world.min(0)-.02) or np.any(v.min(0)>world.max(0)+.02):continue
            points=(v-origin)*1000;m,_,meta=solid(points,tri,str(key))
            if str(m.status())!='Error.NoError' or meta['mesh']['nonmanifold_or_open']:
                raise ValueError(('Indeterminate neighbor',key))
            domain[key]=(m,points.min(0),points.max(0))
    inv=np.linalg.inv(y[t[:,1:]]-y[t[:,:1]])
    g=np.concatenate([-inv.sum(2,keepdims=True),inv],axis=2).transpose(0,2,1)
    ij,ji=np.broadcast_arrays(t[:,:,None],t[:,None,:])
    E=(g@g.transpose(0,2,1))*(d0/6)[:,None,None]
    mass=np.bincount(t.ravel(),weights=np.repeat(d0/24,4))
    ell=np.median(np.linalg.norm(y[f[:,1]]-y[f[:,0]],axis=1))*2
    solve=factorized((coo_matrix((E.ravel(),(ij.ravel(),ji.ravel())),shape=(len(y),)*2).tocsc()+diags(mass/ell**2)).tocsc())
    # Preserve local oriented material under the fit. Source slivers may start
    # below the chosen 0.4 construction margin and must improve monotonically.
    min_ratio=np.minimum(d0/ref,.4)
    def metric(p,grad=False):
        body,tag,_=solid(p,f,'infraspinatus');lo=p.min(0);hi=p.max(0)
        vals=[];rows=[];keys=[]
        for k,(other,a,b) in domain.items():
            if np.any(b<lo) or np.any(hi<a):continue
            overlap=body^other;v=abs(overlap.volume())
            if v<1e-10:continue
            vals.append(v);keys.append(str(k))
            if grad:rows.append(forces(p,f,overlap,tag))
        vals.append(body.volume()-V0);keys.append('volume')
        if grad:
            gradv=np.zeros_like(p);tri=p[f]
            normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])/6
            for i in range(3):np.add.at(gradv,f[:,i],normal)
            rows.append(gradv)
        return np.array(vals),np.array(rows),keys
    logs=[]
    for it in range(32):
        v,A,keys=metric(x,True)
        j,G=determinants(x,t,ref)
        merit=np.linalg.norm(v)
        logs.append(dict(step=it,volumes_mm3=dict(zip(keys,v.tolist())),min_J=float(j.min()),merit_mm3=float(merit)))
        if merit<1e-7:break
        # A damped correction: infeasible anatomical constraints cannot create
        # arbitrary motion through a singular intersection-volume Jacobian.
        RHS=-v
        H=np.array([solve(a) for a in A]);flat=A.reshape(len(A),-1)
        gram=flat@H.reshape(len(A),-1).T
        scale=np.sqrt(np.maximum(gram.diagonal(),1e-24))
        coeff=np.linalg.lstsq(gram/scale[:,None]/scale[None,:],RHS/scale,rcond=1e-9)[0]/scale
        move=np.einsum('a,aij->ij',coeff,H)
        # Apply a minimum-metric feasibility correction to endangered cells,
        # preserving the full overlap gradient in subsequent fresh iterations.
        active=np.array([],int)
        for inner in range(12):
            linear=j+np.einsum('tij,tij->t',G,move[t])
            violating=np.flatnonzero(linear<min_ratio+.01)
            sel=np.union1d(active,violating)
            if len(sel)==len(active) or not len(sel):break
            active=sel
            D=np.zeros((len(active),len(x),3))
            D[np.arange(len(active))[:,None],t[active]]=G[active]
            Z=np.array([solve(a)for a in D]);B=D.reshape(len(D),-1)
            Q=B@Z.reshape(len(D),-1).T
            h=min_ratio[active]+.015-j[active]-B@move.ravel()
            norm=np.sqrt(np.maximum(Q.diagonal(),1.e-22));U=Q/norm[:,None]/norm[None,:]
            target=h/norm
            sol=minimize(lambda a:(.5*a@(U@a)-target@a,U@a-target),np.zeros(len(D)),jac=True,method='L-BFGS-B',
                         bounds=[(0.,None)]*len(D),options={'ftol':1e-14,'gtol':1e-9,'maxiter':300})
            move+=np.einsum('a,aij->ij',sol.x/norm,Z)
        for b in range(18):
            a=2.**-b;p=x+a*move
            if np.min(dets(p,t)/ref-min_ratio)<-1e-6:continue
            if abs(p-y).max()>18.:continue
            vals,_,_=metric(p)
            if np.linalg.norm(vals)<merit:
                x=p;logs[-1]['alpha']=a;break
        else:
            logs[-1]['stop']='No lower-overlap, cell-admissible bounded displacement';break
    vals,_,keys=metric(x);disp=(x-y)/1000
    np.savez_compressed(out/'fitted.npz',displacement_m=disp,source_cells=t)
    r=dict(source=digest(record),capture=digest(rc),neighbors=len(domain),preserved_J_floor_min=float(min_ratio.min()),
           min_J=float((dets(x,t)/ref).min()),peak_correction_mm=float(np.max(np.linalg.norm(x-y,axis=1))),
           initial_volume_mm3=V0,final_volume_error_mm3=float(vals[-1]),final_geometry_mm3=dict(zip(keys,vals.tolist())),iterations=logs)
    (out/'fit.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print('initial and final merit',logs[0]['merit_mm3'],np.linalg.norm(vals),r['min_J'],r['peak_correction_mm'],flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
