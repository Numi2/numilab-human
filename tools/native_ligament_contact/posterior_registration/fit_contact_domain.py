"""Test candidate unilateral contact against the actual closed bones and cut-material determinant constraints."""
import json,sys,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize,nnls
from solve_pelvic_contact import current_bank,poses_at
from cortical_distance import CorticalSurface
from qualify_material_motion import frame,local
from prepare_material_interface import read_inputs
from project_material_cells import barycentric,FACE
from pelvic_specimen import properties

def points(p47,prev,sid):
    d=np.load(p47/'partition-layers'/f'muscle_{sid}.npz');p=np.load(p47/'projected'/f'{sid}.npz');tri=d['triangles']
    source=np.load(prev/'native-release-motion'/f'entrance-{sid}.npz')['faces']
    t=p['cells'].astype(int);x=p['vertices'].astype(float);parent={}
    for i,cell in enumerate(t):
        for face in FACE:
            key=tuple(sorted(cell[list(face)]))
            if key in parent:del parent[key]
            else:parent[key]=i
    ids=np.flatnonzero(d['parent_bank']==f'muscle_{sid}');f=d['parent_face'][ids]
    owners=np.array([parent[tuple(sorted(row))] for row in source[f]])
    exterior=d['vertices'][tri[ids]];weights=np.stack([barycentric(pts,x[t[j]]) for pts,j in zip(exterior,owners)])
    owner=np.r_[p['port_owner_cell'],owners];xyz=np.concatenate((p['port_vertices'],exterior));shape=np.concatenate((p['port_shape'],weights))
    # Retain every material corner plus the centre of every integration facet.
    coordinate=np.concatenate((xyz.reshape(-1,3),xyz.mean(1)))
    n=np.r_[np.repeat(owner,3),owner];s=np.concatenate((shape.reshape(-1,4),shape.mean(1)))
    _,idx=np.unique(np.round(coordinate,13),axis=0,return_index=True)
    if np.max(abs(np.einsum('ij,ijk->ik',s[idx],x[t[n[idx]]])-coordinate[idx]))>1e-10:raise ValueError('Unbound cut-surface point')
    return t[n[idx]],s[idx],xyz,owner,shape

def jacobians(x,t,modes,reference):
    ds=x[t[:,1:]]-x[t[:,:1]]
    d=modes[t[:,1:]]-modes[t[:,:1]]
    c=np.stack((np.cross(ds[:,1],ds[:,2]),np.cross(ds[:,2],ds[:,0]),np.cross(ds[:,0],ds[:,1])),axis=1)
    return np.einsum('ti,ti->t',ds[:,0],c[:,0])/reference,np.einsum('tij,tijk->tk',c,d)/reference[:,None]

def fit(p47,prev,base,response,run,out,step,library):
    out.mkdir(exist_ok=True);results={}
    native=prev/'native-release-motion';rc=native/'step-8.receipt.json'
    data,_=read_inputs(prev/'release-posterior/config.json',rc,native,base)
    full={r['body_index']:frame(r) for r in json.loads(rc.read_text())['initial_anatomical_registration']['body_poses']}
    _,pose=poses_at(run,step)
    for sid,owners in ((29,[(1,1),(2,131),(18,128)]),(30,[(1,1),(3,145),(19,128)])):
        p,x,modes,_,_,_,_,_,_,_=current_bank(sid,p47,prev,response,run,step)
        nodes,w,tri,ot,ws=points(p47,prev,sid)
        for witness_path in sys.argv[9:]:
            d=np.load(Path(witness_path).with_name(Path(witness_path).stem+f'-{sid}.npz'))
            rows=np.c_[d['cell'],np.round(d['weights'],12)]
            _,keep=np.unique(rows,axis=0,return_index=True)
            nodes=np.vstack((nodes,p['cells'][d['cell'][keep]]))
            w=np.vstack((w,d['weights'][keep]))
        src=np.einsum('fi,fij->fj',w,x[nodes]);B=np.einsum('fi,fijk->fjk',w,modes[nodes])
        objects=[]
        for oid,b in owners:
            xx,ff=data[f'bone_{oid}'];obj=CorticalSurface(library,local(xx,full[b]),ff)
            objects.append((oid,b,obj))
        volume=p['cell_volumes'];active=np.flatnonzero(volume>0);t=p['cells'][active].astype(int);v=p['vertices'].astype(float)
        ref=np.linalg.det(v[t[:,1:]]-v[t[:,:1]])
        y=np.zeros(modes.shape[2]);J0=np.linalg.det(x[t[:,1:]]-x[t[:,:1]])/ref
        record=[];work=set();planes=[];rhs=[];sc=.001 # Optimize in milli sqrt(joule) coordinates, keeping physical tolerances below.
        def state(y):
            pos=src+np.einsum('nik,k->ni',B,y);gaps=[];jac=[]
            for oid,b,object_ in objects:
                dist,n,closest,faces=object_.query(local(pos,pose[b]))
                normal=Rotation.from_quat(pose[b][3:]).apply(n)
                gaps.append(dist);jac.append(np.einsum('ni,nik->nk',normal,B))
            xx=x+np.einsum('nik,k->ni',modes,y)
            J,D=jacobians(xx,t,modes,ref)
            return np.r_[*gaps,J-.20001,4.99999-J],np.vstack((*jac,D,-D)),[g.min() for g in gaps],J,xx
        if os.environ.get('NUMI_PELVIC_RESPONSE_SEED'):
            seed=np.load(Path(os.environ['NUMI_PELVIC_RESPONSE_SEED'])/f'{sid}.npz')
            y=seed['coordinates'].copy()
        g,A,min_d,J,xx=state(y);ncontact=len(src)*len(objects);initial=[float(z) for z in min_d]
        for k in range(90):
            # Work in units of distance per unit material energy to choose independent dominant witnesses.
            sizes=np.linalg.norm(A,axis=1);viol=np.flatnonzero(g < -np.r_[np.full(ncontact,2.e-8),np.full(2*len(t),1.e-6)])
            record.append(dict(iteration=k,inside_points=int((g[:ncontact]<-2e-8).sum()),minimum_signed_gaps_mm=[float(z*1000) for z in min_d],min_J=float(J.min()),max_J=float(J.max()),energy_J=float(.5*y@y)))
            if not len(viol):break
            fixed=viol[sizes[viol]<1e-13]
            if len(fixed):record[-1]['fixed_witness']=int(fixed[0]);break
            score=-g[viol]/sizes[viol];worst=viol[np.argsort(-score)]
            add=list(worst[:16])
            for part in range(len(objects)):
                regional=worst[(worst>=part*len(src))&(worst<(part+1)*len(src))]
                add.extend(regional[:8])
            wi=np.unique(add);planes.extend(A[wi]/sizes[wi,None]);rhs.extend((A[wi]@y-g[wi])/sizes[wi]/sc)
            G=np.asarray(planes);bound=np.asarray(rhs)
            sol=minimize(lambda z:(.5*z@z,z),y/sc,jac=True,constraints=[dict(type='ineq',fun=lambda z:G@z-bound,jac=lambda z:G)],method='SLSQP',options=dict(ftol=2e-10,maxiter=150))
            delta=sol.x*sc-y;record[-1].update(active=len(wi),solver=str(sol.message),linear_violation=float(max(0.,max(bound-G@sol.x))))
            if not np.isfinite(delta).all() or max(bound-G@sol.x)>2e-6:break
            if np.linalg.norm(delta)<1e-13:break
            # Retire stale-feature constraints as the closest feature changes; keep tight, current rows.
            work.update(wi.tolist())
            # The accepted source material must stay inside its constitutive domain even for a failed local solve.
            alpha=1.
            while alpha>=2**-13:
                q=y+alpha*delta;xx_trial=x+np.einsum('nik,k->ni',modes,q)
                trialJ=np.linalg.det(xx_trial[t[:,1:]]-xx_trial[t[:,:1]])/ref
                if trialJ.min()>.20 and trialJ.max()<5.:break
                alpha*=.5
            if alpha<2**-13:
                record[-1]['volume_linesearch_exhausted']=True
                break
            y=q;g,A,min_d,J,xx=state(y);record[-1]['alpha']=alpha
        r=dict(sid=sid,step=step,points=len(src),initial_signed_gaps_mm=[v*1000 for v in initial],last=record[-1],history=record)
        results[sid]=r;np.savez_compressed(out/f'{sid}.npz',positions=xx,coordinates=y,selected_constraints=np.array(sorted(work)))
        print(sid,len(record),record[0],record[-1],flush=True);(out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':fit(*map(Path,sys.argv[1:7]),int(sys.argv[7]),Path(sys.argv[8]))
