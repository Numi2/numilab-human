"""Solve compression-only cortical contact using the cut-material elastic response at a recorded pose."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize,nnls
from material_context import sha
from pose_glutes import states,parts,model
from bind_material_ports import gram
from qualify_material_motion import frame,local

def poses_at(run,step):
    rc=json.loads((run/'terminal.receipt.json').read_text());pose=states(run/'resting-body-motion-partition.csv',run/'terminal.receipt.json')
    for row in csv.DictReader((run/'resting-body-motion-partition.csv').open()):
        if int(row['body_index'])==1:
            k=int(row['accepted_step']);pose[k][1]=np.r_[[float(row['root_'+a+'_m'])+float(row['com_relative_root_'+a+'_m']) for a in 'xyz'],[float(row['orientation_'+a]) for a in 'xyzw']]
    end={r['body_index']:frame(r) for r in rc['accepted_body_poses']};off={int(b):v for b,v in json.loads((run/'glutes/motion.json').read_text())['frame_correction_local_m'].items()};off[1]=local(end[1][:3],pose[rc['accepted_step']][1])
    current=pose[step]
    if step:
        for b,d in off.items():current[b][:3]+=Rotation.from_quat(current[b][3:]).apply(d)
    return pose[0],current

def current_bank(sid,p47,prev,response,run,step):
    full={r['body_index']:frame(r) for r in json.loads((prev/'native-release-motion/step-8.receipt.json').read_text())['initial_anatomical_registration']['body_poses']}
    cfg=json.loads((run/'launch.json').read_text())['argv'];asset=Path(cfg[cfg.index('--soft-tissue-payload')+1])
    base,pose=poses_at(run,step);p=np.load(p47/'projected'/f'{sid}.npz');port=np.load(p47/'ports'/f'{sid}.npz');res=np.load(response/f'{sid}.npz')
    if step==0:x=np.load(run/'glutes'/f'entrance-{sid}.npz')['xyz'].astype(float)
    elif step==json.loads((run/'terminal.receipt.json').read_text())['accepted_step']:x=np.load(run/'glutes'/f'accepted-{sid}.npz')['xyz'].astype(float)
    else:x=model(parts(asset,sid),pose,1)[0].astype('f4').astype(float)
    select=port['target_body']>=0
    nodes=p['cells'][p['port_owner_cell'][select]];w=p['port_shape'][select].mean(1)
    rot=(Rotation.from_quat(pose[128][3:])*Rotation.from_quat(full[128][3:]).inv()).as_matrix()
    modes=np.einsum('ij,njk->nik',rot,res['modes'])
    src=np.einsum('fi,fij->fj',w,x[nodes]);rows=np.einsum('fi,fijk->fjk',w,modes[nodes])
    tri=port['target_triangle'][select].astype(float);area=p['port_vertices'][select]
    normal=-np.cross(area[:,1]-area[:,0],area[:,2]-area[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
    faces=np.zeros_like(tri);world_n=np.zeros_like(normal);owners=port['target_body'][select]
    for b in np.unique(owners):
        idx=owners==b;rot=(Rotation.from_quat(pose[b][3:])*Rotation.from_quat(full[b][3:]).inv())
        faces[idx]=rot.apply((tri[idx]-full[b][:3]).reshape(-1,3)).reshape(-1,3,3)+pose[b][:3];world_n[idx]=rot.apply(normal[idx])
    g=np.einsum('ij,ij->i',src-faces[:,0],world_n);A=np.einsum('fi,fik->fk',world_n,rows)
    return p,x,modes,src,rows,faces,world_n,owners,g,A

def unilateral(A,g,tol):
    # Minimum elastic energy with signed gap >= 0. Generate violated rows, then recover nonnegative reactions.
    length=np.linalg.norm(A,axis=1);fixed=length<1.e-13
    if np.any(g[fixed]<-tol):raise ValueError(('Prescribed incompatible contact row',g[fixed].min()))
    select=np.flatnonzero(~fixed);B=A[select]/length[select,None];d=-g[select]/length[select];y=np.zeros(A.shape[1]);working=[]
    for iteration in range(110):
        gap=A@y+g;k=int(np.argmin(gap))
        if gap[k]>=-tol:break
        j=int(np.searchsorted(select,k))
        if j in working:raise ValueError(('Unresolved active contact',k,gap[k]))
        working.append(j);G=B[working];bound=d[working]
        Q=minimize(lambda z:(.5*np.dot(z,z),z),y,jac=True,constraints=[dict(type='ineq',fun=lambda z:G@z-bound,jac=lambda z:G)],method='SLSQP',options=dict(ftol=2.e-14,maxiter=220))
        y=Q.x
        if min(G@y-bound)<-1.e-7:raise ValueError(('Infeasible material contact',Q.message,float(min(G@y-bound))))
    active=np.array(working,dtype=int);binding=active[abs(B[active]@y-d[active])<1e-7]
    lam=np.zeros(len(g))
    if len(binding):
        dual,_=nnls(B[binding].T,y,maxiter=2000);lam[select[binding]]=dual/length[select[binding]]
    return y,lam,dict(iterations=iteration+1,min_gap_m=float(min(A@y+g)),active_rows=int(sum(lam>1e-9)),equilibrium=float(np.linalg.norm(A.T@lam-y)),energy_J=float(.5*np.dot(y,y)))
def run(p47,prev,response,native,out,step):
    out.mkdir(exist_ok=True);results={}
    for sid in (29,30):
        p,x,modes,src,rows,f,n,owners,g,A=current_bank(sid,p47,prev,response,native,step)
        inside=gram(src[:,None],f)[:,0].min(1)>-1e-8;eligible=inside.copy();y=np.zeros(A.shape[1]);z=np.zeros(len(g))
        for refresh in range(6):
            y,load,r=unilateral(A[eligible],g[eligible],1.e-8);z[:]=0.;z[eligible]=load
            current=src+np.einsum('fij,j->fi',rows,y);new=gram(current[:,None],f)[:,0].min(1)>-1e-8
            if not np.any(new&~eligible&((g+A@y)<-1e-8)):break
            eligible|=new
        moved=x+np.einsum('nik,k->ni',modes,y)
        cell=p['cells'][p['cell_volumes']>0];ref=p['vertices'].astype(float)
        J=lambda v:np.linalg.det(v[cell[:,1:]]-v[cell[:,:1]])/np.linalg.det(ref[cell[:,1:]]-ref[cell[:,:1]])
        r.update(initial_in_facet_penetrations=int((g[inside]<-1e-6).sum()),initial_min_gap_m=g[inside].min(),final_in_facet_penetrations=int(((g+A@y)[new]<-1e-6).sum()),released_outside_facet_force_N=z[~new].sum(),max_node_displacement_m=float(np.linalg.norm(moved-x,axis=1).max()),min_J=J(moved).min(),max_J=J(moved).max(),material_force_N=z.sum(),step=step,refresh=refresh)
        np.savez_compressed(out/f'{sid}.npz',positions=moved,coordinates=y,multipliers=z,owners=owners,gap=g+A@y)
        results[sid]=r;print(sid,r,flush=True)
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:6]),int(sys.argv[6]))
