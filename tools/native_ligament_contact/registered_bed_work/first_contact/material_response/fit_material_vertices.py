"""Repair low-quality material stars using exact incident tetrahedra at emitted float precision."""
from pathlib import Path
import json,sys
import numpy as np
from condition_loaded_patch import ratios
from condition_volume_quality import constraints, EDGE
from compile_connected_joint import digest

def run(mesh,planes,witness,out):
    z=np.load(mesh)
    base=z['x'].astype(float)
    x=base.copy()
    t=z['t']
    fixed=z['locked'].copy()
    fixed[np.unique(z['facets'][z['markers']==15])]=True
    fixed[np.load(witness)]=True
    d=np.load(planes)
    ids,normal,anchor=(d[k] for k in ('node','normal','on_plane'))
    init=ratios(x,t)
    history=[]
    out.mkdir()
    fibonacci=np.arange(96)+.5
    zz=1.-2*fibonacci/len(fibonacci)
    phi=fibonacci*(np.pi*(3.-np.sqrt(5.)))
    u=np.column_stack((np.sqrt(1.-zz*zz)*np.cos(phi),np.sqrt(1.-zz*zz)*np.sin(phi),zz))
    steps=np.geomspace(.03,480.,23)*1.e-6
    for sweep in range(15):
        count=0
        values=ratios(x,t)
        order=np.flatnonzero(values<1.)
        for cell in order[np.argsort(values[order])]:
            if ratios(x,t[cell:cell+1])[0]>=1.:continue
            best=None
            for j in t[cell]:
                if fixed[j]:continue
                sel,corner=np.where(t==j)
                mask=ids==j
                N=normal[mask]
                old=ratios(x,t[sel])
                target=np.flatnonzero(sel==cell)[0]
                _,grad,_=constraints(x,t[cell:cell+1])
                direct=grad[0,np.flatnonzero(t[cell]==j)[0]]
                dirs=[u,direct[None]/np.linalg.norm(direct)]
                if len(N):
                    for n in N:
                        dirs.extend((n[None],u-(u@n)[:,None]*n))
                    if len(N)>1:dirs.append(np.cross(N,np.roll(N,1,axis=0)))
                ds=np.vstack(dirs);ds=ds[np.linalg.norm(ds,axis=1)>1.e-9]
                ds/=np.linalg.norm(ds,axis=1)[:,None]
                tries=(x[j]+(ds[:,None]*steps[None,:,None]).reshape(-1,3)).astype('f4').astype(float)
                keep=np.linalg.norm(tries-base[j],axis=1)<=.0005
                if len(N):
                    gaps=(x[j]-anchor[mask])*N
                    initial=((base[j]-anchor[mask])*N).sum(1)
                    keep&=np.all((tries-x[j])@N.T + gaps.sum(1)>=np.minimum(initial,0.)-2.e-9,axis=1)
                pts=x[t[sel]]
                for group in np.array_split(tries[keep],max(1,int(keep.sum())//96)):
                    if not len(group):continue
                    quad=np.broadcast_to(pts,(len(group),)+pts.shape).copy()
                    quad[:,np.arange(len(sel)),corner]=group[:,None,:]
                    a=quad[:,:,1:]-quad[:,:,:1]
                    det=np.einsum('nci,nci->nc',a[:,:,0],np.cross(a[:,:,1],a[:,:,2]))
                    edge=quad[:,:,EDGE[:,0]]-quad[:,:,EDGE[:,1]]
                    L=np.sqrt(np.einsum('ncej,ncej->nce',edge,edge).max(2))
                    v=np.minimum(det/6.e-18,det/(1.e-5*L**3))
                    ok=np.flatnonzero((v.min(1)>0.)&np.all(v[:,old>=1.]>=1.,axis=1)&(v[:,target]>old[target]))
                    if len(ok)==0:continue
                    cost=np.maximum(0.,1.01-v[ok]).sum(1)
                    prior=float(np.maximum(0.,1.01-old).sum())
                    bad=(v[ok]<1.).sum(1)
                    change=bad-(old<1.).sum()
                    displacement=np.linalg.norm(group[ok]-x[j],axis=1)
                    rank=np.lexsort((displacement,cost,change));idx=rank[0]
                    score=(int(change[idx]),float(cost[idx]-prior),float(displacement[idx]))
                    if score[:2]<(0,-1.e-7) and (best is None or score<best[0]):
                        best=(score,j,group[ok[idx]])
            if best is not None:
                _,j,p=best;x[j]=p;count+=1
        v=ratios(x,t);history.append(dict(sweep=sweep,moves=count,low=int((v<1).sum()),min=float(v.min())))
        print(history[-1],flush=True)
        np.savez_compressed(out/'domain.npz',**{k:(x.astype('f4') if k=='x' else z[k]) for k in z.files})
        if not count or not (v<1.).any():break
    delta=np.linalg.norm(x-base,axis=1)
    doc=dict(parent=digest(mesh),cortex=digest(planes),protected_witness=digest(witness),mesh=digest(out/'domain.npz'),history=history,
             moved=int((delta>0).sum()),max_displacement_m=float(delta.max()),initial_low=int((init<1).sum()),final_low=int((v<1).sum()))
    (out/'result.json').write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__': run(*map(Path,sys.argv[1:]))
