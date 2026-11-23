"""Improve positive-tet shape by solving one vertex star at a time against its cortical parent planes."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import linprog
from fit_carpal_clearance import geom
from condition_volume_quality import constraints
from audit_registered_skeleton import digest

def run(source,planes,original,plc,out,*,fixed_surfaces=False):
    z=np.load(source);c=np.load(planes);x=z['x'].astype(float);base=np.load(original)['x'].astype(float);t=z['t']
    f=z['facets'];fixed=z['locked'].copy();fixed[np.unique(f[z['markers']==15])]=True
    if fixed_surfaces:fixed[np.unique(f[z['markers']!=1000])]=True
    retained=set(v.tobytes() for v in np.load(plc)['vertex_world_m'].astype('f4'));steiner=np.array([v.tobytes() not in retained for v in base.astype('f4')])
    ids=c['node'];n=c['normal'];on=c['on_plane'];stars=[[] for _ in x];slots=[[] for _ in x]
    for k,row in enumerate(t):
        for col,j in enumerate(row):stars[j].append(k);slots[j].append(col)
    def metric(p,tt):
        v,q=geom(p[tt]);return np.minimum(v/1e-18,q/1e-5)
    def score(q):return (int(np.count_nonzero(q<1)),float(np.maximum(1-np.minimum(q,1),0).sum()))
    out.mkdir();log=[];initial=score(metric(x,t))
    for sweep in range(34):
        allq=metric(x,t);bad=np.flatnonzero(allq<1);eligible=[]
        for k in bad[np.argsort(allq[bad])]:eligible.extend(t[k].tolist())
        seen=set();improvements=0
        for i in eligible:
            if i in seen or fixed[i]:continue
            seen.add(i);rows=np.asarray(stars[i]);col=np.asarray(slots[i]);tt=t[rows];m=metric(x,tt);prior=score(m)
            if not prior[0]:continue
            wi=np.flatnonzero(ids==i);maxdist=2400. if steiner[i] and not len(wi) else 102.
            off=(x[i]-base[i])*1e6
            D,J,K=constraints(x,tt);J=J[np.arange(len(tt)),col]*1e-6;det=D+K
            # Every incident cell must maintain its original scaled residual or reach the threshold.
            scale=np.maximum(np.max(abs(J),axis=1),1e-24)
            deficiency=np.maximum(-D,0);margin=np.maximum(D,0)
            jj=-J/scale[:,None];A=np.column_stack((jj,deficiency/scale));b=margin/scale
            if len(wi):
                gap=np.einsum('ij,ij->i',n[wi],x[i]-on[wi])*1e6
                A=np.vstack((A,np.column_stack((-n[wi],np.zeros(len(wi))))));b=np.r_[b,gap-np.minimum(gap,.18)]
            trust=180. if steiner[i] else 30.
            lo=np.maximum(-trust,-maxdist-off);hi=np.minimum(trust,maxdist-off)
            if np.any(hi<lo):continue
            opts=list(zip(lo,hi))+[(0,1)]
            lp=linprog([0,0,0,-1],A_ub=A,b_ub=b,bounds=opts,method='highs')
            if not lp.success or lp.x[3]<1e-6:continue
            # Select the smallest step that realizes the locally attainable gain.
            extra=np.block([[np.eye(3),-np.ones((3,1))],[-np.eye(3),-np.ones((3,1))]])
            A2=np.vstack((np.column_stack((A[:,:3],np.zeros(len(A)))),extra));b2=np.r_[b-lp.x[3]*.995*A[:,3],np.zeros(6)]
            fit=linprog([0,0,0,1],A_ub=A2,b_ub=b2,bounds=opts[:3]+[(0,None)],method='highs')
            delta=(fit.x if fit.success else lp.x)[:3]*1e-6
            old=x[i].copy();best=None
            for fstep in (1.,.85,.65,.45,.3,.18,.1,.05,.025,.01):
                x[i]=(old+fstep*delta).astype('f4');q=metric(x,tt)
                if np.min(q)<=0 or np.any(q<np.minimum(m,1.)-1e-4):continue
                if len(wi) and np.any(np.einsum('ij,ij->i',n[wi],x[i]-on[wi])<0):continue
                sc=score(q)
                if sc<prior and (best is None or sc<best[0]):best=(sc,x[i].copy())
            x[i]=best[1] if best is not None else old
            improvements+=best is not None
        sc=score(metric(x,t));log.append((sweep,improvements,sc));print(log[-1],flush=True)
        if not improvements or not sc[0]:break
    res={k:z[k] for k in z.files};res['x']=x.astype('f4');np.savez_compressed(out/'domain.npz',**res)
    (out/'result.json').write_text(json.dumps(dict(input=digest(source),planes=digest(planes),initial=initial,end=score(metric(x,t)),sweeps=log,steiner_max_move_m=float(np.linalg.norm((x-base)[steiner],axis=1).max())),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
