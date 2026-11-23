"""Contract interior material edges to a feasible cortical-clear point of their joined link."""
from pathlib import Path
from itertools import combinations
import sys,json
import numpy as np
from scipy.optimize import linprog
from collapse_material_slivers import manifold,collapse
from condition_volume_quality import constraints
from fit_carpal_clearance import geom
from verify_carpal_mesh import classify
from compile_connected_joint import digest


def rate(x,t):
    v,q=geom(x[t])
    return np.minimum(v/1.e-18,q/1.e-5)


def run(source,out):
    z=np.load(source);x=z['x'].astype(float);t=z['t'].copy();roles=z['role'].copy()
    keep=z['markers']!=1000;f=z['facets'][keep].copy();mk=z['markers'][keep].copy()
    protect=z['locked'].copy();protect[np.unique(f[mk!=1516])]=True
    initial=rate(x,t);hist=[];out.mkdir()
    for iteration in range(12):
        q=rate(x,t);bad=np.flatnonzero(q<1.)
        if not len(bad):break
        edges=sorted({tuple(sorted((int(a),int(b)))) for i in bad for a,b in combinations(t[i],2)},
                     key=lambda e:np.linalg.norm(x[e[0]]-x[e[1]]))
        best=None
        for a,b in edges:
            length=np.linalg.norm(x[a]-x[b])
            if length>.003:break
            if protect[a] or protect[b]:continue
            pop=np.isin(t,[a,b]).any(1);tt=t[pop];m=roles[pop]
            if not manifold(tt,a,b) or any(not manifold(tt[m==v],a,b) for v in np.unique(m)):continue
            touch=np.isin(f,[a,b]).any(1);surf=f[touch];ma=mk[touch]
            if any(not manifold(surf[ma==v],a,b) for v in np.unique(ma)):continue
            if any(np.any(surf==j) for j in (a,b)) and not all(np.any(surf==j) for j in (a,b)):continue
            cells,used=collapse(tt,a,b);cells=cells[used];g=np.where(cells==b)[1];rows=np.arange(len(cells))
            if not len(cells) or len(g)!=len(cells):continue
            faces,uf=collapse(surf,a,b)
            norm=np.cross(x[surf[:,1]]-x[surf[:,0]],x[surf[:,2]]-x[surf[:,0]]) if len(surf) else np.empty((0,3))
            norm/=np.maximum(1.e-25,np.linalg.norm(norm,axis=1))[:,None]
            incident=np.array([x[a] if a in row else x[b] for row in surf]).reshape(-1,3)
            initial_b=x[b].copy();xa=x[a].copy();center=(xa+initial_b)*.5
            x[b]=center;improved=None
            for rep in range(10):
                e,J,K=constraints(x,cells)
                relative=e/K+1.
                A=-J[rows,g]*1e-6/K[:,None]
                lhs=np.column_stack((A,np.ones(len(e))));rhs=relative
                # A surviving interface triangle may move tangentially within its source footprint.
                if len(norm):
                    wall=np.column_stack((norm,np.zeros(len(norm))))
                    dis=1.e6*np.einsum('ij,ij->i',norm,x[b]-incident)
                    lhs=np.vstack((lhs,wall,-wall));rhs=np.r_[rhs,150.-dis,150.+dis]
                off=(x[b]-center)*1.e6;lo=np.maximum(-400.,-850.-off);hi=np.minimum(400.,850.-off)
                res=linprog([0.,0.,0.,-1.],A_ub=lhs,b_ub=rhs,bounds=[*zip(lo,hi),(None,1.12)],method='highs')
                if not res.success:break
                original=x[b].copy();local=None
                for alpha in (1.,.7,.4,.2,.1,.03):
                    x[b]=(original+alpha*res.x[:3]*1.e-6).astype('f4')
                    val=rate(x,cells);s=(int((val<1.).sum()),float(np.maximum(1.008-val,0.).sum()),-float(min(val.min(),1.05)))
                    if local is None or s<local[0]:local=s,x[b].copy()
                    if val.min()<=0 or (len(surf) and np.any(np.einsum('ij,ij->i',norm[uf],np.cross(x[faces[uf,1]]-x[faces[uf,0]],x[faces[uf,2]]-x[faces[uf,0]]))<=0)):continue
                    if improved is None or s<improved[0]:improved=s,x[b].copy()
                x[b]=local[1]
                if local[0][0]==0 or np.linalg.norm(x[b]-original)<1e-9:break
            x[b]=initial_b
            if improved is None:continue
            before=q[pop];score=(improved[0][0]-int((before<1.).sum()),improved[0][1]-float(np.maximum(1.008-before,0.).sum()),length)
            if score[0]<0 and (best is None or score<best[0]):best=score,a,b,improved[1]
        if best is None:break
        score,a,b,p=best;x[b]=p;t,keep=collapse(t,a,b);t=t[keep];roles=roles[keep];f,keep=collapse(f,a,b);f=f[keep];mk=mk[keep]
        hist.append(dict(remove=a,survivor=b,point_m=p.tolist(),gain=-score[0]));print(hist[-1],flush=True)
    role,outer=classify(x,t,f,mk)
    if not np.array_equal(role,roles) or len(np.unique(np.sort(t,axis=1),axis=0))!=len(t):raise ValueError('Contraction damaged material partition')
    v=rate(x,t);np.savez_compressed(out/'domain.npz',x=x.astype('f4'),t=t,facets=f,role=roles,markers=mk,outer=outer,locked=z['locked'],fence=z['fence'])
    (out/'result.json').write_text(json.dumps(dict(input=digest(source),output=digest(out/'domain.npz'),before=int((initial<1.).sum()),after=int((v<1.).sum()),min_ratio=float(v.min()),operations=hist),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
