"""Check a changed volume against its source nodal inertia, density, and pose moments."""
from pathlib import Path
import json
import numpy as np
from transfer_cavities import intersection_fraction
from scipy.optimize import linprog

def run():
    a=np.load('cavity.npz');p=json.loads(Path('proposals.json').read_bytes())[0]
    rows=a['rows']; old=np.searchsorted(rows,p['old']);labels=a['labels'][:,1:]; t=a['local']; new=np.searchsorted(a['ids'],p['new']);xyz=a['xyz'];x=xyz[0].astype('f4').astype('f8')
    m=a['mass'];vol=lambda c:np.linalg.det(x[c[:,1:]]-x[c[:,:1]])/6
    w=np.array([[intersection_fraction(s,c,x) for c in new] for s in t[old]])
    mm=m[old]@w; results={}
    for ring in (0,1,2):
        near=old
        for _ in range(ring):
            near=np.flatnonzero(np.isin(t,np.unique(t[near])).any(1)&(labels==p['label']).all(1))
        more=np.setdiff1d(near,old);dst=np.vstack((new,t[more]));centers=np.concatenate([mm,m[more]])
        ids=np.unique(dst);A=lambda ts:np.stack([(ts==n).any(1) for n in ids]).astype(float)/4
        K=A(dst);rhs=A(t[near])@m[near]
        scale=1e7;v=vol(dst);low=849.99*v;high=950.01*v
        errs=[]
        eps=1e-15
        for pos in xyz:
            off=pos-pos[ids[0]]
            q=off[t[near]].mean(1);qn=off[dst].mean(1)
            err=np.linalg.norm(centers@qn-m[near]@q);errs.append(float(err))
        results[ring]=dict(old=len(near),new=len(dst),exact_overlay_moment_error_kg_m=errs,exact_overlay_max_nodal_diff_kg=np.max(abs(K@centers-rhs)),density_range=[np.min(centers/v),np.max(centers/v)])
        # Solve for a quarter-per-node uniform cell mass assignment first. Then try
        # preserving mass and first moments in all measured configurations.
        for kind,basis in [('nodal',np.eye(len(ids))),('pose_moments',np.vstack([np.ones(len(ids)),*((pos[ids]-pos[ids].mean(0)).T for pos in xyz)]))]:
            E=basis@K; b=basis@rhs
            U,S,V=np.linalg.svd(E,full_matrices=False);good=S>S[0]*1e-13
            if np.linalg.norm(b-U[:,good]@(U[:,good].T@b))>max(np.linalg.norm(b)*1e-11,eps):results[ring][kind]='inconsistent linear equations';continue
            sol=linprog(np.zeros(len(v)),A_eq=V[good],b_eq=(U[:,good].T@b)/S[good]*scale,bounds=np.column_stack((low,high))*scale,method='highs')
            results[ring][kind]=sol.message
            if sol.success:results[ring][kind+'_max_error']=max(abs(E@(sol.x/scale)-b))
    print(json.dumps(results,indent=2));Path('inertia.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':run()
