"""Remove residual negative cortical gaps by moving connected tissue neighborhoods coherently."""
import json,sys
from pathlib import Path
import numpy as np
from fit_carpal_clearance import geom
from audit_registered_skeleton import digest

def run(source,planes,out):
    z=np.load(source);x=z['x'].astype(float);t=z['t'];d=np.load(planes);ids=d['node'];n=d['normal'];p=d['on_plane']
    fix=z['locked'].copy();fix[np.unique(z['facets'][z['markers']==15])]=True
    initial=x.copy()
    def stat(y):
        g=np.einsum('ij,ij->i',n,y[ids]-p)*1e6;v,q=geom(y[t]);return (int((v<=0).sum()),int((g<0).sum()),float(np.maximum(.05-g,0).sum()),int(((v<=1.e-18)|(q<1.e-5)).sum())),g
    start,_=stat(x);rows=[]
    for j in range(25):
        s,g=stat(x)
        if s[1]==0:break
        best=None
        for i in np.unique(ids[g<0]):
            at=int(np.argmin(np.where(ids==i,g,np.inf)))
            incident=np.any(t==i,axis=1);star=np.unique(t[incident])
            # Whole-star translations preserve volumes of incident slivers.
            for ring in range(3):
                if ring:star=np.unique(t[np.isin(t,star).any(1)])
                for rad in (0.,30.,100.,500.,1500.,3500.):
                    w=np.zeros(len(x));w[star]=1.
                    if rad:
                        delta=np.linalg.norm(x[star]-x[i],axis=1)/(rad*1e-6)
                        w[star]=np.exp(-.5*delta*delta)
                    w[fix]=0.
                    for alpha in (1.,1.35,1.9,.5):
                        shift=alpha*(.17-g[at])*1e-6*n[at];y=(x+w[:,None]*shift).astype('f4').astype(float)
                        sc,_=stat(y)
                        if sc[0] or sc[3]>start[3]+2:continue
                        if sc<s and (best is None or sc<best[0]):best=(sc,y,int(i),ring,rad)
        if best is None:break
        sc,x,i,ring,rad=best;rows.append([i,ring,rad,sc]);print(rows[-1],flush=True)
    out.mkdir();res={k:z[k] for k in z.files};res['x']=x.astype('f4');np.savez_compressed(out/'domain.npz',**res)
    (out/'result.json').write_text(json.dumps(dict(input=digest(source),planes=digest(planes),start=start,end=stat(x)[0],moves=rows,max_shift_m=float(np.linalg.norm(x-initial,axis=1).max())),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
