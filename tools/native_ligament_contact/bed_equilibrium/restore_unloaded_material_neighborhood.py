"""Preserve a previously admissible material neighborhood away from bed and attachment constraints."""
import json,shutil,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from screen_registered_dense_posture import load_geometry,measure
from untangle_registered_donor_field import differentials
from recover_material_thickness import audit
from construct_registered_donor_field import errors

def main(full,donor,reference,seed,out):
    x,tgt,owner,t,mat,outer,_,pins=load_geometry(full,donor)
    for src in (reference,seed):
        if json.loads((src/'result.json').read_text())['source']!=pins:raise ValueError('mismatched tissue pose')
    y=x+np.load(seed/'displacement.npy');start=y.copy();previous=x+np.load(reference/'displacement.npy')
    rr=np.concatenate([differentials(x,t[k:k+32768])[0] for k in range(0,len(t),32768)])
    if audit(previous,t,rr)[3].any():raise ValueError('invalid construction reference')
    z=np.load(seed/'construction-boundary.npz')
    locked=z['node'];bed=np.flatnonzero(outer&(y[:,2]<=1e-9))
    total=[]
    for it in range(3):
        _,_,_,bad=audit(y,t,rr);total.append(int(bad.sum()))
        if not bad.any():break
        points=np.unique(t[bad]);influence=cKDTree(y[points])
        dist=influence.query(y)[0]
        separation=min(float(dist[locked].min()),float(dist[bed].min()))
        if separation<=.001:break
        best=None
        for radius in (.003,.007,.014,.025):
            transition=min(radius,separation*.85)
            core=transition*.25
            a=np.clip((dist-core)/(transition-core),0,1)
            weight=(1-a)*(1-a)*(1-a)*(1+3*a+6*a*a)
            trial=y+(previous-y)*weight[:,None];trial[dist<=core]=previous[dist<=core]
            if np.any(trial[:,2]<0) or not np.array_equal(y[locked],trial[locked]) or not np.array_equal(trial[bed],y[bed]):continue
            changed=np.any(trial!=y,axis=1);inc=changed[t].any(1)
            n=audit(trial,t[inc],rr[inc])[3].sum()+bad[~inc].sum()
            key=(int(n),float(np.max(np.linalg.norm(trial-y,axis=1))))
            if best is None or key<best[0]:best=(key,trial)
        if best is None or best[0][0]>=bad.sum():break
        y=best[1]
    j,h,det,bad=audit(y,t,rr);out.mkdir(exist_ok=False);np.save(out/'displacement.npy',y-x)
    shutil.copyfile(seed/'construction-boundary.npz',out/'construction-boundary.npz')
    rec=dict(source=pins,history=total,failed=int(bad.sum()),quality=measure(y,owner,t,mat),minimum_J=float(j.min()),maximum_J=float(j.max()),
             min_node_height_m=float(y[:,2].min()),touching_outer_nodes=int(((y[:,2]==0)&outer).sum()),max_change=errors(y,start))
    (out/'result.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(rec['history'],rec['failed'],flush=True)
if __name__=='__main__':main(*(Path(v).resolve() for v in sys.argv[1:]))
