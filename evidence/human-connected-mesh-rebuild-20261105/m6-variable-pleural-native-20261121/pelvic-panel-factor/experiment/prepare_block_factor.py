"""Precompute 32-row triangular panel solves from retained regional factors."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from material_context import sha

def build(assets,out):
    meta=json.loads((assets/'manifest.json').read_text());out.mkdir()
    rows=np.fromfile(assets/'pelvic-factor-rows.bin','u4').reshape(-1,4)
    col=np.fromfile(assets/'pelvic-factor-columns.bin','u4');val=np.fromfile(assets/'pelvic-factor-values.bin','f4')
    order=np.fromfile(assets/'pelvic-factor-order.bin','u4')
    levels=np.fromfile(assets/'pelvic-factor-levels.bin','u4').reshape(-1,2)
    desc=[];inv=[];indices=[];weights=[];dims=[];worst=0.
    for side,r in enumerate(meta['regional_factor']):
        offsets=[];n=r['coordinates'];start=r['coordinate_start']
        for key in ('lower','upper'):
            at,count=r[key];lv=levels[at:at+count];idx=order[int(lv[0,0]):int(sum(lv[-1]))];rr=rows[idx]
            if len(idx)!=n or set(rr[:,2])!=set(range(start,start+n)):raise ValueError('Invalid stage order')
            rr=rr[np.argsort(rr[:,2])];offsets.append(len(inv))
            for lo in range(0,n,32):
                upper=min(n,lo+32);D=np.eye(32)
                for i in range(lo,lo+32):
                    first=len(indices)
                    if i<upper:
                        pos,length,_,bits=rr[i];end=int(pos)+int(length)
                        c=col[pos:end].astype(int)-start;v=val[pos:end]
                        if np.any(c>=i) if key=='lower' else np.any(c<=i):raise ValueError('Nontriangular factor')
                        inside=(c>=lo)&(c<upper)
                        D[i-lo,c[inside]-lo]=v[inside]
                        scale=np.array(bits,'u4').view('f4').item()
                        D[i-lo,i-lo]=1./scale
                        indices.extend(c[~inside].tolist());weights.extend(v[~inside].tolist())
                    desc.append((first,len(indices)-first))
                X=solve_triangular(D,np.eye(32),lower=key=='lower').astype('f4')
                err=np.max(abs(D@X-np.eye(32)))/(np.linalg.norm(D,np.inf)*np.linalg.norm(X,np.inf))
                if not np.isfinite(err) or err>2e-7:raise ValueError('Panel solve inaccurate')
                worst=max(err,worst);inv.append(X)
        dims.append(offsets)
    names=('factor-panel-spans','factor-panel-indices','factor-panel-values','factor-panel-coefficients')
    for name,data,typ in zip(names,(desc,indices,weights,inv),('u4','u4','f4','f4')):
        np.asarray(data,typ).tofile(out/(name+'.bin'))
    result={'source_manifest':sha(assets/'manifest.json'),'width':32,'offsets':dims,'worst_panel_backward_error':worst,
        'files':{n:sha(out/(n+'.bin')) for n in names}}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':build(*map(Path,sys.argv[1:]))
