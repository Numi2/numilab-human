import json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve_triangular
from material_context import sha
a,p,folder=map(Path,sys.argv[1:])
meta=json.loads((a/'manifest.json').read_text())
def load(k,typ,width):return np.fromfile(a/(k+'.bin'),typ).reshape(-1,width)
R=load('pelvic-factor-rows','u4',4);C=load('pelvic-factor-columns','u4',1).ravel();V=load('pelvic-factor-values','f4',1).ravel()
O=load('pelvic-factor-order','u4',1).ravel();LV=load('pelvic-factor-levels','u4',2);perm=load('pelvic-factor-perm','u4',2)
sc=load('pelvic-factor-scale','f4',2);proj=load('pelvic-projector-rows','u4',2);ids=load('pelvic-projector-cols','u4',1).ravel();coef=load('pelvic-projector-vals','f4',1).ravel()
result={'manifest':sha(a/'manifest.json'),'panel_manifest':sha(p/'manifest.json'),'fields':{}}
for name in ['accepted','random','localized']:
    case=folder/name
    def field(file):return np.fromfile(case/file,'f4').reshape(-1,4)[:,:3].astype(float).ravel()
    rhs=field('rhs.bin');old=field('original-output.bin');new=field('panel-output.bin')
    times=np.loadtxt(case/'timing.csv',delimiter=',',skiprows=1)
    projected=rhs.copy()
    for k,(lo,n) in enumerate(proj):
        if n:projected[k]=coef[lo:lo+n].astype(float)@rhs[ids[lo:lo+n]]
    out={};result['fields'][name]={'median_gpu_ms':{'original':np.median(times[:,1]),'panel':np.median(times[:,2])},'sides':out}
    for dim in meta['regional_factor']:
        start=dim['coordinate_start'];n=dim['coordinates'];sl=slice(start,start+n);ops=[]
        for key in ('lower','upper'):
            at,count=dim[key];levels=LV[at:at+count];idx=O[int(levels[0,0]):int(sum(levels[-1]))]
            row=[];col=[];data=[]
            for lo,num,g,recip in R[idx]:
                rr=int(g)-start;en=int(lo)+int(num);row.append(rr);col.append(rr);data.append(1./np.array(recip,'u4').view('f4').item())
                row.extend([rr]*int(num));col.extend((C[lo:en].astype(int)-start).tolist());data.extend(V[lo:en].tolist())
            ops.append(coo_matrix((data,(row,col)),(n,n)).tocsr())
        L,U=ops;g=projected[perm[sl,0]]*sc[sl,0];oracle=spsolve_triangular(U,spsolve_triangular(L,g,lower=True),lower=False)
        y=oracle*sc[sl,1];r1=old[perm[sl,1]];r2=new[perm[sl,1]]
        norm=max(np.linalg.norm(y),1e-30);basis=max(np.linalg.norm(g),1e-30)
        def measure(x):
            return {'relative_solution_error':np.linalg.norm(x-y)/norm,'relative_factor_residual':np.linalg.norm(L@(U@(x/sc[sl,1]))-g)/basis}
        out[str(dim['muscle'])]={'original':measure(r1),'panel':measure(r2),'relative_to_original':np.linalg.norm(r2-r1)/max(np.linalg.norm(r1),1e-30),'reference_nonzero':bool(np.any(g))}
(folder/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['fields'],indent=2))
