"""Count exact crossings on native Float32 geometry for a complete connected cavity."""
import argparse,ctypes,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from emit_skin_source_registration import sha
class Surface:
    def __init__(self,x,f):
        p=x[f];self.lo=p.min(1);self.hi=p.max(1);self.c=(self.hi+self.lo)*.5
        self.r=((self.hi-self.lo)*.5).max(1); self.groups=[]
        for k in np.unique(np.frexp(self.r)[1]):
            ii=np.flatnonzero(np.frexp(self.r)[1]==k)
            self.groups.append((ii,float(self.r[ii].max()),cKDTree(self.c[ii])))
def main():
    ap=argparse.ArgumentParser()
    for k in ['src','proposal','lib','report']:ap.add_argument(k,type=Path)
    ap.add_argument('--layers',action='store_true');args=ap.parse_args()
    s=np.load(args.src);g=np.load(args.proposal);x=[g['x'].astype('f8')];faces={k:g[k] for k in ['O','I','D','C15','C16']}
    for i in (67,68,73,74):
        k=str(i);faces[k]=s['f'+k]+sum(map(len,x));x.append(s['x'+k].astype('f8'))
    x=np.ascontiguousarray(np.concatenate(x));f=np.ascontiguousarray(np.concatenate(list(faces.values())),dtype='u4');offset={};idx={}
    start=0
    for k,ff in faces.items():
        offset[k]=start;start+=len(ff);idx[k]=Surface(x,ff)
    arg=np.ctypeslib.ndpointer(flags='C_CONTIGUOUS');job=ctypes.CDLL(str(args.lib.resolve())).batch;job.argtypes=[arg,arg,arg,ctypes.c_uint,arg];job.restype=None
    def compare(a,b):
        A,B=idx[a],idx[b];count=0;matches=set();w=[];total=0
        for bi,R,tree in B.groups:
            for start in range(0,len(A.r),256):
                ai=np.arange(start,min(start+256,len(A.r)))
                lists=tree.query_ball_point(A.c[ai],A.r[ai]+R+2e-14,p=np.inf);sizes=np.fromiter(map(len,lists),int,count=len(ai))
                if not sizes.sum():continue
                aa=np.repeat(ai,sizes);bb=bi[np.concatenate(lists).astype(int)]
                ok=(np.all(A.lo[aa]<=B.hi[bb],1)&np.all(B.lo[bb]<=A.hi[aa],1))
                if a==b:ok&=aa<bb
                aa,bb=aa[ok],bb[ok]
                if not len(aa):continue
                pair=np.ascontiguousarray(np.column_stack([aa+offset[a],bb+offset[b]]),'u4')
                hit=np.zeros(len(aa),'u1');job(x,f,pair,len(pair),hit);take=np.flatnonzero(hit);total+=len(aa)
                if len(take):
                    count+=len(take);matches.update(aa[take].tolist());w.extend(zip(aa[take].tolist(),bb[take].tolist()))
        return dict(pairs=count,faces=len(matches),aabb_pairs=total,witnesses=w[:8]),np.asarray(w,'i4').reshape(-1,2)
    checks=[(k,str(i)) for k in 'OID' for i in (67,68,73,74)]
    if args.layers:
        layers=['O','I','D','C15','C16'];checks.extend((k,l) for j,k in enumerate(layers) for l in layers[j:])
    data={};w={}
    for a,b in checks:
        key=a+'_'+b;data[key],w[key]=compare(a,b)
    np.savez_compressed(args.report.with_suffix('.npz'),**w)
    out=dict(input_sha256=sha(args.src),proposal_sha256=sha(args.proposal),scope='entire local material cavity including exact pinned seam; four named native muscle banks in the captured reference pose',comparisons=data,layer_vs_muscle_pairs={k:sum(data[k+'_'+str(i)]['pairs'] for i in (67,68,73,74)) for k in 'OID'})
    print(out['layer_vs_muscle_pairs'],flush=True)
    if args.layers:print('shell',sum(data[a+'_'+b]['pairs'] for a,b in checks if not b.isdigit()),flush=True)
    args.report.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
