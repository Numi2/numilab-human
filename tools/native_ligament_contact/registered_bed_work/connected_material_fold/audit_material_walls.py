"""Audit distinct material triangles across the three physical boundaries."""
import ctypes,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def run(x, groups,out,library):
    x=np.ascontiguousarray(x,dtype='f8')
    tri=np.ascontiguousarray(np.concatenate(groups),dtype='u4')
    layers=np.repeat(np.arange(len(groups)),[len(g) for g in groups])
    pts=x[tri];lo=pts.min(1);hi=pts.max(1);c=(lo+hi)/2
    # L-infinity balls conservatively enclose each triangular AABB. Bin by
    # radius so a few long facets do not flood the small-facet neighborhood.
    radius=np.max((hi-lo)/2,axis=1)
    if min(radius)<=0:raise ValueError("Degenerate point face")
    bins=np.frexp(radius)[1]; starts=np.r_[0,np.cumsum([len(g) for g in groups])]
    classes=[np.flatnonzero(bins==k) for k in np.unique(bins)]
    trees=[cKDTree(c[ids]) for ids in classes]
    del pts
    arg=np.ctypeslib.ndpointer(flags='C_CONTIGUOUS')
    job=ctypes.CDLL(str(library.resolve())).batch
    job.argtypes=[arg,arg,arg,ctypes.c_uint,arg];job.restype=None
    counts=np.zeros((len(groups),len(groups)),dtype='i8');samples={};narrow=0
    st=time.monotonic()
    for i,aa in enumerate(classes):
        for j in range(i,len(classes)):
            bb=classes[j]; rmax=radius[bb].max()
            for k in range(0,len(aa),512):
                rows=aa[k:k+512]
                neighborhoods=trees[j].query_ball_point(c[rows],(radius[rows]+rmax)*(1+5e-14)+1e-15,p=np.inf)
                sizes=np.fromiter(map(len,neighborhoods),dtype=int,count=len(rows))
                if sizes.sum()==0:continue
                a=np.repeat(rows,sizes);b=bb[np.concatenate(neighborhoods).astype(int)]
                valid=(a<b) if j==i else np.ones(len(a),dtype=bool)
                for ax in range(3):valid&=(lo[a,ax]<=hi[b,ax]) & (hi[a,ax]>=lo[b,ax])
                ab=np.ascontiguousarray(np.column_stack((a[valid],b[valid])),dtype='u4')
                if len(ab)==0:continue
                hit=np.zeros(len(ab),dtype='u1')
                job(x,tri,ab,len(ab),hit);narrow+=len(ab)
                for v,w in ab[hit.astype(bool)]:
                    if (layers[v],v)>(layers[w],w):v,w=w,v
                    l,h=int(layers[v]),int(layers[w])
                    counts[l,h]+=1
                    bucket=samples.setdefault(f'{l}-{h}',[])
                    bucket.append([int(v-starts[l]),int(w-starts[h])])
        print('bin',i,'of',len(classes),'exact candidates',narrow,'hits',counts.tolist(),round(time.monotonic()-st,2),flush=True)
    result=dict(layers=['outer skin','inner skin','deep fat','cortical material surface'][:len(groups)],triangle_counts=[len(f) for f in groups],
                pair_counts=counts.tolist(),aabb_pairs=narrow,witnesses=samples)
    out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    with np.load(sys.argv[2]) as z:
        run(np.load(sys.argv[1]),[z[k] for k in ['l1','l4294966545','l4294966546']],Path(sys.argv[3]),Path(sys.argv[4]))
