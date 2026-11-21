"""Estimate the current forward contact search work from one real registered endpoint."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_finite_cortical_crossings import load_vertices
from material_context import sha

def dot(a,b):return np.einsum("ij,ij->i",a,b)
def squared_distance(q,tri):
    a,b,c=tri[:,0],tri[:,1],tri[:,2]
    e=b-a;f=c-a;k=np.cross(e,f);k2=dot(k,k);pa=q-a
    if (k2<=0).any():raise ValueError("Degenerate registered cortical triangle")
    u=dot(np.cross(pa,f),k)/k2;v=dot(np.cross(e,pa),k)/k2
    best=np.where((u>=0)&(v>=0)&(u+v<=1),dot(pa,k)**2/k2,np.inf)
    for lo,hi in [(a,b),(b,c),(c,a)]:
        edge=hi-lo;r=q-lo;t=np.clip(dot(r,edge)/dot(edge,edge),0,1)
        residual=r-t[:,None]*edge;best=np.minimum(best,dot(residual,residual))
    return best
def measure(assets,run):
    spec=json.loads((assets/"manifest.json").read_text())
    g=spec["closed_reverse"];cap=run/"accepted-pelvic-surfaces.csv"
    step,X=load_vertices(cap,{'material_vertex':sum(q["vertices"] for q in g["sides"]),'cortical_vertex':g["unique_vertices"]})
    surf=X["material_vertex"];cortex=X["cortical_vertex"];f=np.fromfile(assets/"forward-cortical-faces.bin","<u4").reshape(-1,4)
    sites=np.fromfile(assets/"forward-surface-samples.bin","<f4").reshape(-1,2,4)
    rows=np.memmap(assets/"pool-material-points.bin",dtype="<f4",mode="r").reshape(-1,7,4)[:g["first"]]
    index=np.fromfile(assets/"pool-cortical-neighborhood.bin","<u4")
    ids=sites[:,0].copy().view("<u4")[:,:3];weights=sites[:,1];a=surf[ids[:,0]]
    tail=weights[:,1,None]*(surf[ids[:,1]]-a)+weights[:,2,None]*(surf[ids[:,2]]-a)
    pos={};rot={}
    with (run/"accepted-pelvic-state.csv").open() as stream:
        for r in csv.DictReader(stream):
            if r["field"] not in ("relative_position","orientation"):continue
            v=np.array([r[k] for k in "xyzw"],dtype="<f4").astype(float);b=int(r["index"])
            if r["field"]=="relative_position":pos[b]=v[:3]
            else:rot[b]=Rotation.from_quat(v)
    banks=rows[:,4].copy().view("<u4");owners=rows[:,3].copy().view("<u4")[:,2]
    ref=rows[:,2,:3].astype(float);center=ref.copy()
    for b in np.unique(owners):
        if b!=128:
            m=owners==b;center[m]=rot[128].inv().apply(rot[b].apply(ref[m])+pos[b])
    site=np.arange(len(rows));minima=np.full(len(rows),np.inf)
    visits=int(banks[:,3].astype("i8").sum())
    for chunk in np.array_split(site,32):
        counts=banks[chunk,3].astype(int); ix=np.repeat(chunk,counts)
        sub=np.arange(len(ix))-np.repeat(np.cumsum(counts)-counts,counts)
        nb=index[banks[ix,2]+sub];t=cortex[f[nb,:3]]
        np.minimum.at(minima,ix,squared_distance(a[ix]+tail[ix],t))
    norm=lambda x:np.linalg.norm(x,axis=1)
    slack=norm((a-center)+tail)+np.sqrt(minima)+32*np.finfo("f4").eps*(norm(ref)+norm(center)+norm(a)+norm(tail))-rows[:,2,3]
    ntri=banks[:,1].astype("i8")
    def count(mask):return dict(sites=int(mask.sum()),triangle_visits=int(ntri[mask].sum()))
    return dict(step=step,surface_sha256=sha(cap),sites=len(site),neighbor_triangle_visits=visits,
                estimated_full_bank=count(slack>=0),full_bank_at_plus_one_micron=count(slack>=1e-6),
                full_bank_at_minus_one_micron=count(slack>=-1e-6),
                scope="Single Float32 emitted endpoint reconstructed in double. One unmasked two-sided forward query. Counts describe triangle evaluations, without assigning elapsed time.")
if __name__=="__main__":
    a,r,p=map(Path,sys.argv[1:]);p.write_text(json.dumps(measure(a,r),indent=2)+"\n")
