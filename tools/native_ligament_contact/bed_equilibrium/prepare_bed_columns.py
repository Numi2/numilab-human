"""Trace vertical columns from the bed to the actual registered O/I/D boundaries.
Faces are sampled in projected area, with coincident edge hits deduplicated.
"""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from material_context import get_layers,sha

def hits(points,v,f):
    tri=v[f].astype(float)
    lo=tri[:,:,:2].min(1);hi=tri[:,:,:2].max(1);center=(lo+hi)/2
    half=((hi-lo)/2).max(1)
    e=tri[:,1]-tri[:,0];d=tri[:,2]-tri[:,0]
    det=e[:,0]*d[:,1]-e[:,1]*d[:,0]
    group=np.frexp(half)[1]
    found=[]
    for g in np.unique(group):
        ids=np.flatnonzero((group==g)&(abs(det)>1e-17))
        if not len(ids):continue
        tr=cKDTree(center[ids]);rad=half[ids].max()+1e-12
        for start in range(0,len(points),2048):
            counts=tr.query_ball_point(points[start:start+2048,:2],rad,p=np.inf)
            sizes=np.fromiter(map(len,counts),int)
            if not sizes.sum():continue
            rows=np.repeat(np.arange(len(sizes))+start,sizes)
            idx=ids[np.concatenate(counts).astype(int)]
            p=points[rows,:2]-tri[idx,0,:2]
            u=(p[:,0]*d[idx,1]-p[:,1]*d[idx,0])/det[idx]
            w=(e[idx,0]*p[:,1]-e[idx,1]*p[:,0])/det[idx]
            keep=(u>=-1e-10)&(w>=-1e-10)&(u+w<=1+1e-10)
            i=idx[keep]
            z=tri[i,0,2]+u[keep]*e[i,2]+w[keep]*d[i,2]
            found.append(np.column_stack((rows[keep],z,i)))
    return np.vstack(found) if found else np.empty((0,3))

def tables(raw,n):
    hit=[[] for _ in range(n)]
    for r,z,i in raw:hit[int(r)].append((z,int(i)))
    result=[]
    for row in hit:
        distinct=[]
        for z,i in sorted(row):
            if not distinct or z-distinct[-1][0]>1e-9:distinct.append((z,i))
        result.append(distinct)
    return result

def face_quadrature(n):
    if not 1<=n<=16:raise ValueError('Expected quadrature subdivision in [1,16]')
    sample=[]
    for i in range(n):
        for j in range(n-i):
            sample.append([(i+1/3)/n,(j+1/3)/n])
            if i+j+1<n:sample.append([(i+2/3)/n,(j+2/3)/n])
    a=np.array(sample);assert len(a)==n*n
    return np.c_[1-a.sum(1),a]

def prepare(w,limit,out,subdivision=1):
    receipt=w/'qualified/motion-008/step-8.receipt.json';pack=w/'qualified/native-coupled-006/native/resting-human.mrvpack'
    layers=get_layers(pack,receipt);v,f=layers['outer'];tri=v[f].astype(float);cent=tri.mean(1)
    norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=abs(norm[:,2])/2
    # Include both windings. The first physical outer surface along the ray
    # determines which side of a fold can actually face the bed.
    bary=face_quadrature(subdivision)
    faces=np.flatnonzero((tri[:,:,2].min(1)<=limit)&(area>1e-14))
    pts=np.einsum('ij,ajk->aik',bary,tri[faces]).reshape(-1,3)
    selected=np.repeat(faces,len(bary));keep=pts[:,2]<=limit
    selected=selected[keep];pts=pts[keep];n=len(selected)
    area=area/len(bary)
    layers={k:tables(hits(pts,*bank),n) for k,bank in layers.items()}
    accepted=[];counts={};areas={};missing=[]
    for j,face in enumerate(selected):
        o,i,d=[layers[k][j] for k in ('outer','inner','deep')]
        if not o or abs(o[0][0]-pts[j,2])>1e-8:why='occluded'
        else:
            following_outer=next((z for z,_ in o if z>pts[j,2]+1e-8),np.inf)
            I=next(((z,t) for z,t in i if z>pts[j,2]+1e-9),(np.inf,-1))
            D=next(((z,t) for z,t in d if z>pts[j,2]+1e-9),(np.inf,-1))
            I2=next((z for z,_ in i if z>I[0]+1e-9),np.inf)
            if not np.isfinite(I[0]) or not np.isfinite(D[0]):why='missing_material_exit'
            elif not pts[j,2]<I[0]<D[0]<min(I2,following_outer):why='interrupted_material_column'
            else:
                accepted.append([face,I[1],D[1],*pts[j],area[face],I[0]-pts[j,2],D[0]-I[0]])
                why='complete'
        if why not in ('complete','occluded'):missing.append([face,*pts[j],area[face]])
        counts[why]=counts.get(why,0)+1;areas[why]=areas.get(why,0.)+area[face]
    arr=np.array(accepted)
    np.savez_compressed(out,missing=np.array(missing).reshape(-1,5),ids=arr[:,:3].astype('u4'),xyz=arr[:,3:6],area_m2=arr[:,6],layer_heights_m=arr[:,7:])
    stats=dict(pack_sha256=sha(pack),capture_sha256=sha(receipt),nodes=len(v),faces=len(f),subdivision=subdivision,maximum_source_height_m=limit,counts=counts,projected_area_m2=areas,minimum_sample_gap_m=float(arr[:,5].min()),candidate_sha256=sha(out))
    out.with_suffix('.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(stats,flush=True)
if __name__=='__main__':prepare(Path(sys.argv[1]),float(sys.argv[2]),Path(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 1)
