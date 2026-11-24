"""Compute the nearest muscle surface in 3D from the deep-fat ends of the ten sampled bed columns."""
import json,mmap,runpy,sys
from pathlib import Path
import numpy as np
from inspect_shoulder_source import nearest
from build_bed_audit import sha
def run(case,rays,audit,lib,out):
    r=json.loads(rays.read_bytes());src=case/'physical-accepted.node';record=case/'native-traces/accepted-geometry/step-1.receipt.json';cap=json.loads(record.read_bytes())
    if sha(src)!=r['origin_frame'] or sha(record)!=r['pose']:raise ValueError('Unmatched material frame')
    x=np.loadtxt(src,skiprows=1,usecols=(1,2,3));rows=r['loaded_outer_nodes'];q=x[[k['node']for k in rows]]+np.array([k['deep_fat_m']for k in rows])[:,None]*r['bed_normal'];geometry=runpy.run_path(str(audit))
    distances=np.full(len(q),np.inf);hits=[None]*len(q);poses={p['body_index']:p for p in cap['accepted_body_poses']};pack=Path(cap['accepted_pack_path'])
    if sha(pack)!=cap['pack_file_sha256']:raise ValueError('Changed accepted pack')
    with pack.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)as buf:
        for key,_,p,tri in geometry['primitives'](buf,poses):
            if key[0]!=51005:continue
            lower=np.linalg.norm(np.maximum(np.maximum(p.min(0)-q,q-p.max(0)),0),axis=1);eligible=np.flatnonzero(lower<distances)
            if not len(eligible):continue
            pos,d,ids=nearest((np.ascontiguousarray(p,'f8'),np.ascontiguousarray(tri,'i4')),q[eligible],lib)
            if not np.isfinite(d).all():raise ValueError('Nonfinite closest point')
            for i in np.flatnonzero(d<distances[eligible]):
                j=eligible[i];distances[j]=d[i];hits[j]=dict(node=rows[j]['node'],deep_fat_point_m=q[j].tolist(),muscle_id=key[1],face=int(ids[i]),muscle_point_m=pos[i].tolist(),unsigned_distance_m=float(d[i]))
    if any(k is None for k in hits):raise ValueError('Missing closest surface')
    owners={h['muscle_id'] for h in hits}
    with pack.open('rb')as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)as buf:
        for key,_,p,tri in geometry['primitives'](buf,poses):
            if key[0]!=51005 or key[1] not in owners:continue
            _,weld=np.unique(p,axis=0,return_inverse=True);ft=weld[tri]
            edges=np.concatenate([ft[:,[0,1]],ft[:,[1,2]],ft[:,[2,0]]])
            unique,first,count=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True)
            directed=np.where(edges[:,0]<edges[:,1],1,-1)
            _,edge_index=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True)
            balanced=np.bincount(edge_index,weights=directed)
            closed=bool(np.all(count==2) and np.all(balanced==0))
            xyz=p[tri].astype('f8')
            for j,h in enumerate(hits):
                if h['muscle_id']!=key[1]:continue
                a,b,c=(xyz-q[j]).transpose(1,0,2)
                u,v,w=(np.linalg.norm(k,axis=1)for k in (a,b,c))
                den=u*v*w+np.einsum('ij,ij->i',a,b)*w+np.einsum('ij,ij->i',b,c)*u+np.einsum('ij,ij->i',c,a)*v
                wind=np.sum(np.arctan2(np.einsum('ij,ij->i',a,np.cross(b,c)),den))/(2*np.pi)
                h.update(closed_oriented_mesh=closed,nonpaired_edges=int(sum(count!=2)),orientation_imbalances=int(sum(balanced!=0)),winding=float(wind),inside=(bool(abs(wind)>.5) if closed and abs(abs(wind)-round(abs(wind)))<1e-5 else None))
    out.write_text(json.dumps(dict(pack=sha(pack),geometry_library=sha(lib),columns=hits),indent=2)+'\n');print([(h['muscle_id'],round(h['unsigned_distance_m']*1000,3))for h in hits])
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
