"""Check the emitted source meshes, unchanged records, facet quality and rigid volume moments."""
from pathlib import Path
import json,sys,struct
import numpy as np
from fast_skin_intersection_audit import owner,install_for_float32
from shoulder_inputs import old,digest

def moments(v,f,origin):
    t=(v-origin)[f];q=np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2]))/6
    s=t.sum(1);vol=q.sum();center=(q[:,None]*s).sum(0)/(4*vol)
    second=(np.einsum('i,ij,ik->jk',q,s,s)+np.einsum('i,ilj,ilk->jk',q,t,t))/(20*vol)-np.outer(center,center)
    return vol,center+origin,np.eye(3)*np.trace(second)-second

def run(launch,bone,out):
    cfg=json.loads(launch.read_text());parent=Path(cfg['argv'][3]);old.audit.ag.RIGID=Path(cfg['argv'][1]);install_for_float32()
    def read(p):
        old.audit.ag.BONES=p;_,raw,_,_,_,_,_,cat,_,vo,io=old.audit.ag.read_payloads()
        return old.load_local_meshes(raw,cat,vo,io),raw,cat,vo,io
    (ms,raw,c,vo,io),(ns,data,c1,vo1,io1)=read(parent),read(bone)
    assert (vo,io,c)==(vo1,io1,c1) and raw[:vo]==data[:vo] and raw[io:]==data[io:]
    allowed=np.zeros((io-vo)//24,bool);rows=[]
    for sid in (8,9):
        start=c[sid]['firstVertex'];n=c[sid]['vertexCount'];allowed[start:start+n]=True
        x,f,n0=ms[sid][:3];y,g,n1=ns[sid][:3];assert np.array_equal(f,g)
        weld,rep,ix=np.unique(x,axis=0,return_index=True,return_inverse=True)
        assert np.array_equal(y,y[rep][ix])
        faces=ix[f];e=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);undir,cnt=np.unique(np.sort(e,axis=1),axis=0,return_counts=True)
        assert np.all(cnt==2) and len(np.unique(e,axis=0))==len(e)
        pairs=[]
        for v in (x,y):
            records,_=owner.float32_surface_records(v[rep].astype(np.float32),faces)
            h=owner._audit_pair_prepared_first(owner._prepare_surface_aabb(records),records,same_surface=True)
            pairs.append(h['triangle_pairs'])
        assert not any(pairs)
        assert np.max(abs(np.linalg.norm(n1,axis=1)-1))<1.e-6
        tri=x[f];new=y[f];a=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);b=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
        na=np.linalg.norm(a,axis=1);nb=np.linalg.norm(b,axis=1);assert min(na.min(),nb.min())>0
        angle=np.einsum('ij,ij->i',a,b)/na/nb;assert angle.min()>0
        original,cent,J=moments(x,f,x.mean(0));now,cnew,K=moments(y,f,x.mean(0));rel=now/original-1
        assert abs(rel)<3.e-7
        rows.append(dict(stable=sid,unique_vertices=len(rep),nonmanifold_edges=int((cnt!=2).sum()),exact_seam_self_pairs=[len(t) for t in pairs],
            face_area_ratio=np.quantile(nb/na,[0,.01,.5,.99,1]).tolist(),face_normal_cosine_min=float(angle.min()),
            original_volume_m3=abs(original),new_volume_m3=abs(now),relative_signed_volume_change=rel,
            geometric_centroid_displacement_m=np.linalg.norm(cnew-cent),
            geometric_inertia_per_constant_mass_relative_frobenius=np.linalg.norm(K-J)/np.linalg.norm(J),
            changed_vertices=int(np.any(y!=x,axis=1).sum()),maximum_vertex_change_m=np.linalg.norm(y-x,axis=1).max()))
    a=np.frombuffer(raw[vo:io],np.uint8).reshape(-1,24);b=np.frombuffer(data[vo:io],np.uint8).reshape(-1,24)
    assert np.array_equal(a[~allowed],b[~allowed])
    out.write_text(json.dumps(dict(parent=digest(parent),candidate=digest(bone),bone_count=len(ms),rows=rows),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
