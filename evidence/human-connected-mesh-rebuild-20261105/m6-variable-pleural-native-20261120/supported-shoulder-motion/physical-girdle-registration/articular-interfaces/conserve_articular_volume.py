"""Conserve discrete rigid-member volume within the measured free surface around an articular repair."""
import json,os,runpy,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from shoulder_inputs import digest,old
from fit_contact_footprint import volume

def area_vectors(x,f):
    tri=x[f];area=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    raw=np.zeros_like(x)
    for col in f.T:np.add.at(raw,col,area)
    return raw

def support_radius(banks,sid,x,f,limit):
    tris=x[f];centers=tris.mean(1)
    radius=np.linalg.norm(tris-centers[:,None],axis=2).max(1)
    lo=x.min(0)-limit;hi=x.max(0)+limit;near=[]
    for k,(v,g) in banks.items():
        if k==(51004,sid) or np.any(lo>v.max(0)) or np.any(hi<v.min(0)):continue
        t=v[g];use=np.all((t.min(1)<=hi)&(t.max(1)>=lo),axis=1)
        if use.any():near.extend(t[use])
    t=np.asarray(near,float);cent=t.mean(1);extent=np.linalg.norm(t-cent[:,None],axis=2).max(1)
    # Restrict the separating-axis check to potentially close triangles.
    tree=cKDTree(cent);by_face=np.full(len(f),limit)
    for i,tri in enumerate(tris):
        idx=np.array(tree.query_ball_point(centers[i],limit+radius[i]+extent.max()),int)
        idx=idx[np.linalg.norm(cent[idx]-centers[i],axis=1)<=limit+radius[i]+extent[idx]]
        if not len(idx):continue
        targets=t[idx];source_edges=np.roll(tri,-1,axis=0)-tri
        target_edges=np.roll(targets,-1,axis=1)-targets
        axes=[np.broadcast_to(np.cross(source_edges[0],source_edges[1]),(len(idx),3)),
              np.cross(target_edges[:,0],target_edges[:,1])]
        axes.extend(np.cross(e,target_edges[:,j]) for e in source_edges for j in range(3))
        # A separation in any unit direction is a lower bound on Euclidean
        # face distance. Intersecting and coplanar ambiguous pairs stay frozen.
        lower=np.zeros(len(idx))
        for axis in axes:
            axis=axis/np.maximum(1.e-30,np.linalg.norm(axis,axis=1))[:,None]
            a=axis@tri.T;b=np.einsum('ijk,ik->ij',targets,axis)
            lower=np.maximum(lower,np.maximum(a.min(1)-b.max(1),b.min(1)-a.max(1)))
        by_face[i]=min(limit,float(lower.min()))
    bound=np.full(len(x),limit)
    for col in f.T:np.minimum.at(bound,col,by_face)
    return np.maximum(0,bound-5.e-6)

def run(scene,loader,zero,bone,out):
    cfg=json.loads((scene/'launch.json').read_text());old.audit.ag.RIGID=Path(cfg['argv'][1])
    parent=Path(cfg['argv'][3])
    os.environ['NUMI_HUMAN_CAPTURED_TIME_ZERO_RECEIPT']=str(zero);sys.argv[1]=str(scene)
    st=runpy.run_path(str(loader));banks=st['banks'];banks[51007,1]=(st['skin'],st['skin_faces'])
    def read(p):
        old.audit.ag.BONES=p;_,raw,_,_,_,_,_,cat,_,vo,ix=old.audit.ag.read_payloads()
        return old.load_local_meshes(raw,cat,vo,ix),raw,cat,vo
    ms,_,_,_=read(parent);new,raw,cat,vo=read(bone);payload=bytearray(raw)
    info=json.loads((bone.parent/'repair.json').read_text());assert info['bones_sha256']==digest(bone) and info['parent_sha256']==digest(parent)
    report=[]
    for row in info['repairs']:
        sid=row['stable_id'];g=cat[sid];oldx,f,n=ms[sid][:3];rest=new[sid][0]
        x,inv=np.unique(rest,axis=0,return_inverse=True)
        face=inv[f];pose=st['poses'][g['body']];rr=R.from_quat(pose['quaternion_xyzw'])
        world=rr.apply(x)+pose['position_m']
        cap=support_radius(banks,sid,world,face,.0035)
        changed=np.linalg.norm(rest-oldx,axis=1)>1.e-7
        # Keep the repaired footprint and its immediate collar fixed. Keep
        # the lateral articular quarter and bone interfaces in their source position.
        ax=(x-row['sc_local_m'])@row['axis'];closest=cKDTree(rest[changed]).query(x)[0]
        mix=np.clip((closest-.001)/.010,0,1)*np.clip((.105-ax)/.018,0,1)
        cap=np.minimum(cap,.0025)*mix
        direction=area_vectors(x,face)*np.sign(volume(x,face))
        direction/=np.maximum(1.e-30,np.linalg.norm(direction,axis=1))[:,None]
        target=volume(oldx,f);before=volume(x,face)
        def points(h): return x+np.minimum(h,cap)[:,None]*direction
        eligible=(volume(points(.0025),face)-target)*np.sign(target)>=0
        if not eligible:
            print('insufficient conservative volume',sid,abs(volume(points(.0025),face)-target),flush=True)
            raise ValueError('insufficient measured free surface')
        # Search on the quantized ABI. Source positions are stored as
        # binary32 in the original atlas frame, well away from its origin.
        matrix=old.audit.qrot(g['boneQ'],np.eye(3))*g['scale']
        imatrix=np.linalg.inv(matrix);tf=np.array(g['translation'])
        def encoded(h):
            trial=points(h)[inv];mask=np.any(trial!=rest,axis=1)
            rows=((trial[mask]-tf)@imatrix).astype('<f4')
            trial[mask]=old.audit.qrot(g['boneQ'],rows.astype(float)*g['scale'])+tf
            return volume(trial,f)-target
        low,high=0.,.0025;options=[]
        orientation=np.sign(target)
        for _ in range(48):
            h=(low+high)*.5
            error=encoded(h);options.append((abs(error),h))
            if error*orientation<0:low=h
            else:high=h
        _,h=min(options)
        final=points(h)[inv];alter=np.any(final!=rest,axis=1)
        # Propagate each native shading normal with the minimal surface
        # rotation; keep preexisting split normals and untouched records.
        u=area_vectors(rest,f);v=area_vectors(final,f)
        for vec in (u,v):vec/=np.maximum(1.e-30,np.linalg.norm(vec,axis=1))[:,None]
        axis=np.cross(u,v);co=np.einsum('ij,ij->i',u,v)
        on=new[sid][2];nn=on+np.cross(axis,on)+np.cross(axis,np.cross(axis,on))/np.maximum(1.e-15,1+co)[:,None]
        nn/=np.maximum(np.linalg.norm(nn,axis=1)[:,None],1.e-30)
        qr=R.from_quat(g['boneQ']);tf=np.array(g['translation'])
        vals=np.frombuffer(payload,dtype='<f4',count=6*g['vertexCount'],offset=vo+24*g['firstVertex']).reshape(-1,6)
        vals[alter,:3]=(final[alter]-tf)@imatrix;vals[alter,3:]=qr.inv().apply(nn[alter])
        actual=old.audit.qrot(g['boneQ'],vals[:,:3].astype(float)*g['scale'])+tf
        signed=volume(actual,f)/target-1
        assert abs(signed)<3.e-7
        report.append(dict(stable_id=sid,original_volume_m3=abs(target),pre_balance_volume_m3=abs(before),final_volume_m3=abs(volume(actual,f)),
            relative_volume_error=signed,max_balancing_shift_m=h,balanced_vertices=int(alter.sum()),candidate_nodes_with_positive_clearance=int((cap>0).sum())))
    out.mkdir(exist_ok=True);dest=out/bone.name;dest.write_bytes(payload)
    doc=json.loads(bone.with_suffix('.manifest.json').read_text());doc['payload']['sha256']=digest(dest)
    doc['source']['bounded_articular_volume_balance']=dict(source_bone_sha256=digest(bone),pose_receipt_sha256=digest(zero),source_program_sha256=digest(Path(__file__)),rows=report)
    (out/bone.with_suffix('.manifest.json').name).write_text(json.dumps(doc,indent=2)+'\n')
    (out/'balance.json').write_text(json.dumps(dict(bones_sha256=digest(dest),rows=report),indent=2)+'\n')
    print(report)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
