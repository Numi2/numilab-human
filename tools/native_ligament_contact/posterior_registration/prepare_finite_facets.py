"""Register occupied muscle cut faces to their own finite cortical parent triangles."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from qualify_material_motion import frame
from material_context import sha

def prepare(part,base,receipt,out):
    out.mkdir(exist_ok=True)
    manifest=json.loads((base/'manifest.json').read_text())
    changed={'pool-material-points.bin','pool-material-regions.bin'}
    for p in base.iterdir():
        if p.name!='manifest.json' and p.name not in changed and not (out/p.name).exists():(out/p.name).hardlink_to(p)
    frames={r['body_index']:frame(r) for r in json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses']}
    def rotation(body):return Rotation.from_quat(frames[body][3:])
    rp=rotation(128);pp=frames[128][:3];spec=[]
    pool=np.fromfile(base/'pool-material-points.bin','f4').reshape(-1,7,4)
    region=np.fromfile(base/'pool-material-regions.bin','u4').reshape(-1,4)
    tri=np.fromfile(base/'cortical-triangles.bin','f4').reshape(-1,10,4)
    samples=np.fromfile(base/'cortical-samples.bin','f4').reshape(2,256,5,4)
    pose=[]
    for body in (128,131,145,1):pose.extend([[*frames[body][:3],0],frames[body][3:]])
    extra=[];groups=[];data=[];first=len(pool);offset=0
    for side,(sid,fem) in enumerate(((29,131),(30,145))):
        p=np.load(part/'projected'/f'{sid}.npz');port=np.load(part/'ports'/f'{sid}.npz');ids=manifest['sides'][side]['quadrature_indices']
        bank={}
        for j,i in enumerate(ids):
            key=str(port['target_bank'][i]);start,n=samples[side,j,4].copy().view('u4')[:2];bank[key]=(int(start),int(n))
        use=np.flatnonzero(port['target_body']>=0);keys=sorted(use,key=lambda i:(str(port['target_bank'][i]),int(port['target_face'][i]),int(i)))
        previous=None;maxerr=0.;area=0.;firstpair=len(data);mincos=1.
        for i in keys:
            name=str(port['target_bank'][i]);b=int(port['target_body'][i]);parent=int(port['target_face'][i]);start,n=bank[name];assert 0<=parent<n
            t=tri[start+parent,:3,:3].astype(float);normal=tri[start+parent,3,:3].astype(float)
            E=t[1:]-t[:1];dual=np.linalg.solve(E@E.T,E)
            x=p['port_vertices'][i];z=rotation(b).inv().apply(x-frames[b][:3]);a=(z-t[:1])@dual.T
            proj=t[:1]+a@E;rest=rp.inv().apply(rotation(b).apply(z-proj))
            err=np.max(np.linalg.norm(z-proj,axis=1));maxerr=max(err,maxerr)
            if err>5.e-7 or min(a.min(),(1-a.sum(1)).min())<-.002:raise ValueError(('Finite parent registration',sid,i,err,a))
            na=np.cross(z[1]-z[0],z[2]-z[0]);area+=np.linalg.norm(na)/2;cos=-na@normal/np.linalg.norm(na);mincos=min(mincos,cos)
            if cos<.99:raise ValueError(('Mismatched cortical side',sid,i,cos))
            pair=np.zeros((17,4),'f4');pair[0].view('u4')[:]=[start+parent,b,side,i];pair[1:4]=p['port_shape'][i]
            pair[4:7,:2]=a;pair[7:10,:3]=proj;pair[10:13,:3]=rp.inv().apply(rotation(b).apply(proj)+frames[b][:3]-pp)
            pair[13:16,:3]=rest;pair[16].view('u4')[:]=p['cells'][p['port_owner_cell'][i]]+offset;data.append(pair)
            row=np.zeros((2,7,4),'f4');row[:,3].view('u4')[:]=[128,fem,b,1];row[:,4,0].view('u4')[:]=start+parent
            row[:,5].view('u4')[:]=pair[16].view('u4');row[:,6]=p['port_shape'][i].mean(0);extra.append(row)
            current=(name,parent)
            if previous!=current:groups.append([first+2*(len(data)-1),0,side*256,0]);previous=current
            groups[-1][1]+=2
        offset+=len(p['vertices'])
        spec.append(dict(muscle=sid,first=firstpair,pairs=len(data)-firstpair,registered_area_m2=area,maximum_reference_plane_offset_m=maxerr,minimum_opposing_normal_cosine=mincos,ports_sha256=sha(part/'ports'/f'{sid}.npz')))
    tables={'pelvic-finite-pairs':np.stack(data),'pelvic-finite-poses':np.asarray(pose,'f4'),'pool-material-points':np.concatenate((pool,np.concatenate(extra))),'pool-material-regions':np.vstack((region,np.asarray(groups,'u4')))}
    for name,array in tables.items():
        f=out/(name+'.bin');array.tofile(f);manifest['files'][f.name]=sha(f)
    manifest['contact_pool']['count']=first+2*len(data);manifest['finite_facets']=dict(start=first,number=len(data),regions=spec,original_regions=len(region),source_manifest=sha(base/'manifest.json'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['finite_facets'])
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
