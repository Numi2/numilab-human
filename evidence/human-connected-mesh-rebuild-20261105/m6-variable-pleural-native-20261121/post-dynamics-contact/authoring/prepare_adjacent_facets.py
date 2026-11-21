"""Register finite neighboring and exterior facets from the connected cortical interface."""
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from qualify_material_motion import frame
from material_context import sha

def build(part,base,receipt,out,reach):
    out.mkdir(exist_ok=True)
    d=json.loads((base/'manifest.json').read_text());changed={'pool-material-points.bin','pool-material-regions.bin'}
    for p in base.iterdir():
        if p.name!='manifest.json' and p.name not in changed and not(out/p.name).exists():(out/p.name).hardlink_to(p)
    frames={r['body_index']:frame(r) for r in json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses']}
    rot={i:Rotation.from_quat(v[3:]) for i,v in frames.items()};P=rot[128];pp=frames[128][:3]
    pool=np.fromfile(base/'pool-material-points.bin','f4').reshape(-1,7,4);regions=np.fromfile(base/'pool-material-regions.bin','u4').reshape(-1,4).tolist()
    tris=np.fromfile(base/'cortical-triangles.bin','f4').reshape(-1,10,4);fixed=np.fromfile(base/'cortical-samples.bin','f4').reshape(2,256,5,4)
    data=[];added=[];lineage=[];nodebase=0;summary=[]
    for side,sid in enumerate((29,30)):
        s=np.load(part/'conforming-boundary'/f'{sid}.npz');ports=np.load(part/'ports'/f'{sid}.npz');mesh=np.load(part/'projected'/f'{sid}.npz')
        F=s['face'];V=s['vertices'];nface=len(F);source=s['source_face'];names=ports['target_bank'];nport=len(names)
        banks={str(names[i]):tuple(map(int,fixed[side,j,4].copy().view('u4')[:2])) for j,i in enumerate(d['sides'][side]['quadrature_indices'])}
        source_bank={};source_neighbors=defaultdict(set);triangles={};derivs={};body={};source_origin={}
        for name,(begin,n) in banks.items():
            b=int(ports['target_body'][np.flatnonzero(names==name)[0]]);body[name]=b;t=tris[begin:begin+n,:3,:3].astype(float);E=t[:,1:]-t[:,:1];dual=np.linalg.solve(E@E.transpose(0,2,1),E)
            x,inv=np.unique(t.reshape(-1,3),axis=0,return_inverse=True);cf=inv.reshape(-1,3);by=defaultdict(set)
            for k,row in enumerate(cf):
                for i in row:by[int(i)].add(k)
            source_neighbors[name]=[set.union(*(by[int(i)] for i in row)) for row in cf]
            triangles[name]=t;derivs[name]=dual;source_origin[name]=rot[b].inv().apply(V-frames[b][:3])
        incident=defaultdict(set);own={}
        for j,i in enumerate(source):
            if i>=nport or names[i] not in banks:continue
            key=(str(names[i]),int(ports['target_face'][i]));own[j]=key
            for k in F[j]:incident[int(k)].add(key)
        items=[]
        for j,row in enumerate(F):
            near=set.union(*(incident[int(k)] for k in row))
            for name,pface in list(near):
                for k in source_neighbors[name][pface]:near.add((name,k))
            for name,k in near:
                t=triangles[name][k];z=source_origin[name][row];delta=np.maximum(z.min(0)-t.max(0),t.min(0)-z.max(0))
                if max(delta)>reach:continue
                reference=own.get(j)==(name,k)
                nm=np.cross(z[1]-z[0],z[2]-z[0]);nm/=np.linalg.norm(nm)
                bg=banks[name][0]+k;nb=tris[bg,3,:3].astype(float);opposing=-float(nm@nb)
                # Preserve lateral/topological witnesses in the atlas; the GPU chooses the currently opposing side.
                sourceType=0 if reference else (1 if j in own else 2)
                items.append((name,k,j,sourceType))
        items.sort();prior=None;stats={};maxoffset=0;first=len(data)
        for name,k,j,kind in items:
            b=body[name];bg=banks[name][0]+k;t=triangles[name][k];z=source_origin[name][F[j]];uv=(z-t[:1])@derivs[name][k].T
            proj=t[:1]+uv@(t[1:]-t[:1]);offset=P.inv().apply(rot[b].apply(z-proj))
            pair=np.zeros((17,4),'f4');pair[0].view('u4')[:]=[bg,b,side,j];pair[1:4]=s['shape'][j];pair[4:7,:2]=uv;pair[4,3]=kind
            pair[7:10,:3]=proj;pair[10:13,:3]=P.inv().apply(rot[b].apply(proj)+frames[b][:3]-pp);pair[13:16,:3]=offset
            pair[16].view('u4')[:]=mesh['cells'][s['owner'][j]]+nodebase
            if kind==0:maxoffset=max(maxoffset,np.linalg.norm(offset,axis=1).max())
            rec=np.zeros((2,7,4),'f4');rec[:,3].view('u4')[:]=[128,131 if side==0 else 145,b,1];rec[:,4,0].view('u4')[:]=bg
            rec[:,5].view('u4')[:]=pair[16].view('u4');rec[:,6]=s['shape'][j].mean(0)
            key=(name,k)
            if key!=prior:regions.append([len(pool)+2*len(data),0,side*256,0]);prior=key
            regions[-1][1]+=2;data.append(pair);added.append(rec);lineage.append([side,j,k,b,kind]);stats[kind]=stats.get(kind,0)+1
        summary.append(dict(muscle=sid,first=first,pairs=len(data)-first,kinds=stats,registered_face_error_m=maxoffset,source=sha(part/'conforming-boundary'/f'{sid}.npz')));nodebase+=len(mesh['vertices'])
    pose=[]
    for b in (128,131,145,1):pose.extend([[*frames[b][:3],0],frames[b][3:]])
    arrays={'pelvic-finite-pairs':np.stack(data),'pelvic-finite-poses':np.array(pose,'f4'),'pool-material-points':np.concatenate((pool,np.concatenate(added))),'pool-material-regions':np.array(regions,'u4')}
    for name,v in arrays.items():
        f=out/(name+'.bin');v.tofile(f);d['files'][f.name]=sha(f)
    np.savez_compressed(out/'facet-ancestry.npz',pairs=np.array(lineage,'u4'));d['contact_pool']['count']=len(pool)+2*len(data)
    d['finite_facets']=dict(start=len(pool),number=len(data),regions=summary,source_manifest=sha(base/'manifest.json'),reference_aabb_reach_m=reach,ancestry=sha(out/'facet-ancestry.npz'))
    (out/'manifest.json').write_text(json.dumps(d,indent=2)+'\n');print(d['finite_facets'],flush=True)
if __name__=='__main__':build(*map(Path,sys.argv[1:5]),float(sys.argv[5]))
