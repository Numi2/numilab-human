"""Prepare closed material boundary geometry and one reverse query per true cortical vertex."""
from pathlib import Path
import json,sys
import numpy as np
from material_context import sha

def prepare(part,surface,assets,out):
    out.mkdir(exist_ok=True);manifest=json.loads((assets/'manifest.json').read_text())
    changed={'pool-material-points.bin','pool-material-regions.bin'}
    for f in assets.iterdir():
        if f.is_file() and f.name!='manifest.json' and f.name not in changed and not (out/f.name).exists():(out/f.name).hardlink_to(f)
    base=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4);rr=np.fromfile(assets/'pool-material-regions.bin','u4').reshape(-1,4)
    fixed=np.fromfile(assets/'cortical-samples.bin','f4').reshape(2,256,5,4);tri=np.fromfile(assets/'cortical-triangles.bin','f4').reshape(-1,10,4)[:,:3,:3]
    vertices=[];faces=[];faceWeights=[];vertexRanges=[];incident=[];roles=[]
    pool=[base];regions=[rr];nodeBase=0;vertexBase=0;faceBase=0;nextRec=len(base);iBase=0;desc=[]
    for side,sid in enumerate((29,30)):
        p=np.load(part/'projected'/f'{sid}.npz');v=np.load(surface/f'{sid}.npz');xyz=v['vertices'];f=v['face'];owner=v['owner'];wt=v['shape']
        centroid=xyz[f].mean(1);u=np.minimum(1023,np.maximum(0,((centroid-xyz.min(0))/(xyz.max(0)-xyz.min(0))*1023).astype('u4')))
        morton=np.zeros(len(f),'u4')
        for k in range(10):
            for axis in range(3):morton|=((u[:,axis]>>k)&1)<<(3*k+axis)
        ix=np.argsort(morton,kind='stable');f=f[ix];owner=owner[ix];wt=wt[ix];faceWeights.append(wt.astype('f4'))
        raw=np.zeros((len(f),3,4),'u4');raw[:,0,:3]=f+vertexBase;raw[:,0,3]=side;raw[:,2]=p['cells'][owner]+nodeBase
        edges={};by=[[] for z in xyz]
        for k,t in enumerate(f):
            for j,a in enumerate(t):
                by[a].append(k*3+j+3*faceBase);b=t[(j+1)%3];key=tuple(sorted([a,b]))
                if key in edges:
                    old,axis=edges.pop(key);raw[old,1,axis]=k+faceBase;raw[k,1,j]=old+faceBase
                else:edges[key]=(k,j)
        if edges:raise ValueError('Material shell does not have paired edges')
        faces.append(raw);vdata=np.zeros((len(xyz),2,4),'f4');vdata[:,0]=v['vertex_shape'];vdata[:,1].view('u4')[:]=v['node']+nodeBase;vertices.append(vdata)
        vr=[]
        for r in by:vr.append([iBase,len(r),side,0]);incident.extend(r);iBase+=len(r)
        vertexRanges.append(np.array(vr,'u4'))
        banks={}
        for row in fixed[side,:manifest['sides'][side]['rows']]:
            start,n=map(int,row[4].copy().view('u4')[:2]);banks[start]=(n,row[3].copy().view('u4'))
        for start,(n,domain) in sorted(banks.items()):
            points=np.unique(tri[start:start+n].reshape(-1,3),axis=0)
            # Spatial groups share only a work proposal. Every vertex has its own exact closest-feature query.
            labels=np.floor(points/.02).astype(int);_,group=np.unique(labels,axis=0,return_inverse=True)
            order=np.argsort(group,kind='stable');points=points[order];g=group[order]
            ptr=np.r_[0,1+np.flatnonzero(np.diff(g)),len(g)]
            regions.append(np.array([[nextRec+a,b-a,side*256,0] for a,b in zip(ptr[:-1],ptr[1:])],'u4'))
            rec=np.zeros((len(points),7,4),'f4');rec[:,2,:3]=points;rec[:,3].view('u4')[:]=domain;rec[:,4,0:2].view('u4')[:]=[start,n]
            rec[:,5].view('u4')[:]=p['cells'][owner[0]]+nodeBase
            pool.append(rec);nextRec+=len(points)
        desc.append(dict(vertices=len(xyz),first_vertex=vertexBase,faces=len(f),first_face=faceBase))
        vertexBase+=len(xyz);faceBase+=len(f);nodeBase+=len(p['vertices'])
    data={'pool-material-points':np.concatenate(pool),'pool-material-regions':np.concatenate(regions),'closed-material-vertices':np.concatenate(vertices),'closed-material-triangles':np.concatenate(faces),'closed-material-shapes':np.concatenate(faceWeights),'closed-material-ranges':np.concatenate(vertexRanges),'closed-material-incidence':np.asarray(incident,'u4')}
    for k,a in data.items():
        f=out/(k+'.bin');a.tofile(f);manifest['files'][f.name]=sha(f)
    leaf=1<<(int(np.ceil(max(x['faces'] for x in desc)/8))-1).bit_length()
    manifest['contact_pool']['count']=nextRec;manifest['closed_reverse']=dict(first=len(base),unique_vertices=nextRec-len(base),groups=len(data['pool-material-regions']),sides=desc,leaves=leaf)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['closed_reverse'])
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
