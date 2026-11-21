"""Register reachable cortical vertices against P1 material boundary faces for reverse contact tests."""
import json,sys,shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from fit_contact_domain import points
from qualify_material_motion import frame,local
from material_context import sha

def build(part,prev,assets,out):
    out.mkdir(exist_ok=True);manifest=json.loads((assets/'manifest.json').read_text())
    for f in assets.iterdir():
        if f.is_file() and f.name not in ('manifest.json','cortical-flexibility.bin','cortical-free-index.bin'):
            if f.name not in ('pool-material-points.bin','pool-material-regions.bin','pool-cortical-neighborhood.bin') and not (out/f.name).exists():(out/f.name).hardlink_to(f)
    original=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4);regions=np.fromfile(assets/'pool-material-regions.bin','u4').reshape(-1,4)
    oldnear=np.fromfile(assets/'pool-cortical-neighborhood.bin','u4')
    packed=np.fromfile(assets/'cortical-triangles.bin','f4').reshape(-1,10,4)[:,:3,:3].astype(float)
    fixed=np.fromfile(assets/'cortical-samples.bin','f4').reshape(2,256,5,4)
    src=np.fromfile(assets/'material-nodes.bin','f4').reshape(-1,3,4)
    neutral={r['body_index']:frame(r) for r in json.loads((prev/'native-release-motion/step-8.receipt.json').read_text())['initial_anatomical_registration']['body_poses']}
    start=len(original);nearoff=len(oldnear);nodebase=0;target=[];extended=[];groups=[];additional=[];memo={};lookup=[]
    for side,sid in enumerate((29,30)):
        _,_,xyz,owner,shape=points(part,prev,sid);p=np.load(part/'projected'/f'{sid}.npz');t=p['cells'];segments={}
        for row in fixed[side,:manifest['sides'][side]['rows']]:
            dom=row[3].copy().view('u4');first,n=map(int,row[4].copy().view('u4')[:2]);segments[first]=(n,int(dom[2]),dom)
        side_rows=[]
        for first,(n,b,dom) in sorted(segments.items()):
            f=packed[first:first+n];vertices=np.unique(f.reshape(-1,3),axis=0);tree=cKDTree(vertices)
            triangle=local(xyz.reshape(-1,3),neutral[b]).reshape(-1,3,3);centre=triangle.mean(1);rad=np.linalg.norm(triangle-centre[:,None],axis=2).max(1)
            nearby=tree.query_ball_point(centre,rad+.003)
            lo=f.min(1);hi=f.max(1);ctree=cKDTree((lo+hi)/2);maxbound=np.linalg.norm((hi-lo)/2,axis=1).max()
            for k,near in enumerate(nearby):
                if not near:continue
                vs=np.asarray(near);z=vertices[vs];a,e,h=triangle[k,0],triangle[k,1]-triangle[k,0],triangle[k,2]-triangle[k,0];v=z-a;normal=np.cross(e,h);area=normal@normal
                if area<=0:continue
                u=np.cross(v,h)@normal/area;w=np.cross(e,v)@normal/area;dist=(v@normal)**2/area
                bad=(u<0)|(w<0)|(u+w>1);outside=np.full(len(z),np.inf)
                for j in range(3):
                    edge=triangle[k,(j+1)%3]-triangle[k,j];d=z-triangle[k,j];alpha=np.clip(d@edge/(edge@edge),0,1)
                    outside=np.minimum(outside,np.sum((d-alpha[:,None]*edge)**2,axis=1))
                inside=np.where(bad,outside,dist)<=.003**2
                for idx in np.flatnonzero(inside):
                    key=(first,int(vs[idx]))
                    if key not in memo:
                        cand=np.asarray(ctree.query_ball_point(z[idx],.004+maxbound),int)
                        delta=np.maximum(0,np.maximum(lo[cand]-z[idx],z[idx]-hi[cand]));faces=cand[np.sum(delta*delta,axis=1)<=.004**2]
                        assert len(faces);memo[key]=(nearoff,len(faces));additional.append((faces+first).astype('u4'));nearoff+=len(faces)
                    loc,length=memo[key];cell=t[owner[k]]+nodebase
                    entry=np.zeros((7,4),'f4');entry[2]=[*z[idx],.004];entry[3].view('u4')[:]=dom
                    entry[4].view('u4')[:]=[first,n,loc,length];entry[5].view('u4')[:]=cell
                    wt=shape[k].mean(0);entry[6]=wt
                    for d in (0,1):
                        scale=wt*src[cell,d,3];vsum=scale.sum();entry[d,3]=vsum
                        if abs(vsum)>1e-15:entry[d,:3]=scale@src[cell,d,:3]/vsum
                    spatial=tuple(np.floor(centre[k]/.02).astype(int));side_rows.append(((first,*spatial),entry,shape[k],(sid,k,b,int(vs[idx]))))
        side_rows.sort(key=lambda a:a[0]);current=None
        for key,entry,w,identity in side_rows:
            if current!=key:groups.append([start,0,side*256,0]);current=key
            groups[-1][1]+=1;start+=1;extended.append(entry);target.append(w);lookup.append(identity)
        nodebase+=len(p['vertices'])
    files={'pool-material-points':np.concatenate((original,np.stack(extended))),'pool-material-regions':np.vstack((regions,np.array(groups,'u4'))),'pool-cortical-neighborhood':np.concatenate([oldnear,*additional]),'cortical-face-weights':np.array(target,'f4'),'cortical-face-lineage':np.array(lookup,'u4')}
    for k,v in files.items():
        f=out/(k+'.bin');v.tofile(f);manifest['files'][f.name]=sha(f)
    for k in ('cortical-flexibility.bin','cortical-free-index.bin'):manifest['files'].pop(k,None)
    manifest.pop('contact_flexibility',None);manifest['contact_pool']['count']=start
    manifest['cortical_face_contacts']=dict(first=len(original),pairs=len(extended),new_search_groups=len(groups),source_vertex_to_face_reach_m=.003,point_registration_sha256=sha(assets/'manifest.json'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['cortical_face_contacts'],flush=True)
if __name__=='__main__':build(*map(Path,sys.argv[1:]))
