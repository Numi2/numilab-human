"""Register cut-boundary points to cortical broad-phase neighborhoods and FEM shape functions."""
import json,sys,shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from fit_contact_domain import points
from cortical_distance import CorticalSurface
from material_context import sha
from qualify_material_motion import frame,local
from pose_glutes import parts

def build(p47,prev,response,assets,out,library):
    out.mkdir(exist_ok=True)
    for f in assets.iterdir():
        if f.is_file():shutil.copyfile(f,out/f.name)
    d=json.loads((assets/'manifest.json').read_text())
    source=json.loads((prev/'release-posterior/config.json').read_text())['argv']
    tissue=Path(source[source.index('--soft-tissue-payload')+1])
    ref=json.loads((prev/'native-release-motion/step-8.receipt.json').read_text())
    pose={r['body_index']:frame(r) for r in ref['initial_anatomical_registration']['body_poses']}
    packed=np.fromfile(assets/'cortical-triangles.bin','f4').reshape(-1,10,4)[:,:3,:3].astype(float)
    original=np.fromfile(assets/'cortical-samples.bin','f4').reshape(2,256,5,4)
    domain=[];neighbors=[];rules=[];group_map=[];node_offset=0;offset=0;start=0
    for side,sid in enumerate((29,30)):
        p=np.load(p47/'projected'/f'{sid}.npz');port=np.load(p47/'ports'/f'{sid}.npz')
        region=np.load(response/f'{sid}.npz')['pressure_region']
        cell,w,*_=points(p47,prev,sid);x=np.einsum('ni,nij->nj',w,p['vertices'][cell])
        cortical_samples=p['port_vertices'].mean(1)
        bone={};coords,weights,ids,_,aff=parts(tissue,sid)
        for j,f in enumerate(d['sides'][side]['quadrature_indices']):
            row=original[side,j];key=int(port['target_bank'][f][5:]);first,n=row[4].view('u4')[:2];bone[key]=(int(first),int(n),int(port['target_body'][f]))
        chunks={}
        for key,(first,n,owner) in sorted(bone.items()):
            faces=packed[first:first+n];xx=faces.reshape(-1,3);ff=np.arange(len(xx)).reshape(-1,3)
            surface=CorticalSurface(library,xx,ff);v=local(x,pose[owner]);g,*_=surface.query(v)
            eligible=np.flatnonzero(g<=.002)
            patch=np.flatnonzero(port['target_bank']==f'bone_{key}')
            _,near=cKDTree(cortical_samples[patch]).query(x[eligible]);labels=region[patch[near]]
            lo=faces.min(1);hi=faces.max(1);middle=(hi+lo)/2
            bounds=np.linalg.norm((hi-lo)/2,axis=1);tree=cKDTree(middle)
            all_neighbors=tree.query_ball_point(v[eligible],.004+bounds.max())
            for j,idx in enumerate(eligible):
                pt=v[idx];cand=np.asarray(all_neighbors[j],int);disp=np.maximum(0,np.maximum(lo[cand]-pt,pt-hi[cand]))
                nearfaces=cand[np.einsum('ij,ij->i',disp,disp)<=.004**2]
                if len(nearfaces)==0:raise ValueError(('Uncovered source',sid,key,idx))
                group=int(labels[j]);s=np.zeros((7,4),dtype='f4');nodes=cell[idx];bary=w[idx]
                for k in range(2):
                    wk=bary*weights[nodes,k];s[k,3]=wk.sum()
                    if abs(s[k,3])>1e-14:s[k,:3]=wk@(coords[nodes]@aff[k][0].T+aff[k][1])/wk.sum()
                s[2]=[*pt,.004];s[3].view('u4')[:]=[128,ids[1],owner,d['sides'][side]['modes']]
                s[4].view('u4')[:]=[first,n,offset,len(nearfaces)];offset+=len(nearfaces)
                neighbors.append((first+nearfaces).astype('u4'));s[5].view('u4')[:]=nodes+node_offset;s[6]=bary
                chunks.setdefault(group,[]).append(s)
        groups=sorted(chunks);capacity=256//len(groups)
        if capacity<3:raise ValueError('Too few independent active sites per pressure region')
        for j,key in enumerate(groups):
            entries=chunks[key];domain.append(np.stack(entries));rules.append([start,len(entries),side*256+j*capacity,capacity]);start+=len(entries)
            group_map.append(dict(side=side,source_region=key,records=len(entries),slots=capacity))
        node_offset+=len(p['vertices'])
    for name,data in [('pool-material-points',np.concatenate(domain)),('pool-cortical-neighborhood',np.concatenate(neighbors)),('pool-material-regions',np.asarray(rules,'u4'))]:
        f=out/(name+'.bin');data.tofile(f);d['files'][f.name]=sha(f)
    d['contact_pool']=dict(count=start,groups=group_map,reference_band_m=.002,neighborhood_m=.004)
    (out/'manifest.json').write_text(json.dumps(d,indent=2)+'\n')
    print(start,len(rules),sum(len(k) for k in neighbors),[(g['side'],g['slots']) for g in [group_map[0],group_map[-1]]])
if __name__=='__main__':build(*map(Path,sys.argv[1:]))
