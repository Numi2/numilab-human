"""Emit the free-node material stiffness for independent GPU end-of-iteration residual certification."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from pelvic_elastic_response import stiffness
from pelvic_specimen import properties
from qualify_material_motion import frame
from material_context import sha
def prepare(part,response,base,receipt,law,out):
    out.mkdir(exist_ok=True)
    for p in base.iterdir():
        if p.is_file() and p.name!='manifest.json' and not (out/p.name).exists():(out/p.name).hardlink_to(p)
    d=json.loads((base/'manifest.json').read_text());spec=properties(law)
    root=json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses'];R=Rotation.from_quat(frame(next(b for b in root if b['body_index']==128))[3:]).inv()
    ptr=[];columns=[];values=[];offset=0;start=0;hist={}
    for side,sid in enumerate((29,30)):
        p=np.load(part/'projected'/f'{sid}.npz');b=np.load(response/f'{sid}.npz');x=p['vertices'].astype(float);x=R.apply(x-x.mean(0));t=p['cells'];mask=p['cell_volumes']>0
        A,_=stiffness(x,t[mask],p['cell_volumes'][mask],spec['mu'],spec['lam']);K=A.tobsr(blocksize=(3,3))
        free=set(np.setdiff1d(np.unique(t[mask]),b['fixed']));rng=np.random.default_rng(sid);u=rng.normal(size=x.shape)*.001
        check=np.zeros_like(u)
        for i in range(len(x)):
            ids=K.indices[K.indptr[i]:K.indptr[i+1]];blocks=K.data[K.indptr[i]:K.indptr[i+1]]
            keep=(ids!=i)&(np.any(blocks!=0,axis=(1,2)));ids=ids[keep];blocks=blocks[keep]
            ptr.append([offset,len(ids),int(i in free),side]);offset+=len(ids);columns.extend(ids+start)
            v=np.zeros((len(ids),3,4),'f4');v[:,:,:3]=blocks.transpose(0,2,1);values.append(v)
            check[i]=np.einsum('nij,nj->i',v[:,:,:3].transpose(0,2,1).astype(float),u[ids]-u[i])
        exact=(A@u.ravel()).reshape(-1,3);err=np.linalg.norm(exact-check)/np.linalg.norm(exact)
        if err>1e-5:raise ValueError(('Independent residual emission failed',err))
        hist[sid]=float(err);start+=len(x)
    for name,data in [('free-node-rows',np.asarray(ptr,'u4')),('free-node-columns',np.asarray(columns,'u4')),('free-node-stiffness',np.concatenate(values))]:
        f=out/(name+'.bin');data.tofile(f);d['files'][f.name]=sha(f)
    d['free_residual_float32_parity']=hist;(out/'manifest.json').write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
