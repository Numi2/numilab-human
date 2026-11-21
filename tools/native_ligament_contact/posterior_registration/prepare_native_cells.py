"""Pin native bulk and all cut-carrier positions for the same cortical contact candidate."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pose_glutes import parts
from qualify_material_motion import frame
from material_context import sha
from pelvic_specimen import properties
def run(p47,prev,response,material,out):
    cfg=json.loads((prev/'release-posterior/config.json').read_text())['argv']
    asset=Path(cfg[cfg.index('--soft-tissue-payload')+1]);rc=json.loads((prev/'native-release-motion/step-8.receipt.json').read_text())
    ref=frame(next(r for r in rc['initial_anatomical_registration']['body_poses'] if r['body_index']==128));rot=Rotation.from_quat(ref[3:]).inv().as_matrix()
    nodes=[];modes=[];cells=[];inverse=[];start=0
    for si,sid in enumerate((29,30)):
        p=np.load(p47/'projected'/f'{sid}.npz');a=parts(asset,sid);x,w,ids,_,aff=a
        node=np.zeros((len(x),3,4),'f4');st=np.zeros((len(x),96,4),'f4')
        for k in range(2):node[:,k,:3]=x@aff[k][0].T+aff[k][1];node[:,k,3]=w[:,k]
        m=np.load(response/f'{sid}.npz')['modes'];node[:,2,:]=np.array([si,ids[0],ids[1],m.shape[-1]],'u4').view('f4')
        st[:,:m.shape[-1],:3]=np.einsum('ij,njk->nki',rot,m)
        t=p['cells'][p['cell_volumes']>0];v=p['vertices'].astype(float);ds=v[t[:,1:]]-v[t[:,:1]];inv=np.linalg.inv(ds)
        blocks=np.zeros((len(t),3,4),'f4');blocks[:,:,:3]=inv
        cells.append(t+start);inverse.append(blocks);nodes.append(node);modes.append(st);start+=len(x)
    ver=out/'manifest.json';d=json.loads(ver.read_text())
    for name,v in [('material-nodes',nodes),('material-modes',modes),('material-cells',cells),('material-inverse',inverse)]:
        file=out/(name+'.bin');np.concatenate(v).tofile(file);d['files'][file.name]=sha(file)
    spec=properties(material);d['specimen']=spec;d['material_sha256']=sha(material);ver.write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
