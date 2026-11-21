"""Bind both donor transforms to the same undeformed material coordinates."""
import json,sys
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
from material_context import sha
from qualify_material_motion import frame
def prepare(part,base,receipt,out):
    out.mkdir(exist_ok=True)
    for p in base.iterdir():
        if p.name!='manifest.json' and not (out/p.name).exists():(out/p.name).hardlink_to(p)
    d=json.loads((base/'manifest.json').read_text());parents=np.fromfile(base/'material-nodes.bin','f4').reshape(-1,3,4)
    frames={r['body_index']:frame(r) for r in json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses']}
    pp=frames[128];rp=Rotation.from_quat(pp[3:]);offset=0;refs=[];invs=[];regs=[];poses=[]
    for i in [128,131,145]:
        v=frames[i];poses.extend([[*v[:3],0],v[3:]])
    for sid,b in ((29,131),(30,145)):
        p=np.load(part/'projected'/f'{sid}.npz');v=p['vertices'].astype(float);y=rp.inv().apply(v-pp[:3]);rf=Rotation.from_quat(frames[b][3:]);f=rf.inv().apply(v-frames[b][:3]);h=parents[offset:offset+len(v)]
        disagreement=max(np.linalg.norm(y-h[:,0,:3],axis=1)[h[:,0,3]>1e-7].max(),np.linalg.norm(f-h[:,1,:3],axis=1)[h[:,1,3]>1e-7].max())
        if disagreement>1e-6 or max(abs(h[:,0,3]+h[:,1,3]-1))>2e-7:raise ValueError(('Donor registration disagrees with physical mesh',sid,disagreement))
        arr=np.zeros((len(v),3,4),'f4');arr[:,0,:3]=y;arr[:,1,:3]=y-arr[:,0,:3].astype(float);arr[:,2,:3]=f;refs.append(arr)
        t=p['cells'][p['cell_volumes']>0];D=y[t[:,1:]]-y[t[:,:1]];inverse=np.zeros((len(t),3,4),'f4');inverse[:,:,:3]=np.linalg.inv(D);invs.append(inverse)
        regs.append(dict(muscle=sid,largest_active_donor_difference_m=disagreement,source=sha(part/'projected'/f'{sid}.npz')));offset+=len(v)
    for key,table in [('pelvic-reference-nodes',np.concatenate(refs)),('pelvic-reference-inverse',np.concatenate(invs)),('pelvic-reference-poses',np.asarray(poses,'f4'))]:
        f=out/(key+'.bin');table.tofile(f);d['files'][f.name]=sha(f)
    d['common_material_reference']={'pose':sha(receipt),'regions':regs}
    (out/'manifest.json').write_text(json.dumps(d,indent=2)+'\n');print(regs)
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
