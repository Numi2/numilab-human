"""Reconstruct source-to-world affine donor maps against native body traces."""
import csv,json,struct
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from bilateral_bindings import decode
def states(trajectory,receipt):
    rc=json.loads(receipt.read_text())
    poses={}
    for row in (rc['initial_anatomical_registration']['body_poses']):
        poses[row['body_index']]=np.r_[row['position_m'],row['quaternion_xyzw']]
    frames={0:poses}
    for r in csv.DictReader(trajectory.open()):
        j=int(r['body_index'])
        if j not in (128,131,145):continue
        step=int(r['accepted_step'])
        frames.setdefault(step,{})[j]=np.r_[[float(r['root_'+k+'_m'])+float(r['com_relative_root_'+k+'_m']) for k in 'xyz'],[float(r['orientation_'+k]) for k in 'xyzw']]
    return frames
def parts(payload,sid):
    ident=np.tile([0,0,0,0,0,0,1.],(157,1));src,_,weights,donors,faces=decode(payload,[sid],ident)[sid]
    raw=payload.read_bytes();nr=struct.unpack_from('<I',raw,12)[0];rows=np.frombuffer(raw,'<u4',nr*8,64).reshape(-1,8)
    row=next(r for r in rows if r[6]==sid)
    aff=[]
    for k,b in enumerate(donors):
        body,*param=struct.unpack_from('<I8f',raw,64+nr*32+36*(int(row[0])+k))
        if b!=body:raise ValueError('Wrong body frame')
        aff.append((Rotation.from_quat(param[3:7]).as_matrix()*param[7],np.array(param[:3])))
    return src.astype(float),weights,donors,faces,aff
def model(data,pose,sign):
    src,w,bodies,f,aff=data
    pos=[];dirs=[];mat=[]
    for j,b in enumerate(bodies):
        r=Rotation.from_quat(pose[b][3:]).as_matrix();A=r@aff[j][0];B=pose[b][:3]+r@aff[j][1]
        pos.append(src@A.T+B);dirs.append(sign*A[:,0]);mat.append(A)
    return w[:,0,None]*pos[0]+w[:,1,None]*pos[1],w@np.array(dirs),pos[1]-pos[0],mat
def plane(frames,step):
    reference=frames[0];x=(reference[145][:3]-reference[131][:3]);x/=np.linalg.norm(x)
    top=(reference[145][:3]+reference[131][:3])/2
    ref=Rotation.from_quat(reference[128][3:]);rot=Rotation.from_quat(frames[step][128][3:])*ref.inv()
    return rot.apply(top-reference[128][:3])+frames[step][128][:3],rot.apply(x)
