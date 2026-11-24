"""Measure body-local bounding-box culling on the actual installed contact coordinates."""
from pathlib import Path
import re,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from rebind_ligament_material import banks

def run(job,scene,output):
    load=lambda p:json.loads(p.read_bytes());cfg=load(scene/'launch.json');material=job/'interval-certified-scene/material';bone=Path(cfg['argv'][3])
    pose=job/'distal-registered-native/target-pose.json';capture=load(pose)['all_bodies'];audit=load(job/'interval-certified-scene/bone-audit/result.json')
    if audit['bone']!=digest(bone) or audit['capture']!=digest(pose) or cfg['pins'][str(bone)]!=audit['bone']:raise ValueError('Mixed body frames')
    bound=np.load(job/'registered-boundary/registered-boundary.npz');cort=np.load(job/'interval-certified-scene/contact/material-contact.npz')['samples']
    xyz=np.loadtxt(material/'current.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    if digest(material/'current.node')!=load(material/'result.json')['initial']:raise ValueError('Invalid starting pose')
    log=(scene/'stdout.log').read_text();anchor=list(map(int,re.findall(r'whole_body_skin_attachment source_vertex=\d+ donor_body=\d+ local_vertex=(\d+)',log)))
    if len(set(anchor))!=14:raise ValueError('Incorrect attached deep-fat surface')
    nodes=np.unique(np.r_[bound['O_node_joint'],np.delete(bound['D_node_joint'],anchor),cort[:,0].astype(int)])
    if len(nodes)!=182244:raise ValueError('Native nodal contact ownership count disagrees')
    xyz=xyz[nodes];body={}
    for (owner,bank),(_,v,_) in banks(bone).items():body.setdefault(owner,[]).append(v)
    if len(body)!=86:raise ValueError('Incomplete body banks')
    counts=[0,0];maximum=[0,0];per_point=np.zeros((len(nodes),2),dtype='u2')
    for owner,vs in body.items():
        entry=capture[owner]
        if entry['body_index']!=owner:raise ValueError('Incorrect rigid pose ordering')
        q=entry['current'];vp=np.concatenate(vs);p=Rotation.from_quat(q['orientation_world_xyzw']).inv().apply(xyz-q['position_world_m'])
        # More generous candidate bounds than the registered pose precision.
        low=vp.min(0)-2e-5;high=vp.max(0)+2e-5;away=np.maximum(np.maximum(p-high,low-p),0);d2=np.einsum('ij,ij->i',away,away)
        for k,band in enumerate([.05,0.]):
            guard=128*np.finfo('f4').eps*(1.+np.max(abs(p),axis=1))**2
            near=d2<=band**2+guard
            counts[k]+=int(near.sum());per_point[:,k]+=near
    total=len(nodes)*len(body)
    output.write_text(json.dumps(dict(mesh=digest(material/'current.node'),bone=audit['bone'],capture=audit['capture'],nodes=len(nodes),bodies=len(body),all_pairs=total,band_50mm_candidates=counts[0],closed_solid_candidates=counts[1],max_bodies_per_node=per_point.max(axis=0).astype(int).tolist(),scope='Starting pose only. Conservative box comparison in bone-local metres with padded extents and squared-distance margin. Recompute on the GPU at every candidate. No measured speedup.'),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
