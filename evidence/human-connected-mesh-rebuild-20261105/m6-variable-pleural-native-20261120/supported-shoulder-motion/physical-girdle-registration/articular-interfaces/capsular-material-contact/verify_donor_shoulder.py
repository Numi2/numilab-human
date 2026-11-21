"""Bound the viability of the native donor frame using the donor bones and the full-scene bed reference."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from compare_atlas import closest_distance
from verify_source_wraps import closest,source_model,path
from shoulder_inputs import digest,old
from source_pair_audit import inputs
from fast_skin_intersection_audit import install_for_float32
from bone_overlap import intersections

def run(launch,anchors,motion,donor,src,zero,out):
    _,poses,rest,_=inputs(launch,anchors,motion)
    cap={r['body_index']:np.array(r['position_m']+r['quaternion_xyzw']) for r in json.loads(zero.read_text())['accepted_body_poses']}
    poses.update({0:rest,-4:cap});meshes=np.load(donor);install_for_float32();rows=[]
    for r in json.loads(src.read_text())['routes']:
        if not r['name'].startswith('s_glen'):continue
        ids=r['sites'];shapes=[];ends=[];topology=[];bind=[]
        for label,site in zip(['humerus','scapula'],ids):
            label+='_'+r['side'];v=meshes[label+'_local'];f=meshes[label+'_faces']
            dist,face=closest_distance(v[f],site['point_local_m'])
            w=closest(v[f[face]],np.array(site['point_local_m']));ep=w@v[f[face]]
            _,inv=np.unique(v,axis=0,return_inverse=True);fi=inv[f]
            edges=np.sort(np.vstack((fi[:,[0,1]],fi[:,[1,2]],fi[:,[2,0]])),axis=1)
            _,n=np.unique(edges,axis=0,return_counts=True)
            topology.append(dict(mesh=label,boundary_edges=int((n==1).sum()),nonmanifold_edges=int((n>2).sum())))
            bind.append(dict(face=face,offset_m=dist,local_m=ep.tolist(),barycentric=w.tolist()))
            shapes.append((v,f));ends.append(ep)
        m,d=source_model(*ends,r['source_wrap'])
        result=[]
        for st in [-4,-3,-2,-1,0,32,256,8000]:
            poseset=[poses[st][o['body']] for o in ids];world=[];xf=[]
            for (v,f),q in zip(shapes,poseset):
                rot=R.from_quat(q[3:]);world.append(((q[:3]+rot.apply(v)).astype('f4'),f))
                xf.append((q[:3],rot))
            x,l,sag=path(m,d,*poseset,r['source_wrap']['center_local_m'])
            witnesses=[]
            for bi,((v,f),(q,rot)) in enumerate(zip(shapes,xf)):
                y=rot.inv().apply(x-q)
                for j,(a,b) in enumerate(zip(y,y[1:])):
                    hits=old.audit.lp.vector_hits(a,b,(v,f))
                    if hits:witnesses.append([bi,j,len(hits),min(hits)])
            result.append(dict(step=st,baseline_pairs=len(intersections(*world[0],*world[1])),
             path_length_m=l,chord_count=len(x)-1,faces_hit=witnesses,max_arc_sag_m=sag))
        rows.append(dict(side=r['side'],source=r['name'],topology=topology,sites=bind,samples=result))
    out.write_text(json.dumps(dict(source_sha256=digest(src),donor_mesh_sha256=digest(donor),rows=rows),indent=2,default=lambda o:o.item())+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
