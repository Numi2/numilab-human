"""Resolve material boundaries using the consistently oriented input faces and their exact arrangement."""
from pathlib import Path
import json,sys
import igl
import numpy as np
from igl.copyleft import cgal
from repair_registered_surface import per_pose,state,volume
from registration_stages import parts
from prepare_shoulder import read
from compile_connected_joint import digest


def parents(new,fnew,old,fold,birth,weights):
    faces=fold[birth]
    old_tri=old[faces]
    ba=igl.barycentric_coordinates(
        new[fnew].reshape(-1,3),
        *[np.repeat(old_tri[:,k],3,axis=0)for k in range(3)])
    ba=ba.reshape(-1,3,3)
    if not np.isfinite(ba).all() or ba.min()< -1e-5 or ba.max()>1+1e-5:raise ValueError('Lost source ancestry')
    recon=np.einsum('fij,fjk->fik',ba,old_tri)
    if np.max(abs(recon-new[fnew]))>1e-7:raise ValueError('Off-parent material vertex')
    local=np.einsum('fij,fjk->fik',ba,weights[faces])
    w=np.zeros((len(new),weights.shape[1]));np.add.at(w,fnew,local)
    w/=np.bincount(fnew.ravel(),minlength=len(new))[:,None]
    return w,ba,float(np.max(abs(local-w[fnew])))


def build(case,base,out):
    out.mkdir(exist_ok=True)
    cfg=read(case/'launch.json');a=cfg['argv'];src=Path(a[a.index('--soft-tissue-payload')+1])
    opening=read(case/'native-initial-bodies.json')['all_bodies'];rc=next((case/'native-traces/accepted-geometry').glob('*.receipt.json'))
    poses=[{r['body_index']:dict(quaternion_xyzw=r[k]['orientation_world_xyzw'],position_m=r[k]['position_world_m']) for r in opening} for k in ['rest','current']]
    poses.append({r['body_index']:r for r in read(rc)['accepted_body_poses']})
    rows={}
    for sid in (75,85,87):
        _,_,owners,_,aff=parts(src,sid)
        for label in ('current','atlas'):
            trial=np.load(base/f'{sid}-{label}.npz');x=trial['source'].astype('f4').astype('f8');f=trial['faces'];w=trial['weights']
            if volume(x,f)<=0:raise ValueError('Source must be outward')
            if len(cgal.remesh_self_intersections(x,f,detect_only=True)[2]):
                y,fy,birth=cgal.mesh_boolean(x,f,type_str='union')
                wy,ba,conflict=parents(y,fy,x,f,birth,w)
            else:
                y,fy,wy=x,f,w;birth=np.arange(len(f));ba=np.broadcast_to(np.eye(3),(len(f),3,3));conflict=0.
            tag=f'{sid}-{label}';path=out/(tag+'.npz')
            ys=y.astype('f4').astype('f8');hits=per_pose(ys,wy,fy,aff,owners,poses)
            # Distinguish edge-manifold orientation from disconnected vertex links and internal cavities.
            rows[tag]=dict(hits=[len(h)for h in hits],faces=len(fy),vertices=len(y),nonmanifold_vertices=int((~igl.is_vertex_manifold(fy)).sum()),
                connected_components=len(np.unique(igl.vertex_components(fy))),edge_manifold=bool(igl.is_edge_manifold(fy)[0]),source_weight_disagreement=conflict,
                delta_volume_mm3=(volume(ys,fy)-volume(x,f))*1e9)
            np.savez_compressed(path,source=x,source_faces=f,vertices=ys,weights=wy,faces=fy,birth=birth,barycentric=ba,accepted=state(ys,wy,aff,owners,poses[-1]).astype('f4'))
            print(tag,rows[tag],flush=True)
    (out/'results.json').write_text(json.dumps(dict(rows=rows,source=digest(src),capture=digest(rc)),indent=2)+'\n')
if __name__=='__main__':build(*(Path(a).resolve() for a in sys.argv[1:]))
