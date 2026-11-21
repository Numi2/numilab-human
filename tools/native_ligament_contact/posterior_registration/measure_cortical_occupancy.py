"""Measure deformed material inside the actual cortical solids, including the interior of every cut cell."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from solve_pelvic_contact import poses_at,current_bank
from prepare_material_interface import read_inputs,solid
from qualify_material_motion import frame,local
from project_material_cells import make_solid,FACE
from material_context import sha
def run(p47,prev,base,response,native,candidate,out,step):
    rc=prev/'native-release-motion/step-8.receipt.json';surfs,_=read_inputs(prev/'release-posterior/config.json',rc,prev/'native-release-motion',base)
    full={r['body_index']:frame(r) for r in json.loads(rc.read_text())['initial_anatomical_registration']['body_poses']}
    _,pose=poses_at(native,step);stat={}
    for sid,owners in ((29,[(1,1),(2,131),(18,128)]),(30,[(1,1),(3,145),(19,128)])):
        p,x,*_=current_bank(sid,p47,prev,response,native,step)
        d=np.load(p47/'partition-layers'/f'muscle_{sid}.npz');shape=make_solid(d['vertices'],d['triangles'])[0]
        obj={}
        for key,b in owners:
            verts,faces=surfs[f'bone_{key}'];points=Rotation.from_quat(pose[b][3:]).apply(local(verts,full[b]))+pose[b][:3]
            obj[key]=solid(points,faces,'bone')[0]
        v=p['vertices'].astype(float);t=p['cells'];vol=p['cell_volumes'];new=np.load(candidate/f'{sid}.npz')['positions']
        witnesses=[]
        res={k:{key:dict(overlap_ml=0.,largest_cell_ml=0.,cell=None) for key in obj} for k in ['before','after']}
        for i in np.flatnonzero(vol>0):
            verts=v[t[i]];cut=make_solid(verts,FACE)[0]^shape;inv=np.linalg.inv(verts[1:]-verts[:1])
            for tag,pos in (('before',x),('after',new)):
                current=pos[t[i]];F=(inv@(current[1:]-current[:1])).T;b=current[0]-F@verts[0]
                moved=cut.transform(np.column_stack((F,b)))
                for key,body in obj.items():
                    piece=moved^body;volume=piece.volume()*1e6
                    if not np.isfinite(volume) or volume < -1e-11:raise ValueError((i,key,volume))
                    m=res[tag][key];m['overlap_ml']+=volume
                    if tag=='after' and volume>1.e-15:
                        poly=piece.to_mesh64();coords=np.asarray(poly.vert_properties)[:,:3];tri=np.asarray(poly.tri_verts)
                        # All barycentric coordinates refer to this actual deformed P1 material cell.
                        # Face midpoints and subvolume centroids add witnesses missed by surface-only sampling.
                        pts=np.vstack((coords,coords[tri].mean(1),(coords[tri].sum(1)+coords.mean(0))*.25))
                        u=(pts-current[0])@np.linalg.inv(current[1:]-current[:1]);w=np.column_stack((1-u.sum(1),u))
                        witnesses.append((i,key,w))
                    if volume>m['largest_cell_ml']:m.update(largest_cell_ml=volume,cell=int(i))
        if witnesses:
            wi=np.concatenate([np.full(len(w),i) for i,b,w in witnesses])
            ww=np.vstack([w for i,b,w in witnesses]);bb=np.concatenate([np.full(len(w),b) for i,b,w in witnesses])
            np.savez_compressed(out.with_name(out.stem+f'-{sid}.npz'),cell=wi,weights=ww,bone=bb)
        stat[sid]=res;print(sid,res,flush=True);out.write_text(json.dumps(dict(step=step,input=sha(candidate/'result.json'),results=stat),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:8]),int(sys.argv[8]))
