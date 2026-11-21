"""Intersect each accepted, deformed cut cell with the pinned, moving cortical solids."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_material_interface import solid
from project_material_cells import make_solid,FACE
from material_context import sha

def run(partition,assets,snapshot,out):
    with snapshot.open() as f:rows=list(csv.DictReader(f))
    steps={int(r['accepted_step']) for r in rows};assert len(steps)==1
    def values(key):
        return {int(r['index']):np.array([float(r[k]) for k in 'xyzw'],dtype='f4') for r in rows if r['field']==key}
    pos=values('relative_position');rot={b:Rotation.from_quat(q) for b,q in values('orientation').items()}
    nodes=values('material_node');node=np.stack([nodes[i][:3] for i in range(len(nodes))]).astype(float)
    table=np.fromfile(assets/'material-nodes.bin',dtype='f4').reshape(-1,3,4)
    modes=np.fromfile(assets/'material-modes.bin',dtype='f4').reshape(len(table),96,4)
    modal=values('coordinate_block');modal=np.stack([modal[i] for i in range(48)]).reshape(2,96)
    bank=np.fromfile(assets/'cortical-samples.bin',dtype='f4').reshape(2,256,5,4)
    triangles=np.fromfile(assets/'cortical-triangles.bin',dtype='f4').reshape(-1,10,4)
    manifest=json.loads((assets/'manifest.json').read_text());models={}
    stats={};num=0
    for side,(sid,fem,bones) in enumerate([(29,131,(1,2,18)),(30,145,(1,3,19))]):
        pr=np.load(partition/'projected'/f'{sid}.npz');clip=np.load(partition/'partition-layers'/f'muscle_{sid}.npz')
        mesh=make_solid(clip['vertices'],clip['triangles'])[0]
        v=pr['vertices'].astype(float);t=pr['cells'];has=pr['cell_volumes']>0
        stop=num+len(v);records=table[num:stop];current=node[num:stop]
        shift=rot[128].inv().apply(pos[fem][:3])
        x=records[:,0,:3]*records[:,0,3:]+(shift+rot[128].inv().apply(rot[fem].apply(records[:,1,:3])))*records[:,1,3:]
        reconstructed=x+np.einsum('nkj,k->nj',modes[num:stop,:,:3],modal[side])
        err=np.linalg.norm(reconstructed-current,axis=1).max()
        if err>3.e-7:raise ValueError(('GPU node/modal or pose mismatch',sid,err))
        ports=np.load(partition/'ports'/f'{sid}.npz')
        for j,i in enumerate(manifest['sides'][side]['quadrature_indices']):
            oid=int(ports['target_bank'][i][5:])
            if oid in models:continue
            owner=int(ports['target_body'][i]);a,n=(int(k) for k in bank[side,j,4].view('u4')[:2])
            source=triangles[a:a+n,:3,:3].reshape(-1,3);xyz=rot[128].inv().apply(rot[owner].apply(source)+pos[owner][:3])
            models[oid]=solid(xyz,np.arange(3*n).reshape(-1,3),'cortex')[0]
        obs={b:{'source_bound_ml':0.,'accepted_material_ml':0.,'largest_cell_ml':0.,'worst_cell':None,'positive_cells':0} for b in bones}
        for i in np.flatnonzero(has):
            vertices=v[t[i]];cut=make_solid(vertices,FACE)[0]^mesh
            inv=np.linalg.inv(vertices[1:]-vertices[:1])
            for key,coords in [('source_bound_ml',x),('accepted_material_ml',current)]:
                pts=coords[t[i]];F=(inv@(pts[1:]-pts[:1])).T;shift=pts[0]-F@vertices[0]
                moved=cut.transform(np.column_stack((F,shift)))
                for b in bones:
                    part=moved^models[b];vol=part.volume()*1e6
                    if not np.isfinite(vol) or vol < -1.e-11 or str(part.status())!='Error.NoError':raise ValueError((i,b,vol,str(part.status())))
                    d=obs[b];d[key]+=vol
                    if key=='accepted_material_ml' and vol>0:
                        d['positive_cells']+=1
                        if vol>d['largest_cell_ml']:d['largest_cell_ml']=vol;d['worst_cell']=int(i)
        stats[sid]=dict(cortex=obs,number_cut_cells=int(has.sum()),maximum_reconstruction_error_m=float(err));num=stop
    assert num==len(node)
    out.write_text(json.dumps(dict(accepted_step=steps.pop(),snapshot_sha256=sha(snapshot),registration_sha256=sha(assets/'manifest.json'),muscles=stats),indent=2)+'\n')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('partition','assets','snapshot','out'):parser.add_argument(name,type=Path)
    run(**vars(parser.parse_args()))
