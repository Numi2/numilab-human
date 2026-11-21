"""Compile the original registered P1 boundary without rounding its weights to one float."""
import json,sys
from pathlib import Path
import numpy as np
from material_context import sha

def prepare(part,base,out):
    doc=json.loads((base/'manifest.json').read_text())
    ref=np.fromfile(base/'pelvic-reference-nodes.bin','f4').reshape(-1,3,4).astype(float)
    ref=ref[:,0,:3]+ref[:,1,:3]
    bind=np.fromfile(base/'closed-material-vertices.bin','f4').reshape(-1,2,4)
    chunks=[]
    for side,spec in enumerate(doc['contact_flexibility']):
        m=spec['muscle'];p=part/'conforming-boundary'/f'{m}.npz'
        if sha(p)!=doc['surface_material_reference']['sides'][side]['source']:raise ValueError('Stale material partition')
        shape=np.load(p);w=shape['vertex_shape'];ids=shape['node']+spec['first_node']
        g=doc['closed_reverse']['sides'][side];rows=bind[g['first_vertex']:g['first_vertex']+g['vertices']]
        if not np.array_equal(ids,rows[:,1].copy().view('u4')) or not np.array_equal(w.astype('f4'),rows[:,0]):
            raise ValueError('Original P1 weights or material-node identities differ')
        xyz=np.einsum('ij,ijk->ik',w,ref[ids])
        packed=np.zeros((len(w),3,4),dtype='f4');packed[:,0,:3]=xyz
        packed[:,1,:3]=xyz-packed[:,0,:3].astype(float);packed[:,2]=w-rows[:,0].astype(float)
        if np.abs(packed[:,0,:3].astype(float)+packed[:,1,:3].astype(float)-xyz).max()>1e-15:
            raise ValueError('Split rest position lost precision')
        chunks.append(packed)
    out.mkdir(exist_ok=False)
    for p in base.iterdir():
        if p.name!='manifest.json':(out/p.name).hardlink_to(p)
    data=out/'constitutive-boundary-reference.bin'
    np.concatenate(chunks).tofile(data)
    doc['constitutive_boundary']={'source_manifest':sha(base/'manifest.json'),'rows':sum(map(len,chunks)),
        'layout':'P1 reference high, reference low, original weight residual'}
    doc['files'][data.name]=sha(data)
    (out/'manifest.json').write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
