"""Use the accepted GPU displacements and cortical forces to recompute the free nodal residual and elastic energy."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pelvic_elastic_response import stiffness
from pelvic_specimen import properties
from qualify_material_motion import frame
from material_context import sha
from kinetic_weights import contact_weights

def check(part,resp,assets,receipt,material,snapshot,out):
    with snapshot.open() as f:rows=list(csv.DictReader(f))
    def field(k):
        return {int(r['index']):np.array([float(r[a]) for a in 'xyzw']) for r in rows if r['field']==k}
    contacts=field('normal_and_force');displacement=field('node_displacement');accepted_shape=field('reverse_shape');accepted_nodes=field('reverse_parent_nodes')
    if not displacement or not contacts:raise ValueError('Unpublished accepted material/traction state')
    pset=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4)
    meta=json.loads((assets/'manifest.json').read_text()).get('cortical_face_contacts')
    feature=np.fromfile(assets/'cortical-face-weights.bin','f4').reshape(-1,3,4).astype(float) if meta else None
    threshold=meta['first'] if meta else len(pset)
    captured=field('material_node');points=np.stack([captured[i][:3] for i in range(len(captured))])
    rot_now={i:Rotation.from_quat(q) for i,q in field('orientation').items()};pos_now=field('relative_position')
    frames=json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses']
    rot=Rotation.from_quat(frame(next(r for r in frames if r['body_index']==128))[3:]).inv()
    law=properties(material);start=0;stat={}
    for side,(sid,body) in enumerate(((29,131),(30,145))):
        q=np.load(part/'projected'/f'{sid}.npz');red=np.load(resp/f'{sid}.npz')
        x=q['vertices'].astype(float);v=rot.apply(x-x.mean(0));t=q['cells'];valid=q['cell_volumes']>0
        K,_=stiffness(v,t[valid],q['cell_volumes'][valid],law['mu'],law['lam'])
        free=np.setdiff1d(np.unique(t[valid]),red['fixed']);u=np.stack([displacement[start+i][:3] for i in range(len(v))])
        external=np.zeros_like(u);count=0;maximum=0
        for i,val in contacts.items():
            row=pset[i]
            if row[3,1:2].copy().view('u4')[0]!=body or val[3]==0:continue
            nodes=(accepted_nodes[i].astype(int) if i in accepted_nodes else row[5].copy().view('u4').astype(int))-start
            assert val[3]>=0 and nodes.min()>=0 and nodes.max()<len(v)
            w=accepted_shape[i] if i in accepted_shape else contact_weights(i,pset,feature,threshold,points,rot_now,pos_now)
            np.add.at(external,nodes,w[:,None]*val[:3]*val[3]);count+=1;maximum=max(maximum,val[3])
        internal=np.asarray(K@u.ravel()).reshape(-1,3);res=(internal-external)[free]
        E=.5*np.sum(u*internal);work=np.sum(u*external)
        largest=float(np.linalg.norm(res,axis=1).max());amplitude=float(np.linalg.norm(external[free],axis=1).max())
        stat[str(sid)]=dict(binding_force_rows=count,largest_site_reaction_N=maximum,maximum_free_force_residual_N=largest,maximum_free_nodal_contact_force_N=amplitude,relative_free_force_residual=largest/max(amplitude,1.e-12),stored_elastic_energy_J=E,contact_virtual_work_J=work,relative_work_error=abs(2*E-work)/max(2*E,1e-18))
        start+=len(v)
    out.write_text(json.dumps(dict(accepted_step=int(rows[0]['accepted_step']),source=sha(snapshot),per_side=stat),indent=2)+'\n')
if __name__=='__main__':check(*map(Path,sys.argv[1:]))
