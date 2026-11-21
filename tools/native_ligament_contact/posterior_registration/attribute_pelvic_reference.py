"""Separate rigid-reference rounding from actual accepted deformations in the cut-muscle residual."""
import csv,json,sys
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
from audit_complete_pelvic_forces import material
from qualify_material_motion import frame
from material_context import sha

def audit(part,response,assets,receipt,state,out):
    data=json.loads((assets/'manifest.json').read_text());r=json.loads(receipt.read_text())
    ref=frame(next(s for s in r['initial_anatomical_registration']['body_poses'] if s['body_index']==128))
    rot=Rotation.from_quat(ref[3:]).inv();mu,lam=(data['specimen'][i] for i in ('mu','lam'))
    with state.open() as f:rows=list(csv.DictReader(f))
    def cols(k):
        by={int(z['index']):np.array([z[q] for q in 'xyz'],'f4').astype(float) for z in rows if z['field']==k}
        return np.stack([by[i] for i in range(len(by))])
    y=cols('material_node');u=cols('node_displacement');all_inv=np.fromfile(assets/'material-inverse.bin','f4').reshape(-1,3,4)[:,:,:3].astype('f8')
    first=0;idx=0;result={}
    for sid in (29,30):
        p=np.load(part/'projected'/f'{sid}.npz');b=np.load(response/f'{sid}.npz');x=p['vertices'].astype(float);mask=p['cell_volumes']>0;vol=p['cell_volumes'][mask];t=p['cells'][mask]
        free=np.setdiff1d(np.unique(t),b['fixed']);rest=rot.apply(x-ref[:3]);D=x[t[:,1:]]-x[t[:,:1]];ideal=np.linalg.inv(D);native=all_inv[first:first+len(t)]
        side=y[idx:idx+len(x)]-u[idx:idx+len(x)]
        obs={}
        for label,xyz,inv in [('rigid_double',rest,ideal),('packed_inverse',rest,native),('packed_inverse_and_positions',rest.astype('f4').astype(float),native),('donor_at_accepted_pose',side,native)]:
            f,e,J=material(xyz,t,inv,vol,mu,lam);r=np.linalg.norm(f[free],axis=1);obs[label]=dict(maximum_N=float(max(r)),node=int(free[np.argmax(r)]),energy_J=float(e),min_J=float(min(J)),max_J=float(max(J)))
        marker=np.linalg.norm(side-rest,axis=1);obs['max_donor_motion_vs_reference_m']=float(max(marker))
        result[sid]=obs;first+=len(t);idx+=len(x)
    out.write_text(json.dumps(dict(accepted_step=int(rows[0]['accepted_step']),snapshot=sha(state),reference=sha(receipt),sides=result),indent=2)+'\n')
if __name__=='__main__':audit(*map(Path,sys.argv[1:]))
