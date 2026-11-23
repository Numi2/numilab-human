"""Compare complete native root partitions at equal time and test a fitted patch on the alternate displacement."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from register_material_surfaces import sha
from supported_material_admission import certify, stress_limits


def compare(control,subdivision,candidate,out):
    def load(run):
        result=json.loads((run/'material-motion.json').read_bytes())
        if result['root_assistance'] or not result['variable_pleural']:
            raise ValueError('Unexpected release conditions')
        field='root_generalized_force_xyz_n'
        loads=[np.array(json.loads(re.search(field+r'=(\[[^\]]+\])',s)[1]))
               for s in (run/'stdout.log').read_text().splitlines() if s.startswith('accepted_fem_body_load_capture=accepted ')]
        return result,loads
    a,la=load(control);b,lb=load(subdivision)
    if (b['binary']!=a['binary'] or a['accepted_steps']!=1 or b['accepted_steps']!=2 or
            b['accepted_seconds']!=a['accepted_seconds'] or len(la)!=1 or len(lb)!=2):
        raise ValueError('Unmatched native time or source')
    if sha(control/'physical-reference.node')!=sha(subdivision/'physical-reference.node'):
        raise ValueError('Different initial loaded coordinates')
    x,y,t,mat,pins=inputs(control);x=x.astype(float);y=y.astype(float)
    data=np.load(candidate/'changed-coordinates.npz')
    if sha(candidate/'changed-coordinates.npz')!=json.loads((candidate/'result.json').read_bytes())['patch_sha256']:
        raise ValueError('Unverified proposal')
    cfg=json.loads((control/'launch.json').read_bytes())
    props,ident=materials(Path(cfg['compiled_material_owner']))
    raw=np.loadtxt(subdivision/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    full=np.loadtxt(control/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    delta=np.linalg.norm(raw-full,axis=1);x[data['ids']]=data['points']
    assessment=certify(x,x+(raw-y),t,mat,props,stress_limits(ident))
    r=dict(parent_launch=sha(control/'launch.json'),subdivided_launch=sha(subdivision/'launch.json'),
        native_pose_sha256=sha(subdivision/'physical-accepted.node'),physical_time_s=a['accepted_seconds'],
        one_root=a,two_roots=b,first_root_FEM_force_on_articulated_xyz_N={'one':la[0].tolist(),'two':lb[0].tolist()},
        initial_root_force_norm_ratio=float(np.linalg.norm(lb[0])/np.linalg.norm(la[0])),
        matched_time_displacement_difference={'rms_m':float(np.sqrt(np.mean(delta**2))),'max_m':float(delta.max()),'max_node':int(delta.argmax())},
        candidate_sha256=sha(candidate/'changed-coordinates.npz'),offline_patch_on_two_root_displacements=assessment)
    out.write_text(json.dumps(r,indent=2)+'\n')
    print('Base vs subdivided',a['material'],b['material'],flush=True)
    print('Patch on independent displacement',assessment,flush=True)


if __name__=='__main__':compare(*(Path(a).resolve() for a in sys.argv[1:]))
