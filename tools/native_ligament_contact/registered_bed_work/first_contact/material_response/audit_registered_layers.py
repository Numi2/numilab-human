"""Run the retained exact three-layer predicate on the material mesh in reference and posed coordinates."""
import json,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from layer_contacts import run as layer_audit

def run(job,lib,out):
    e=json.loads((job/'registered-donor-cook/scene/launch.json').read_bytes())['environment']
    z=np.load(job/'registered-boundary/registered-boundary.npz');groups=[np.asarray(z[k+'_node_joint'][z[k+'_face_nodes']],dtype='i4') for k in 'OID'];out.mkdir()
    files=[Path(e['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE']),Path(e['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'])]
    for name,f in zip(('reference','posed'),files):
        x=np.loadtxt(f,skiprows=1,usecols=(1,2,3),dtype='f4').astype('f8')
        layer_audit(x,groups,out/(name+'.json'),lib)
        r=json.loads((out/(name+'.json')).read_bytes());print(name,r['pair_counts'],flush=True)
        (out/(name+'-identity.json')).write_text(json.dumps(dict(nodes=digest(f),faces=digest(job/'registered-boundary/registered-boundary.npz'),predicate=digest(lib),full_counts=r['pair_counts'],broad_phase_pairs=r['aabb_pairs']),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
