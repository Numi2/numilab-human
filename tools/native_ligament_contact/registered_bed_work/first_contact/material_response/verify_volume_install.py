"""Exercise the same transactional file importer and anatomical attachments as the native scene."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from register_material_surfaces import PREFIX,pinned,sha


def run(launch, pose, material, surfaces, joints, facets, output):
    parent=json.loads(launch.read_bytes())['environment']
    surface_check=json.loads((surfaces/'native-acceptance.json').read_bytes())
    cmd=surface_check['build'].copy()
    output.mkdir()
    cmd[-1]=str(output/'verify-volume')
    subprocess.run(cmd,check=True)
    b=json.loads(pose.read_bytes())
    bindings={body['body_index']:i for i,body in enumerate(b['bodies'])}
    with (output/'native-body-frames.txt').open('w')as f:
        for body in b['all_bodies']:
            idx=body['body_index'];values=[]
            for phase in ('current','rest'):
                for k in ('position_world_m','local_to_world_row_major_f32'):
                    values.extend(np.asarray(body[phase][k],dtype='f4').astype('f8'))
            print(bindings.get(idx,2**32-1),idx,*(format(v,'.17g') for v in values),file=f)
    cfg=dict(parent,**surface_check['changed_settings'])
    def select(key,h,p,digest):
        if sha(p)!=digest:raise ValueError('Input changed: '+key)
        cfg[PREFIX+key]=str(p);cfg[PREFIX+h]=digest
    receipt=json.loads((material/'native-import.json').read_bytes())
    for k,name,h in [('TETGEN_NODE','mesh.node','node_sha256'),('TETGEN_ELEMENTS','mesh.ele','element_sha256'),('CHILD_LABELS','current-child.labels.csv','label_sha256')]:
        select(k,k+'_SHA256',material/name,receipt[h])
    mapping=json.loads((joints/'registration.json').read_bytes())
    select('COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256',joints/'combined-node-map.csv',mapping['combined_map_sha256'])
    facet=json.loads((facets/'facets.json').read_bytes())
    if facet['tetrahedra']!=receipt['element_sha256']:raise ValueError('Facets differ from candidate')
    select('FACET_ANCESTRY_CSV','FACET_ANCESTRY_SHA256',facets/'facet-ancestry-current.csv',facet['result'])
    for key,h in [('DEEP_ATTACHMENT_REQUESTS_CSV','DEEP_ATTACHMENT_REQUESTS_SHA256'),('PARENT_LABELS_CSV','PARENT_LABELS_SHA256'),('PARENT_PACKET_CSV','PARENT_PACKET_SHA256')]:
        pinned(parent,key,h)
    invocation=[str(output/'verify-volume'),str(output/'native-body-frames.txt'),str(len(b['all_bodies'])),'--volume']
    results={}
    for name,env in [('control',parent),('rebuilt',cfg)]:
        proc=subprocess.run(invocation,capture_output=True,text=True,env={**os.environ,**env})
        (output/(name+'.log')).write_text(proc.stdout+proc.stderr)
        if proc.returncode:raise ValueError(name+' '+proc.stderr[-1800:])
        results[name]=json.loads(proc.stdout)
    src=Path(__file__).parent
    results.update(command=invocation,build=cmd,binary=sha(output/'verify-volume'),driver=sha(src/'verify_registered_surfaces.cpp'),
        verifier=sha(src/'verify_volume_bindings.hpp'),pose=sha(pose),launch=sha(launch),settings={k:v for k,v in cfg.items() if parent.get(k)!=v})
    (output/'native-acceptance.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
