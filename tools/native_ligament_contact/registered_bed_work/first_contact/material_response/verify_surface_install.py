"""Read the updated surface and parent rows using the production native registration owner."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from register_material_surfaces import PREFIX,sha,pinned


def run(launch, snapshot, material, layers, parents, out):
    base=json.loads(launch.read_bytes())['environment']
    build=json.loads((material/'native-import-build.json').read_bytes())['command']
    inc=Path(build[build.index('-I')+1])
    if not (inc/'NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp').is_file():
        raise ValueError('Native build no longer uses the recorded surface header directory')
    src=Path(__file__).with_name('verify_registered_surfaces.cpp').resolve()
    out.mkdir()
    exe=out/'verify-registration'
    args=list(build)
    args[args.index(str(material/'verify-native-material'))]=str(exe)
    args[next(i for i,a in enumerate(args) if a.endswith('/verify_native_material.cpp'))]=str(src)
    subprocess.run(args,check=True)
    pose=json.loads(snapshot.read_bytes())
    if len(pose['bodies'])!=pose['binding_count'] or pose['binding_count']!=86:raise ValueError('Incomplete native pose')
    with (out/'native-frames.txt').open('w') as f:
        for i,b in enumerate(pose['bodies']):
            values=[]
            for key in ('current','rest'):
                for field in ('position_world_m','local_to_world_row_major_f32'):
                    values.extend(np.asarray(b[key][field],np.float32).astype('f8'))
            print(i,b['body_index'],*(format(v,'.17g') for v in values),file=f)
    update=dict(base)
    for k in 'OID':
        pinned(base,'ROLE_'+k+'_SOURCE_ASSET','ROLE_'+k+'_SOURCE_ASSET_SHA256')
        p=layers/('registered-'+k+'.csv')
        update[PREFIX+'SURFACE_'+k+'_NODES_CSV']=str(p)
        update[PREFIX+'SURFACE_'+k+'_NODES_SHA256']=sha(p)
    p=parents/'node-parent-registration.csv'
    update[PREFIX+'NODE_REGISTRATION_CSV']=str(p)
    update[PREFIX+'NODE_REGISTRATION_SHA256']=sha(p)
    receipts={}
    cmd=[str(exe),str(out/'native-frames.txt'),str(len(pose['all_bodies']))]
    for name,env in [('control',base),('rebuilt',update)]:
        result=subprocess.run(cmd,env={**os.environ,**env},capture_output=True,text=True)
        (out/(name+'.log')).write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError(name+' '+result.stderr)
        receipts[name]=json.loads(result.stdout)
    receipts.update(build=args,command=cmd,source=sha(src),owner_header=sha(inc/'NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp'),
        executable=sha(exe),actual_pose=sha(snapshot),parent_launch=sha(launch),changed_settings={k:v for k,v in update.items() if v!=base.get(k)})
    (out/'native-acceptance.json').write_text(json.dumps(receipts,indent=2)+'\n')


if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
