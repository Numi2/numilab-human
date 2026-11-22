"""Compare accepted distal MyoSim force sites and skin, using their actual rigid-body owners."""
import json
import sys
from pathlib import Path

import numpy as np
from cortical_distance import CorticalSurface
from digital_clearance import complete_remote_loops
from material_context import read_inputs,applied,sha

def measure(native,manifest,lib,out):
    j=json.loads((native/'launch.json').read_text());rc=json.loads((native/'terminal.receipt.json').read_text())
    h=json.loads((native/'verified-endpoint.json').read_text())
    if (rc['physical_endpoint']!='accepted' or rc['accepted_step']!=h['accepted_steps'] or h['skin_self_intersections']!=0
            or rc['pack_file_sha256']!=h['frame_sha256'] or sha(native/'physical-accepted.node')!=h['physical_nodes_sha256']):raise ValueError('Unmatched material and articulated motion')
    poses=sorted(rc['accepted_body_poses'],key=lambda r:r['body_index'])
    if [r['body_index'] for r in poses]!=list(range(len(poses))):
        raise ValueError('Missing anatomical body transform')
    matrices=np.array([r['position_m']+r['quaternion_xyzw'] for r in poses])
    routes,users,_,_,_=read_inputs(native/'launch.json',manifest,None,accepted_poses=matrices)
    env=j['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def pinned(a,b):
        p=Path(env[prefix+a])
        if sha(p)!=env[prefix+b]:raise ValueError('Unknown surface mapping')
        return p
    from csv import DictReader
    face=pinned('SURFACE_O_FACES_CSV','SURFACE_O_FACES_SHA256');mp=pinned('COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256')
    f=np.loadtxt(face,skiprows=1,usecols=(1,2,3),delimiter=',',dtype=int)
    lookup={}
    with mp.open() as stream:
        for r in DictReader(stream):
            if r['registered_surface_layer']=='O':
                i=int(r['registered_surface_vertex_row'])
                if i in lookup or r['source_partition']!='skin':raise ValueError('Ambiguous skin mapping')
                lookup[i]=int(r['joint_node_id'])-1
    v=np.loadtxt(native/'physical-accepted.node',skiprows=1,usecols=(1,2,3))[np.array([lookup[i] for i in range(len(lookup))])]
    sites=[(name,s) for name,r in sorted(routes.items()) for s in r[-6:]]
    pts=np.array([s['world'] for name,s in sites])
    V,F,gap=complete_remote_loops(v,f,pts)
    model=CorticalSurface(lib,V,F);d,n,near,idx=model.query(pts)
    if np.any(idx>=len(f)) or not np.allclose(np.linalg.norm(pts-near,axis=1),abs(d),rtol=2e-7,atol=1e-10):raise ValueError('Nonphysical closest surface')
    evidence=[dict(muscle=name,site=int(s['site']),body=int(s['body']),resolved_attachment=bool(s['endpoint']),
         shared_site_users=users[s['site']],signed_mm=float(w*1000),world_m=s['world'].tolist(),closest_skin_face=int(i)) for (name,s),w,i in zip(sites,d,idx)]
    footprints=[]
    for name,route in sorted(routes.items()):
        s=route[-1];ep=s['endpoint'];surf=ep['surface'];body=s['body']
        if ep['attachment_mode']!='registered_bone_distributed_envelope':raise ValueError('Unknown cortical support')
        points=applied(matrices[body],np.array(surf['node_local_points_m']))
        ds,_,nearest,faces=model.query(points)
        if (faces>=len(f)).any() or np.max(abs(np.linalg.norm(points-nearest,axis=1)-abs(ds)))>1e-9:raise ValueError('Nonphysical load footprint')
        footprints.append(dict(muscle=name,body=body,bone_stable_id=surf['bone_stable_id'],bone_member_id=surf['bone_member_id'],
                 cortical_vertex_indices=surf['node_vertex_indices'],force_reference_to_cortex_mm=1000*surf['surface_distance_m'],
                 cortical_support_signed_mm=(1000*ds).tolist(),support_method=surf['surface_patch_method']))
    out.write_text(json.dumps(dict(physical_endpoint=h['frame_sha256'],physical_nodes_sha256=h['physical_nodes_sha256'],launch_sha256=sha(native/'launch.json'),
        manifest_sha256=sha(manifest),compiled_muscle_sha256=sha(Path(j['argv'][2])),tendon_sha256=sha(Path(j['argv'][j['argv'].index('--tendon-payload')+1])),
        head_opening_separation_m=gap,scope='Six distal source sites on each of eight digital extensor routes. Positive distances are external to the physical outer skin at this accepted endpoint.',rows=evidence,cortical_footprints=footprints),indent=2)+'\n')
if __name__=='__main__':measure(*(Path(s).resolve() for s in sys.argv[1:]))
