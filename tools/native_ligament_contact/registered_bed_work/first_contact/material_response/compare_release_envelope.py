"""Audit the exported reference surface in the native pose and compare its exact crossings with motion."""
from collections import Counter
import hashlib
import json
import mmap
from pathlib import Path
import runpy
import sys

import numpy as np
from audit_registered_initial_envelope import scan_banks
from measure_forearm_enclosure import Enclosure, measure
from run_cavity_motion import digest

def run(job, reader, lib, out):
    out.mkdir()
    args=json.loads((job/'launch.json').read_text())
    ctl=json.loads((job/'verified-endpoint.json').read_text())
    original=job/'native-traces/resting-human.mrvpack'
    posefile=job/'native-initial-bodies.json'
    info=json.loads(posefile.read_text())
    poses={v['body_index']:dict(position_m=v['current']['position_world_m'],quaternion_xyzw=v['current']['orientation_world_xyzw'])
           for v in info['all_bodies']}
    if sorted(poses)!=list(range(len(poses))):raise ValueError('Incomplete articulated pose')
    audit=runpy.run_path(str(reader))
    provenance={'release_pack_sha256':digest(original),'release_pose_sha256':digest(posefile),
                'native_launch_sha256':digest(job/'launch.json'),'endpoint_sha256':ctl['frame_sha256']}
    env=args['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_'
    points=Path(env[prefix+'NODES_CSV']);faces=Path(env[prefix+'FACES_CSV'])
    if digest(points)!=env[prefix+'NODES_SHA256'] or digest(faces)!=env[prefix+'FACES_SHA256']:raise ValueError('Skin inputs changed')
    xyz=np.loadtxt(points,delimiter=',',skiprows=1,usecols=(2,3,4),dtype=np.float32)
    tri=np.loadtxt(faces,delimiter=',',skiprows=1,usecols=(1,2,3),dtype=np.int32)
    with original.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as raw:
        ((_,_,v,ft),)=audit['primitives'](raw,poses,skin_only=True)
        if not np.array_equal(tri,ft):raise ValueError('Release skin connectivity mismatch')
        diff=np.linalg.norm(v.astype(float)-xyz.astype(float),axis=1).max()
        if diff>2e-7:raise ValueError(f'Release skin registration mismatch: {diff}')
        prov=dict(provenance,max_release_registration_error_m=float(diff))
        scan_banks(audit,raw,poses,v,ft,prov,out,'release-envelope.json')
        manifest=Path(args['argv'][args['argv'].index('--soft-tissue-payload')+1]).with_suffix('.manifest.json')
        names={r['stable_id']:r for r in json.loads(manifest.read_bytes())['source']['surfaces']}
        skin=Enclosure(lib,v,ft);rows=[]
        try:
            for key,_,x,fc in audit['primitives'](raw,poses):
                if key[0] not in (51005,51006):continue
                m=names[key[1]]
                area=measure(skin,x,fc)
                rows.append(dict(semantic=key[0],stable_id=key[1],label=m.get('label',m.get('member_id')),**area))
        finally:skin.close()
    def physical(p):
        s=set()
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if r['ocular_source_id'] or r['semantic']==51010:continue
            for a,b in r['triangle_pairs']:s.add((r['semantic'],r['stable_id'],a,b))
        return s
    first=physical(out/'target-witnesses.jsonl');last=physical(job/'envelope/target-witnesses.jsonl')
    if len(last)!=ctl['physical_skin_tissue_pairs']:raise ValueError('Unmatched motion classification')
    result=dict(**provenance,physical_release=len(first),physical_endpoint=len(last),identical_pairs=len(first&last),
                release_only=len(first-last),endpoint_only=len(last-first),release_by_semantic=dict(Counter(r[0] for r in first)),
                exterior_muscle_tendon=rows)
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({k:v for k,v in result.items() if k not in ['exterior_muscle_tendon']},flush=True)

if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
