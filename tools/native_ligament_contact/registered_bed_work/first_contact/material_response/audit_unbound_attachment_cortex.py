"""Measure the native rigid cortex available to exterior point-mode tendon endpoints."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import runpy
import struct
import sys
import numpy as np
from recount_registered_crossings import digest, read
from measure_forearm_enclosure import Enclosure


def audit(parent, trial, sites, lib, output):
    report=read(sites)
    pose_path=trial/'native-initial-bodies.json'
    src=trial/'native-traces/resting-human.mrvpack'
    shell=read(parent.parent/'native-release-banks/release-envelope.json')
    if digest(src)!=shell['pack_sha256'] or digest(pose_path)!=report['articulated_pose'] or digest(lib)!=report['enclosure_library']:
        raise ValueError('Mismatched native geometry')
    reader=next(d/'resting-surface-clearance-001/audit.py' for d in parent.parents if (d/'resting-surface-clearance-001/audit.py').is_file())
    sys.path.insert(0,str(reader.parent/'src'))
    api=runpy.run_path(str(reader))
    raw=src.read_bytes()
    h,d=api['clearance']._PACK_HEADER,api['clearance']._PACK_DIRECTORY
    sec={r[0]:r for r in (d.unpack_from(raw,h.size+i*d.size) for i in range(h.unpack_from(raw)[2]))}
    poses={r['body_index']:dict(position_m=np.asarray(r['current']['position_world_m'],dtype='f4').tolist(),
                                quaternion_xyzw=np.asarray(r['current']['orientation_world_xyzw'],dtype='f4').tolist())
           for r in read(pose_path)['all_bodies']}
    rows=[dict(s) for s in report['exceptions'] if s['current']=='outside' and s['role']!='via' and s['attachment']=='source_site_point']
    by=defaultdict(list)
    for i,row in enumerate(rows):
        by[row['body']].append(i)
    region=defaultdict(int)
    for (semantic,stable),idx,v,f in api['primitives'](raw,poses):
        if semantic!=51004:
            continue
        record=struct.unpack_from('<8I',raw,sec[4][2]+64*idx)
        instance=struct.unpack_from('<8f12I',raw,sec[5][2]+80*record[3])
        if (record[4],record[7])!=(semantic,stable) or instance[10]!=3:
            raise ValueError('Rigid cortical identity mismatch')
        body=instance[9]
        if body not in by:
            continue
        region[body]+=1
        surface=Enclosure(lib,v,f)
        try:
            inside,dist=surface.query([rows[i]['world_m'] for i in by[body]])
            for i,lab,x in zip(by[body],inside,dist):
                witness=dict(stable_id=stable,primitive=idx,mm=1000*x,inside_bone=int(lab))
                if 'nearest' not in rows[i] or x*1000<rows[i]['nearest']['mm']:
                    rows[i]['nearest']=witness
        finally:
            surface.close()
    a=read(trial/'launch.json')['argv']
    owner=Path(a[a.index('--tendon-payload')+1])
    if digest(owner)!=report['tendon_payload']:
        raise ValueError('Mismatched tendon owner')
    limits=read(owner.with_suffix('.manifest.json'))['admission']
    gates={k:1000*limits[k] for k in ('maximum_surface_distance_m','maximum_migrated_endpoint_distance_m')}
    for row in rows:
        row['available_cortical_primitives']=region[row['body']]
    result=dict(pose=report['articulated_pose'],mrvpack=digest(src),sites=digest(sites),point_endpoints=len(rows),
                limits_mm=gates,missing_cortex=sum('nearest' not in r for r in rows),
                nearest_within_generic_attachment_distance=int(sum(r.get('nearest',{}).get('mm',np.inf)<=gates['maximum_surface_distance_m'] for r in rows)),
                nearest_within_maximum_migration_distance=int(sum(r.get('nearest',{}).get('mm',np.inf)<=gates['maximum_migrated_endpoint_distance_m'] for r in rows)),
                cases=rows,scope='Distance to the nearest published rigid bone surface owned by the same body. A near surface alone does not identify a muscle insertion.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='cases'})


if __name__=='__main__':
    audit(*(Path(s).resolve() for s in sys.argv[1:]))
