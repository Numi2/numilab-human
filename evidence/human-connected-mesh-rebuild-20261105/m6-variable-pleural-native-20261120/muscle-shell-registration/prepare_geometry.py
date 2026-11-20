"""Extract paired source and exact native poses without duplicating large visual packs."""
from collections import Counter
import json
import mmap
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

here=Path(__file__).resolve().parent
w=here.parent
sys.path[:0]=[str(w/'resting-clearance-and-atrium-006'),
              str(w/'resting-surface-clearance-001/src')]
from register_tissue_bellies import (READER, SKIN, CASES, source, blend, hash_file)

v=w/'resting-clearance-and-atrium-017'
r=v/'floor-native-live-parent-026'
launch=json.loads((r/'launch.json').read_text())
argv=launch['argv']
asset=Path(argv[argv.index('--soft-tissue-payload')+1])
manifest=json.loads(asset.with_suffix('.manifest.json').read_text())
asset_hash=hash_file(asset)
assert asset_hash==manifest['payload']['sha256']
latest=(r/'native/accepted-geometry/step-16.mrvpack',r/'exact-skin-surface.json')
cases=(latest,CASES[-1])
eligible = {row['stable_id'] for pose,report in cases for row in
            json.loads(report.read_text())['surfaces'] if row['semantic']==51005 and row['count']}
typed=source(asset)
assert eligible and eligible<=typed.keys()
manifest_by_id={row['stable_id']:row for row in manifest['source']['surfaces']}
arrays={}
report=[]
for pose,(path,expected) in enumerate(cases):
    receipt=json.loads(path.with_suffix('.receipt.json').read_text())
    counts=json.loads(expected.read_text())
    baseline={(row['semantic'],row['stable_id']):row for row in counts['surfaces']}
    bodies={row['body_index']:row for row in receipt['accepted_body_poses']}
    compressed=path.with_suffix('.mrvpack.zst').resolve(strict=True)
    by_id={}
    with tempfile.TemporaryFile(dir=here) as f:
        subprocess.run(['/opt/homebrew/bin/zstd','-qdc',compressed],stdout=f,check=True)
        f.seek(0)
        import hashlib
        assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['pack_file_sha256']==counts['accepted_pack_sha256']
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as m:
            for key,i,world,faces in READER['primitives'](m,bodies):
                if key==SKIN:
                    arrays[f'pose{pose}_skin_v']=world
                    arrays[f'pose{pose}_skin_f']=faces.astype(np.uint32)
                    continue
                if key[0]==51004:
                    arrays[f'pose{pose}_bone{key[1]}_v']=world
                    arrays[f'pose{pose}_bone{key[1]}_f']=faces.astype(np.uint32)
                    continue
                if key[0]!=51005 or key[1] not in eligible:
                    continue
                sid=key[1]
                owner=typed[sid]
                assert np.array_equal(owner['source_faces'],faces), (pose,sid)
                registered,jacobian=blend(owner,bodies)
                error=np.linalg.norm(world.astype(float)-registered,axis=1)
                determinant=np.linalg.det(jacobian)
                hits=baseline[key]['triangle_pairs']
                arrays[f'pose{pose}_b{sid}_v']=world
                arrays[f'pose{pose}_b{sid}_j']=jacobian.astype(np.float32)
                arrays[f'pose{pose}_b{sid}_hits']=np.asarray(hits,dtype=np.uint32).reshape(-1,2)
                by_id[sid]=dict(max_registration_error_m=float(error.max()),
                    p95_registration_error_m=float(np.quantile(error,.95)),
                    minimum_blend_determinant=float(determinant.min()),
                    triangles=len(faces),pairs=len(hits),
                    label=manifest_by_id[sid].get('label') or manifest_by_id[sid]['source_name'])
    assert by_id.keys()==eligible
    report.append(dict(pose=pose,step=receipt['accepted_step'],time_s=receipt['accepted_time_s'],
        pack_sha256=receipt['pack_file_sha256'],banks=by_id))
    print(pose,receipt['accepted_step'],len(by_id),Counter(
        row['max_registration_error_m']>0.00012 for row in by_id.values()),flush=True)
for sid in sorted(eligible):
    owner=typed[sid]
    arrays[f'b{sid}_v']=owner['points'].astype(np.float32)
    arrays[f'b{sid}_f']=owner['source_faces'].astype(np.uint32)
    arrays[f'b{sid}_rows']=owner['row_ids'].astype(np.uint32)
np.savez_compressed(here/'paired-geometry.npz',**arrays)
(here/'registration-observation.json').write_text(json.dumps(dict(
    source_path=str(asset),source_sha256=asset_hash,poses=report),indent=2)+'\n')
