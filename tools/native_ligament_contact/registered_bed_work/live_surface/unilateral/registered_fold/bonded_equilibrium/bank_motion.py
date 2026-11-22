"""Separate template deformation from body pose using paired native mesh buffers."""
import collections, hashlib, json, runpy, subprocess, sys
from pathlib import Path
import numpy as np
from material_context import sha
from opening import read_opening

def compare(case,run,reader):
    r=json.loads((run/'terminal.receipt.json').read_text());a=json.loads((run/'admission.json').read_text())
    o=json.loads((case/'opening.json').read_text())
    if r['physical_endpoint']!='accepted' or r['base_pack_file_sha256']!=o['pack'] or sha(run/'accepted-terminal.mrvpack.zst')!=a['terminal_archive_sha256']:
        raise ValueError('Incompatible opening or terminal pack')
    sys.path.insert(0,str(reader.parent/'src'));get=runpy.run_path(str(reader))['primitives']
    terminal=subprocess.check_output(['zstd','-dq','-c',str(run/'accepted-terminal.mrvpack.zst')])
    if hashlib.sha256(terminal).hexdigest()!=r['pack_file_sha256']:raise ValueError('Changed accepted mesh')
    poses={s['body_index']:s for s in r['accepted_body_poses']}
    wrong=set(); crossed=set()
    for l in (run/'envelope/target-witnesses.jsonl').read_text().splitlines():
        d=json.loads(l);crossed.add((d['semantic'],d['stable_id'],d['primitive']))
    results=[]
    with read_opening(case,o['pack']) as raw:
        for u,v in zip(get(raw,poses),get(terminal,poses),strict=True):
            k,idx,x,t=u;kk,idx2,y,tt=v
            if (k,idx)!=(kk,idx2) or not np.array_equal(t,tt) or x.shape!=y.shape:raise ValueError('Changed anatomical identity or topology')
            dist=np.linalg.norm(x.astype(float)-y.astype(float),axis=1)
            results.append(dict(semantic=k[0],stable=k[1],primitive=idx,maximum_m=float(dist.max()),rms_m=float(np.sqrt((dist**2).mean())),crossing=(*k,idx) in crossed))
    roles=collections.defaultdict(list)
    for v in results:roles[v['semantic']].append(v)
    return dict(source_pose='both buffers transformed with identical native terminal body matrices',opening_sha256=o['pack'],terminal_sha256=r['pack_file_sha256'],
                by_semantic={k:dict(banks=len(v),changed_above_2um=sum(r['maximum_m']>2e-6 for r in v),peak_m=max(r['maximum_m'] for r in v),crossing_banks_above_2um=sum(r['maximum_m']>2e-6 and r['crossing'] for r in v)) for k,v in roles.items()},
                affected_crossing_banks=[r for r in results if r['crossing'] and r['maximum_m']>2e-6])
if __name__=='__main__':print(json.dumps(compare(*(Path(k).resolve() for k in sys.argv[1:])),indent=2))
