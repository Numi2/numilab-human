"""Resolve the accepted skin/tissue pairs into their authored tissue identities."""
import json
from collections import Counter
from pathlib import Path
import sys
from material_context import sha

def classify(case):
    def read(p):return json.loads(p.read_text())
    launch=read(case/'launch.json')
    arg=launch['argv']
    p=Path(arg[arg.index('--soft-tissue-payload')+1])
    m=p.with_suffix('.manifest.json')
    manifest=read(m)
    if sha(p)!=launch['pins'][str(p)] or sha(p)!=manifest['payload']['sha256']:
        raise ValueError('Anatomical inventory differs from the launched asset')
    source=manifest['source']['surfaces']
    ids={r['stable_id']:r for r in source}
    if len(ids)!=len(source):raise ValueError('Duplicate soft tissue identities')
    e=case/'envelope'
    audit=read(e/'physical-banks.json')
    if audit['pack_sha256']!=read(case/'terminal.receipt.json')['pack_file_sha256']:
        raise ValueError('Unmatched frame')
    types={51005:'muscle',51006:'tendon',51011:'vessel'}
    by_role=Counter();banks=[]
    for line in (e/'target-witnesses.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['ocular_source_id']:
            continue
        tag=types.get(r['semantic'])
        if tag is None:continue
        pts=r['triangle_pairs']
        if len(pts)!=r['count'] or len({tuple(pair) for pair in pts})!=len(pts):raise ValueError('Crossing witnesses differ')
        item=dict(semantic=r['semantic'],stable_id=r['stable_id'],role=tag,count=len(pts))
        if tag in ('muscle','tendon'):
            entry=ids[r['stable_id']]
            if any(j<0 or j>=entry['triangle_count'] for _,j in pts):raise ValueError('Mismatched source face')
            item.update(label=entry.get('label') or entry['source_name'],source_triangles=entry['triangle_count'])
        by_role[tag]+=len(pts);banks.append(item)
    if sum(by_role.values())!=audit['physical_nonocular_pairs']:
        raise ValueError('Unclassified physical crossing')
    return dict(frame=audit['pack_sha256'],manifest=sha(m),witnesses=sha(e/'target-witnesses.jsonl'),
                totals=by_role,banks=sorted(banks,key=lambda x:(-x['count'],x['semantic'],x['stable_id'])))
if __name__=='__main__':print(json.dumps(classify(Path(sys.argv[1])),indent=2))
