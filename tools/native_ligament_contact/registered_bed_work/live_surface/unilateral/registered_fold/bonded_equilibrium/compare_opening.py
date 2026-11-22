"""Distinguish pre-existing anatomical crossings from crossings introduced during release."""
import collections
import json
from pathlib import Path
import sys
from material_context import sha

def read(p):return json.loads(p.read_text())

def compare(case,endpoint):
    o=read(case/'opening.json');v=read(case/'executed-parity.json')
    a=read(endpoint/'terminal.receipt.json');b=read(endpoint/'envelope/physical-banks.json')
    if (a['base_pack_file_sha256']!=o['pack'] or sha(endpoint/'registered-initial-bodies.json')!=o['poses']
            or v['pack_sha256']!=o['pack'] or v['max_opening_skin_error_m']>2e-6 or v.get('pose_source')!='executed'):
        raise ValueError('Opening geometry and terminal transaction do not match')
    eyes=read(endpoint/'native-traces/resting-ocular-sclera-window.json')
    excluded={(r['derived_skin_identity']['semantic'],r['derived_skin_identity']['stable_id']) for r in eyes['windows']+eyes['posterior_lens_backings']}
    def pairs(d):
        banks=read(d/'physical-banks.json');result=set()
        for l in (d/'target-witnesses.jsonl').read_text().splitlines():
            r=json.loads(l); sem,stable=r['semantic'],r['stable_id']
            if (sem,stable) in excluded or (sem==51010 and 381<=stable<=397):continue
            if len(r['triangle_pairs'])!=r['count']:raise ValueError('Incomplete crossing witnesses')
            result.update((sem,stable,r['primitive'],i,j) for i,j in r['triangle_pairs'])
        if len(result)!=banks['physical_nonocular_pairs'] or banks['checked_banks']!=878:raise ValueError('Incomplete anatomical coverage')
        return result
    prefix=case/'executed-envelope';early=pairs(prefix);late=pairs(endpoint/'envelope')
    if b['pack_sha256']!=a['pack_file_sha256'] or read(prefix/'physical-banks.json')['pack_sha256']!=o['pack']:
        raise ValueError('Unmatched frame identities')
    return dict(opening=len(early),terminal=len(late),persisted=len(early&late),
                cleared=len(early-late),new=len(late-early),opening_by_semantic=dict(collections.Counter(k[0] for k in early)),
                opening_sha256=o['pack'],terminal_sha256=a['pack_file_sha256'],accepted_s=a['accepted_time_s'],
                opening_pairs_sha256=sha(prefix/'target-witnesses.jsonl'),terminal_pairs_sha256=sha(endpoint/'envelope/target-witnesses.jsonl'))
if __name__=='__main__':print(json.dumps(compare(*(Path(x).resolve() for x in sys.argv[1:])),indent=2))
