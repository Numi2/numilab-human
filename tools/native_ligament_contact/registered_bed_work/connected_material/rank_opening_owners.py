"""Bind opening crossing witnesses to the loaded soft-tissue surface and actuator identities."""
import json
from pathlib import Path
from run_bed_work import digest


def rank(root):
    cfg=json.loads((root/'pose-one/launch.json').read_text());args=cfg['argv']
    payload=Path(args[args.index('--soft-tissue-payload')+1]);manifest=payload.with_suffix('.manifest.json')
    m=json.loads(manifest.read_text())
    if m['payload']['sha256']!=digest(payload) or cfg['pins'][str(manifest)]!=digest(manifest):
        raise ValueError('Unmatched mechanical provenance')
    catalog={s['stable_id']:s for s in m['source']['surfaces'] if s['layer']=='muscle'}
    evidence={}
    for n in ('native_opening','dense_reference','dense_loaded'):
        d=root/'material-source/geometry'/n;records=[json.loads(x) for x in (d/'target-witnesses.jsonl').read_text().splitlines()]
        proof=json.loads((d/'geometry.json').read_text())
        by_bank={}
        for r in records:
            if r['semantic']!=51005:continue
            sid=r['stable_id'];pairs=r['triangle_pairs']
            if sid in by_bank or sid not in catalog:raise ValueError('Ambiguous source muscle')
            source=catalog[sid];routes=source.get('matched_muscles',[])
            by_bank[sid]=dict(id=sid,primitive=r['primitive'],source_label=source.get('label'),member=source.get('member'),
                              member_sha256=source.get('member_sha256'),source_actuators=[{'name':s['name'],'index':s['source_actuator_index']} for s in routes],
                              surface_pairs=len(pairs),distinct_target_faces=len({q[1] for q in pairs}))
        if sum(r['surface_pairs'] for r in by_bank.values())!=proof['intersection_pairs_by_semantic']['51005']:
            raise ValueError('Incomplete muscle witnesses')
        evidence[n]=sorted(by_bank.values(),key=lambda r:(-r['surface_pairs'],r['id']))
    return dict(payload=digest(payload),manifest=digest(manifest),banks=evidence,
                scope='Exact triangle witnesses mapped to NHTISS4 source banks and recorded MyoSim actuator IDs. Empty actuator lists retain unresolved ownership.')
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    (root/'material-source/muscle-witnesses.json').write_text(json.dumps(rank(root),indent=2,allow_nan=False)+'\n')
