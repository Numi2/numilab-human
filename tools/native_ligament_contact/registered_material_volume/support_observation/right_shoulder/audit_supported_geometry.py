"""Apply the retained exact Float32 skin audit to every native geometry receipt of a run."""
import json
import runpy
import sys
from pathlib import Path

from compile_connected_joint import digest


def audit(source, adapter, case, output):
    owner = runpy.run_path(str(source))['run']
    runpy.run_path(str(adapter))['install_for_float32']()
    output.mkdir()
    config = json.loads((case/'launch.json').read_bytes())
    captures = sorted((case/'native-traces/accepted-geometry').glob('*.mrvpack'))
    if not captures or json.loads((case/'execution.json').read_bytes())['exit'] != 0:
        raise ValueError('Missing completed scene')
    frames = []
    for pack in captures:
        target = output/(pack.stem+'.json')
        owner(pack.resolve(), target)
        doc = json.loads(target.read_bytes())
        if doc['accepted_pack_sha256'] != digest(pack):
            raise ValueError('Unmatched pack')
        ocular = sum(r['count'] for r in doc['surfaces'] if r['ocular_source_id'])
        total = sum(doc['crossing_pairs_by_semantic'].values())
        rows = sorted((r for r in doc['surfaces'] if r['count']),key=lambda r:-r['count'])
        top = [dict(semantic=r['semantic'], stable=r['stable_id'], pairs=r['count']) for r in rows[:10]]
        frames.append(dict(step=doc['accepted_step'], time_s=doc['accepted_time_s'], pack=digest(pack), report=digest(target),
                           skin_self=doc['self_intersections']['count'], target_banks=len(doc['surfaces']),
                           touched_banks=len(rows), all_pairs=total, ocular_pairs=ocular, nonocular_pairs=total-ocular,
                           by_semantic=doc['crossing_pairs_by_semantic'], largest=top))
    result = dict(launch=digest(case/'launch.json'), regional_registration=config['registered_shoulder'],
                  auditor=digest(source), adapter=digest(adapter), frames=frames,
                  scope='Exact emitted Float32 triangle pairs. Surface intersection tests at the retained native poses. No inter-sample or solid-containment claim.')
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    audit(*(Path(p).resolve() for p in sys.argv[1:]))
