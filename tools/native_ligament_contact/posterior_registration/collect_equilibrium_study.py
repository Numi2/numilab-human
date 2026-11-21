"""Compare physics at the first rejected native endpoint and its preceding state."""
import csv
import json
import sys
from pathlib import Path
from material_context import sha

def record(path):
    q=json.loads((path/'qualification.json').read_text())
    result={key:q[key] for key in (
        'accepted_steps','requested_steps','accepted_seconds','rtf_including_initialization',
        'completed_requested_duration','native_transverse_pairs','cortical_volume_with_elasticity_ml',
        'minimum_accepted_J','maximum_accepted_J','motion_scope')}
    result['material_state_sha256']=sha(path/'accepted-pelvic-state.csv')
    result['surface_sha256']=sha(path/'accepted-pelvic-surfaces.csv')
    if q.get('first_rejected_geometry') is not None:
        geometry=q['first_rejected_geometry']
        if geometry['candidate_step']!=q['accepted_steps']:
            raise ValueError('Geometry and accepted clock disagree')
        with (path/'fresh-force-certificate.csv').open() as f:
            last=[r for r in csv.DictReader(f) if int(r['candidate_step'])==geometry['candidate_step'] and int(r['endpoint'])==1]
        if len(last)!=2:raise ValueError('Incomplete bilateral force check')
        witnesses={}
        for r in last:
            side={'0':'right','1':'left'}[r['side']]
            external=float(r['external_force_N'])
            limit=.01*max(external,1e-6)
            witnesses[side]={'residual_N':float(r['free_residual_N']),'external_N':external,
                'limit_N':limit,'residual_over_limit':float(r['free_residual_N'])/limit,
                'penetration_m':float(r['penetration_m'])}
        result['first_rejection']={'candidate_step':geometry['candidate_step'],
            'exact_transverse_pairs':geometry['transverse_pairs'],'sides':witnesses}
    return result

if __name__=='__main__':
    folders=list(map(Path,sys.argv[1:-1]))
    Path(sys.argv[-1]).write_text(json.dumps({p.name:record(p) for p in folders},indent=2)+'\n')
