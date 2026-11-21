"""Read native acceptance and the independent material/cortex audit at the same step."""
import csv,json,sys
from pathlib import Path
import numpy as np
from material_context import sha

def report(run,ctrl):
    rows=list(csv.DictReader((run/'candidate-contact.csv').open()))
    with (run/'accepted-pelvic-state.csv').open() as f:accepted=int(next(csv.DictReader(f))['accepted_step'])
    end=run/'terminal.receipt.json'
    if end.is_file():assert json.loads(end.read_text())['accepted_step']==accepted
    fail=run/'failure.json'
    if fail.exists():
        f=json.loads(fail.read_text());l=f['first_failure_latch']
        assert accepted==l['step']==l['respiration_accepted_status'][0]
        assert l['brain_accepted_metadata'][1]==2000*accepted
    trace=rows[:accepted];net=[np.linalg.norm([float(r['net_f'+a+'_N']) for a in 'xyz']) for r in trace]
    stat=json.loads((run/'solid-overlap.json').read_text());assert stat['accepted_step']==accepted and stat['snapshot_sha256']==sha(run/'accepted-pelvic-state.csv')
    parts=[p for z in stat['muscles'].values() for p in z['cortex'].values()]
    before=sum(p['source_bound_ml'] for p in parts);after=sum(p['accepted_material_ml'] for p in parts)
    execution=json.loads((run/'execution.json').read_text());result=dict(accepted_steps=accepted,accepted_seconds=accepted*.002,wall_seconds=execution['wall_seconds'],rtf_including_initialization=accepted*.002/execution['wall_seconds'],completed_requested_duration=execution['native_exit_code']==0,variable_pleural_pressure=json.loads((run/'launch.json').read_text())['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE'],max_internal_force_imbalance_N=max(net),minimum_accepted_J=min(float(r['min_J']) for r in trace),maximum_accepted_J=max(float(r['max_J']) for r in trace),maximum_accepted_sampled_penetration_m=max(float(r[k]) for r in trace for k in ('gap_r_m','gap_l_m')),endpoint_wrench_source='before dynamics',endpoint_material_stats='after dynamics',cortical_volume_before_elasticity_ml=before,cortical_volume_with_elasticity_ml=after,probe_fractional_reduction=1-after/before,solid_overlap_gate=after==0.)
    if fail.exists():result['rejecting_candidate']=rows[accepted]
    if end.exists() and ctrl:
        a=json.loads(end.read_text());b=json.loads((ctrl/'terminal.receipt.json').read_text());assert a['accepted_step']==b['accepted_step']
        A={r['body_index']:r for r in a['accepted_body_poses']};B={r['body_index']:r for r in b['accepted_body_poses']};assert set(A)==set(B)
        result['contact_off_comparison']=dict(bodies=len(A),maximum_body_origin_change_m=max(np.linalg.norm(np.array(A[i]['position_m'])-B[i]['position_m']) for i in A),changed_body_sha256=a['accepted_body_state_sha256']!=b['accepted_body_state_sha256'],control_state=b['accepted_body_state_sha256'],candidate_state=a['accepted_body_state_sha256'])
    (run/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(report(Path(sys.argv[1]),Path(sys.argv[2]) if len(sys.argv)==3 else None))
