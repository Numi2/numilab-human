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
    trace=rows[:accepted];net=[np.linalg.norm([float(r['net_f'+a+'_N']) for a in 'xyz']) for r in trace]
    stat=json.loads((run/'solid-overlap.json').read_text());assert stat['accepted_step']==accepted and stat['snapshot_sha256']==sha(run/'accepted-pelvic-state.csv')
    parts=[p for z in stat['muscles'].values() for p in z['cortex'].values()]
    before=sum(p['source_bound_ml'] for p in parts);after=sum(p['accepted_material_ml'] for p in parts)
    launch=json.loads((run/'launch.json').read_text())
    args=launch['argv']
    def argument(name,convert):
        if args.count(name)!=1: raise ValueError('Missing or ambiguous native argument '+name)
        return convert(args[args.index(name)+1])
    requested=argument('--muscle-step-count',int)
    dt=argument('--muscle-step-seconds',float)
    if requested<=0 or not np.isfinite(dt) or dt<=0 or not 0<=accepted<=requested:
        raise ValueError('Invalid accepted clock or native step budget')
    if fail.exists() and round(l['brain_accepted_metadata'][1])!=round(1e6*dt*accepted):
        raise ValueError('Brain clock differs from accepted mechanics clock')
    env=launch['environment']
    def switch(name):
        value=env.get(name)
        return {'0':False,'1':True}.get(value)
    execution=json.loads((run/'execution.json').read_text())
    duration=accepted*dt
    result=dict(accepted_steps=accepted,requested_steps=requested,step_seconds=dt,
                accepted_seconds=duration,wall_seconds=execution['wall_seconds'],
                rtf_including_initialization=duration/execution['wall_seconds'],
                completed_requested_duration=bool(end.exists() and not fail.exists() and
                    execution['native_exit_code']==0 and accepted==requested),
                variable_pleural_pressure=env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE'],
                max_internal_force_imbalance_N=max(net,default=None),
                minimum_accepted_J=min((float(r['min_J']) for r in trace),default=None),
                maximum_accepted_J=max((float(r['max_J']) for r in trace),default=None),
                maximum_accepted_sampled_penetration_m=max((float(r[k]) for r in trace for k in ('gap_r_m','gap_l_m')),default=None),
                endpoint_wrench_source='before dynamics',endpoint_material_stats='after dynamics',
                cortical_volume_before_elasticity_ml=before,cortical_volume_with_elasticity_ml=after,
                probe_fractional_reduction=(1-after/before) if before>0 else None,
                solid_overlap_gate=after==0.)
    result['motion_scope']={
        'local_bed_layers_requested': switch('NUMI_HUMAN_LOCAL_BED_LAYERS'),
        'supported_prelude_steps_requested':env.get('NUMI_HUMAN_SUPPORTED_PRELUDE_STEPS'),
        'whole_body_crossing_audit':'outside this regional audit',
        'continuous_between_step_clearance':'unmeasured',
        'qualification_target_seconds':16.,
        'whole_body_skin_fat_fem_requested':switch('NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM'),
        'passive_fascia_requested':switch('NUMI_HUMAN_RESTING_PASSIVE_FASCIA'),
        'pressure_loaded_aortic_wall_requested':switch('NUMI_HUMAN_RESTING_AORTIC_WALL'),
    }
    log=run/'stdout.log'
    if log.exists():
        with log.open() as lines:
            owners={line.strip() for line in lines if line.startswith('resting_bed_contact_owner=')}
        if len(owners)>1:raise ValueError('Inconsistent bed contact owner')
        if owners:
            roles=dict(word.split('=',1) for word in owners.pop().split())
            result['motion_scope']['observed_bed_contact']=roles
    skin=run/'exact-skin-surface.json'
    if skin.is_file():
        receipt=json.loads(end.read_text())
        audit=json.loads(skin.read_text())
        if (audit['accepted_step']!=accepted or audit['accepted_pack_sha256']!=receipt['pack_file_sha256']
                or audit['accepted_root_fingerprint_hex']!=receipt['accepted_root_fingerprint_hex']
                or audit['accepted_time_s']!=receipt['accepted_time_s']):
            raise ValueError('Whole-skin audit belongs to another accepted pose')
        targets=audit['surfaces']
        keys={(row['semantic'],row['stable_id'],row['primitive']) for row in targets}
        if len(keys)!=len(targets) or not targets:raise ValueError('Missing or duplicate anatomical target identities')
        counts={k:0 for k in audit['crossing_pairs_by_semantic']}
        for row in targets:
            if row['count']!=len(row['triangle_pairs']):raise ValueError('Incomplete skin crossing witnesses')
            counts[str(row['semantic'])]+=row['count']
        if counts!=audit['crossing_pairs_by_semantic']:raise ValueError('Skin audit totals disagree')
        self_pairs=audit['self_intersections']['count']
        if self_pairs!=len(audit['self_intersections']['triangle_pairs']):raise ValueError('Incomplete skin self witnesses')
        nonocular=sum(row['count'] for row in targets if not row['ocular_source_id'])
        result['whole_skin_endpoint']=dict(step=accepted,time_s=audit['accepted_time_s'],pack_sha256=audit['accepted_pack_sha256'],
            audit_sha256=sha(skin),targets=len(keys),self_pairs=self_pairs,nonocular_pairs=nonocular,
            surface_clearance_witnesses_present=bool(self_pairs or nonocular),limits=audit['limit'])
        result['motion_scope']['whole_body_crossing_audit']='outer skin versus registered surface targets at this endpoint'
    crossing=run/'native-crossings.json'
    if crossing.exists():
        c=json.loads(crossing.read_text())
        if c['accepted_step']!=accepted or c['native_surface_sha256']!=sha(run/'accepted-pelvic-surfaces.csv'):
            raise ValueError('Crossing witness belongs to another accepted surface')
        result['motion_scope']['audited_muscle_bone_pairs']=sorted(c['regional'])
        result['native_transverse_pairs']=c['total_transverse']
        result['native_boundary_pairs']=c['total_boundary_only']
        result['regional_endpoint_clearance_passed']=after==0 and c['total_transverse']==0
    endpoint_trace=run/'pelvic-endpoint-crossings.csv'
    if endpoint_trace.exists():
        with endpoint_trace.open() as f: checks=list(csv.DictReader(f))
        if len(checks)<accepted or any(int(r['candidate_step'])!=i for i,r in enumerate(checks)):
            raise ValueError('Missing or out-of-order cortical certificate')
        verified=checks[:accepted]
        if any(int(r['transverse_pairs']) or int(r['nonfinite']) for r in verified):
            raise ValueError('A crossing or nonfinite mesh advanced the accepted clock')
        if not crossing.exists() or (verified and int(verified[-1]['transverse_pairs'])!=c['total_transverse']):
            raise ValueError('Native and independent endpoint predicates disagree')
        result['certified_regional_endpoints']=len(verified)
        result['maximum_accepted_reported_contact_violation_m']=result.pop('maximum_accepted_sampled_penetration_m')
        result['cortical_endpoint_trace_sha256']=sha(endpoint_trace)
        if fail.exists() and len(checks)>accepted:
            result['first_rejected_geometry']={k:int(v) for k,v in checks[accepted].items()}
    volume=run/'constitutive-volume.json'
    solids=run/'native-closed-solids.json'
    if not solids.exists():solids=crossing
    if solids.exists():
        data=json.loads(solids.read_text())
        if data['accepted_step']!=accepted or data['native_surface_sha256']!=sha(run/'accepted-pelvic-surfaces.csv'):
            raise ValueError('Closed-solid audit belongs to another capture')
        if all('emitted_solid' in bank for bank in data['regional'].values()):
            result['emitted_closed_solids_exterior']=all(
                bank['emitted_solid']['closed_and_exterior'] and bank['intersecting_pairs']==0
                for bank in data['regional'].values())
    if volume.exists():
        data=json.loads(volume.read_text())
        if (data['accepted_step']!=accepted or data['snapshot_sha256']!=sha(run/'accepted-pelvic-state.csv')
            or data['native_surface_sha256']!=sha(run/'accepted-pelvic-surfaces.csv')
            or data['registration_sha256']!=stat['registration_sha256']):
            raise ValueError('Constitutive-volume evidence belongs to another state')
        measured=sum(bank['accepted_material_ml'] for side in data['muscles'].values() for bank in side['cortex'].values())
        result['constitutive_overlap_ml']=measured
        if 'emitted_closed_solids_exterior' in result:
            result['regional_terminal_all_representations_clear']=bool(
                result['emitted_closed_solids_exterior'] and measured==0 and after==0)
    physical_boundary=run/'constitutive-boundary.json'
    if physical_boundary.exists():
        evidence=json.loads(physical_boundary.read_text())
        if (evidence['accepted_step']!=accepted or evidence['material_capture']!=sha(run/'accepted-pelvic-state.csv')
            or evidence['cortex_capture']!=sha(run/'accepted-pelvic-surfaces.csv')):
            raise ValueError('Constitutive boundary audit belongs to another state')
        result['constitutive_boundary_transverse_pairs']=evidence['total_transverse']
        result['constitutive_boundary_touch_pairs']=sum(b['touch_pairs'] for b in evidence['banks'].values())
        if 'regional_terminal_all_representations_clear' in result:
            result['regional_terminal_all_representations_clear'] &= (evidence['total_transverse']==0)
        if 'constitutive_overlap_ml' in result and 'emitted_closed_solids_exterior' in result:
            result['emitted_and_constitutive_terminal_clear']=bool(result['emitted_closed_solids_exterior']
                and evidence['total_transverse']==0 and result['constitutive_overlap_ml']==0)
    if fail.exists():result['rejecting_candidate']=rows[accepted]
    if end.exists() and ctrl:
        a=json.loads(end.read_text());b=json.loads((ctrl/'terminal.receipt.json').read_text());assert a['accepted_step']==b['accepted_step']
        A={r['body_index']:r for r in a['accepted_body_poses']};B={r['body_index']:r for r in b['accepted_body_poses']};assert set(A)==set(B)
        result['contact_off_comparison']=dict(bodies=len(A),maximum_body_origin_change_m=max(np.linalg.norm(np.array(A[i]['position_m'])-B[i]['position_m']) for i in A),changed_body_sha256=a['accepted_body_state_sha256']!=b['accepted_body_state_sha256'],control_state=b['accepted_body_state_sha256'],candidate_state=a['accepted_body_state_sha256'])
    (run/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');return result
def require_clear_skin_endpoint(result):
    skin=result.get('whole_skin_endpoint')
    if skin is None:
        raise ValueError('Required whole-skin endpoint audit is missing')
    if skin['self_pairs'] or skin['nonocular_pairs']:
        raise ValueError(f"Registered outer skin at step {skin['step']} has {skin['self_pairs']} self pairs and {skin['nonocular_pairs']} nonocular pairs")

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('control',type=Path,nargs='?')
    parser.add_argument('--require-clear-skin-endpoint',action='store_true',
        help='Require an authenticated outer-skin endpoint with zero self and nonocular intersections.')
    args=parser.parse_args()
    result=report(args.run,args.control)
    if args.require_clear_skin_endpoint:
        try:require_clear_skin_endpoint(result)
        except ValueError as exc:parser.exit(1,str(exc)+'\n')
    print(result)
