"""Measure initialization and motion with traction iterated at the same anatomical opening."""
import csv
import json
from pathlib import Path
import numpy as np
from run_bed_work import digest


def compare(root):
    def read(p):return json.loads((root/p).read_text())
    def rows(p):
        with (root/p).open() as f:return list(csv.DictReader(f))
    rows_out=[];prior=None;common_q=None;common_v=None;registered_pose=None
    reference_cell=read('material-source/corrected/connected-columns.json')
    before_cell=read('material-source/control/connected-columns.json')
    later_cell=read('material-source/later/connected-columns.json')
    if reference_cell['paths']!=later_cell['paths'] or reference_cell['paths']!=read('installed-assigned/assignment.json')['source']:
        raise ValueError('Fitted columns depend on the wrong datum')
    for c,one,final,attempt in (('normal','shear/control-one','registered-79','recruited-16'),('one shear estimate','shear/loaded/one','shear/loaded/matched-79','shear/loaded/fullaudit'),
                                ('updated shear estimate','shear/refined/one','shear/refined/matched-79','shear/refined/run-16')):
        text=(root/one/'stdout.log').read_text().splitlines()
        state,=[json.loads(s.split('=',1)[1]) for s in text if s.startswith('stand_terminal_state=')]
        init,=[dict(t.split('=',1) for t in s.split()) for s in text if s.startswith('resting_initialization=dynamic_release')]
        end=read(final+'/terminal.receipt.json');capture=read(final+'/envelope/accepted-envelope.json')
        classification=read(final+'/envelope/physical-banks.json')
        if (end['accepted_step']!=79 or end['pack_file_sha256']!=capture['pack_sha256'] or classification['checked_banks']!=878 or
            classification['audit_sha256']!=digest(root/final/'envelope/accepted-envelope.json') or state['step_count']!=1):raise ValueError('Unmatched capture')
        if common_q is None:
            common_q,common_v=state['initial_q'],state['initial_v']
            registered_pose=end['initial_anatomical_registration']['body_poses']
        if common_q!=state['initial_q'] or common_v!=state['initial_v'] or end['initial_anatomical_registration']['body_poses']!=registered_pose:
            raise ValueError('Different initial conditions')
        coupled=rows(final+'/resting-coupled.csv')
        if any(float(r[k])!=0 for r in coupled for k in ('root_assistance_n','root_assistance_nm','body_mass_residual_kg')):
            raise ValueError('External aid or mass imbalance')
        dt=state['timestep_seconds']
        acc=(np.asarray(state['v'])-state['initial_v'])/dt
        impulses=sorted(rows(one+'/resting-com-support-impulses.csv'),key=lambda r:int(r['contact_index']))
        tangents=np.array([[float(r['impulse_world_'+k+'_ns'])/dt for k in 'xy'] for r in impulses])
        diff=None if prior is None else np.linalg.norm(tangents-prior)/np.linalg.norm(prior)
        prior=tangents
        motion=read(final+'/shoulder-motion.json')
        failed=read(attempt+'-source/witness.json');eq=read(attempt+'/failure.json')
        if (read(attempt+'/execution.json')['requested_steps']!=8000 or not failed['invalid_columns_observed'] or
            failed['material_step_index']!=eq['respiratory_accepted_status'][0]):raise ValueError('Different rejected transaction')
        rows_out.append(dict(recruitment=c,initial_max_root_acceleration_m_s2=float(max(abs(acc[:3]))),initial_joint_rms_rad_s2=float(np.sqrt(np.mean(acc[6:]**2))),
            initial_joint_peak_rad_s2=float(max(abs(acc[6:]))),static_residual_rms=float(init['normalized_residual_rms']),static_balanced=(init['balanced']=='true'),
            local_shear_sum_N=float(np.linalg.norm(tangents,axis=1).sum()),net_xy_reaction_N=tangents.sum(0).tolist(),
            relative_change_of_local_shear=diff,native_accepted_steps_before_failure=failed['material_step_index'],failed_vertex=failed['vertex'],
            excess_at_rejection_m=failed['depth_m']-failed['allowed_depth_m'],matched_time_s=end['accepted_time_s'],matched_pack=end['pack_file_sha256'],
            matched_skin_self=capture['skin_self_intersections'],matched_crossings=classification['physical_nonocular_pairs'],
            matched_bone_crossings=classification['bone_pairs'],matched_pelvis_dz_m=motion['pelvis_translation_m'],matched_shoulder_dz_m=motion['total_z_m']))
    trial=read('shear/refined/accepted-82/terminal.receipt.json')
    if trial['accepted_step']!=rows_out[-1]['native_accepted_steps_before_failure']:raise ValueError('Rejection disagrees with available prefix')
    surfaces={}
    for region in ('native_opening','dense_reference','dense_loaded'):
        p='material-source/geometry/'+region+'/'
        g=read(p+'geometry.json');d=read(p+'physical-banks.json')
        if d['audit_sha256']!=digest(root/p/'geometry.json') or d['checked_banks']!=878:raise ValueError('Unqualified source boundary')
        surfaces[region]=dict(nonocular_crossings=d['physical_nonocular_pairs'],skin_self=g['skin_self_intersections'],
            bone_crossings=d['bone_pairs'],muscle_crossings=g['intersection_pairs_by_semantic']['51005'],vessel_crossings=g['intersection_pairs_by_semantic']['51011'])
    return dict(controls=rows_out,longest_measured_prefix_s=trial['accepted_time_s'],matched_opening_surfaces=surfaces,
                installed_cells=reference_cell['cells'],total_skin_fat_mass_kg=reference_cell['mass']['total_kg'],
                earlier_pose_loaded_f32_inversions=before_cell['loaded_state']['float32_inversions'],
                corrected_pose_loaded_f32_inversions=reference_cell['loaded_state']['float32_inversions'],
                latest_dense_trial_pose_loaded_f32_inversions=later_cell['loaded_state']['float32_inversions'],
                old_to_native_dense_root_shift_m=reference_cell['loaded_state']['registration_to_loaded_root_offset_m'],
                sixteen_seconds_qualified=False,sustained_factor=None,
                disposition='Keep the registered normal-traction launch as the comparison scene. Both shear trials improve initial joint accelerations but worsen the matched skin and tissue crossing audits. Preserve them as tested controls.')
if __name__=='__main__':
    p=Path(__file__).resolve().parent
    (p/'shear/decision.json').write_text(json.dumps(compare(p),indent=2,allow_nan=False)+'\n')
