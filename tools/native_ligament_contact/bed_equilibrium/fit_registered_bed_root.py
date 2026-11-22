"""Fit the source floating root to weight and moment with the measured terminal body mass."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from emit_skin_source_registration import sha


def records(path):
    with path.open() as f:return list(csv.DictReader(f))


def select_points(path):
    seen={}
    for row in records(path):
        i=int(row['region'])
        if i in seen and row!=seen[i]:raise ValueError('ambiguous same-step material point')
        if int(row['pre_dynamics_step_index']) or int(row['accepted_step'])!=1 or any(int(row[k]) for k in ('global_error','region_error','nonfinite_jacobian_count')):
            raise ValueError('support point did not come from the first physical step')
        seen[i]=row
    if set(seen)!=set(range(32)):raise ValueError('incomplete supported interface')
    return seen


def main(run, donor, cfg, out):
    base=json.loads(cfg.read_text());launch=json.loads((run/'launch.json').read_text())
    assert launch['environment']['NUMI_HUMAN_LOCAL_BED_REGISTERED_OUTER']=='1'
    assert base['construction_posture']==launch['construction_posture']
    assert launch['construction_posture']['sha256']==sha(donor/'stdout.log')
    assert '--root-pose' not in base['argv']
    previous=launch['argv']
    donor_state=None
    with (donor/'stdout.log').open() as f:
        for line in f:
            if line.startswith('stand_terminal_state='):donor_state=json.loads(line.split('=',1)[1])
    assert donor_state and not donor_state['root_assistance']
    original=np.asarray(donor_state['q'][:7])
    current=np.array(previous[previous.index('--root-pose')+1:previous.index('--root-pose')+8],dtype=float) if '--root-pose' in previous else original.copy()
    current[2]+=float(launch['environment'].get('NUMI_HUMAN_REGISTERED_BED_VERTICAL_PLACEMENT_M',0))
    initial_R=Rotation.from_quat(original[3:]);now_R=Rotation.from_quat(current[3:])
    body=records(donor/'resting-com-momentum-diagnostic.csv')[-1]
    assert int(body['accepted_step'])==donor_state['step_count']
    mass=float(body['total_body_mass_kg']); target=mass*9.81
    original_center=np.array([float(body['com_'+a+'_m']) for a in 'xyz'])
    center=current[:3]+(now_R*initial_R.inv()).apply(original_center-original[:3])
    selected=select_points(run/'resting-support-selected-skin-witness.csv')
    mat=records(run/'resting-local-bed-sections.csv')
    assert len(mat)==32 and all(int(m['contact_index'])==i and m['source_geometry_index']==selected[i]['source_geometry_index'] for i,m in enumerate(mat))
    p=np.array([[float(selected[i]['selected_point_world_'+a+'_m']) for a in 'xyz'] for i in range(32)])
    compliance=np.array([float(s['skin_compliance_m_per_n'])+float(s['fat_compliance_m_per_n']) for s in mat])
    stock=np.array([float(s['skin_thickness_m'])+float(s['fat_thickness_m']) for s in mat])
    def evaluate(q):
        rot=Rotation.from_rotvec([q[1],q[2],0])
        xyz=rot.apply(p-center)+center+[0,0,q[0]]
        compression=np.maximum(-xyz[:,2],0)
        forces=compression/compliance
        torque=np.cross(xyz-center,np.column_stack((np.zeros((len(xyz),2)),forces))).sum(0)
        return np.r_[forces.sum()-target,torque[:2]],float(max(compression/stock)),rot
    fit=least_squares(lambda q:evaluate(q)[0],np.zeros(3),xtol=1e-14,gtol=1e-12,ftol=1e-14,x_scale='jac',max_nfev=150)
    residual,fraction,rot=evaluate(fit.x)
    if not fit.success or max(abs(residual))>1e-5 or fraction>=1:raise ValueError(('no valid free placement',residual,fraction))
    new=np.r_[rot.apply(current[:3]-center)+center+[0,0,fit.x[0]],(rot*now_R).as_quat()]
    base['argv']+=['--root-pose',*[format(v,'.17g') for v in new]]
    base['environment']['NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS']='1'
    out.mkdir(exist_ok=False)
    (out/'run_supported.py').hardlink_to(cfg.parent/'run_supported.py')
    (out/'config.json').write_text(json.dumps(base,indent=2)+'\n')
    result=dict(source_selector_sha256=sha(run/'resting-support-selected-skin-witness.csv'),source_sections_sha256=sha(run/'resting-local-bed-sections.csv'),
                source_posture_sha256=sha(donor/'stdout.log'),body_mass_kg=mass,target_N=target,
                previous_force_error_N_and_moments_Nm=evaluate(np.zeros(3))[0].tolist(),center_m=center.tolist(),original_q=original.tolist(),new_root_q=new.tolist(),
                dz_m=float(fit.x[0]),roll_pitch_rad=fit.x[1:].tolist(),after_force_error_N_and_moments_Nm=residual.tolist(),maximum_fraction_of_stock=fraction,
                scope='Placement fit with frozen contact selection. The next native launch reselects every support node and recomputes the forces.')
    (out/'fit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('root fit',result['previous_force_error_N_and_moments_Nm'],fit.x.tolist(),fraction)
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
