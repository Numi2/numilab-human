"""Seat the floating root against the existing flat bed using its measured pressure points."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from emit_skin_source_registration import sha
run, donor, cfg, output=map(Path,sys.argv[1:])
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
launch=json.loads((run/'launch.json').read_text())
configuration=json.loads(cfg.read_text())
receipt=json.loads((run/'terminal.receipt.json').read_text())
assert receipt['accepted_step']==1 and launch['environment']['NUMI_HUMAN_LOCAL_BED_REGISTERED_OUTER']=='1'
assert launch['construction_posture']['sha256']==sha(donor/'stdout.log')
assert configuration['construction_posture']==launch['construction_posture']
selected={}
for row in read(run/'resting-support-selected-skin-witness.csv'):
    i=int(row['region'])
    assert i not in selected or selected[i]==row
    selected[i]=row
    assert int(row['accepted_step'])==1 and int(row['pre_dynamics_step_index'])==0
    assert not any(int(row[k]) for k in ('region_error','global_error','nonfinite_jacobian_count'))
sections=read(run/'resting-local-bed-sections.csv')
assert len(sections)==len(selected)==32
assert all(int(s['contact_index'])==i and s['source_geometry_index']==selected[i]['source_geometry_index'] for i,s in enumerate(sections))
points=np.array([[float(selected[i]['selected_point_world_'+a+'_m']) for a in 'xyz'] for i in range(32)])
compliance=np.array([float(s['skin_compliance_m_per_n'])+float(s['fat_compliance_m_per_n']) for s in sections])
stock=np.array([float(s['skin_thickness_m'])+float(s['fat_thickness_m']) for s in sections])
state=read(donor/'resting-com-momentum-diagnostic.csv')[-1]
assert int(state['accepted_step'])==json.loads((donor/'terminal.receipt.json').read_text())['accepted_step']
center=np.array([float(state['com_'+a+'_m']) for a in 'xyz'])
target=float(state['total_body_mass_kg'])*9.81
ground=Path(launch['argv'][launch['argv'].index('--support-contact-payload')+1])
import struct
assert np.array_equal(struct.unpack_from('<6f',ground.read_bytes(),56), [0.,0.,0.,0.,0.,1.])
def normal(d):
    return np.maximum(0.,-points[:,2]-d)/compliance
lo=float(np.max(-points[:,2]-stock))
hi=float(-points[:,2].min())
if normal(lo).sum()<=target:raise ValueError('weight cannot be supported within registered tissue thickness')
delta=brentq(lambda d:normal(d).sum()-target,lo,hi,xtol=2e-15)
force=normal(delta)
assert np.all(force*compliance<stock) and abs(force.sum()-target)<1.e-6
expected=dict(source_selector_sha256=sha(run/'resting-support-selected-skin-witness.csv'),
              measured_body_sha256=sha(donor/'resting-com-momentum-diagnostic.csv'),sections_sha256=sha(run/'resting-local-bed-sections.csv'),
              target_N=target,before_N=float(normal(0).sum()),predicted_N=float(force.sum()),
              vertical_placement_m=delta,active_rows=int((force>0).sum()),
              stock_compression_fraction_max=float(max(force*compliance/stock)),
              center_of_mass_m=center.tolist(),normal_moment_about_com_Nm=np.cross(points-center, np.c_[np.zeros((32,2)),force]).sum(0).tolist())
output.mkdir(exist_ok=False);(output/'height.json').write_text(json.dumps(expected,indent=2)+'\n')
configuration['environment']['NUMI_HUMAN_REGISTERED_BED_VERTICAL_PLACEMENT_M']=format(delta,'.17g')
configuration['environment']['NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS']='1'
(output/'config.json').write_text(json.dumps(configuration,indent=2)+'\n')
(output/'run_supported.py').hardlink_to(cfg.parent/'run_supported.py')
print(expected)
