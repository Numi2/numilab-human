import argparse
import csv
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
args=p.parse_args()
out=args.run
native=out/'native'
def read(name):
    with (native/name).open(newline='') as f:
        return list(csv.DictReader(f))
sections=read('resting-local-bed-sections.csv')
d=read('resting-coupled.csv')
s=read('resting-surface-audit.csv')
impulses=read('resting-com-support-impulses.csv')
step_time=.002
result=json.loads((out/'execution.json').read_text())
assert result['exit_code']==0 and len(s)>1
m=read('resting-com-momentum-diagnostic.csv')
assert len(d)==len(m)
assert all(float(r['root_assistance_n'])==float(r['root_assistance_nm'])==0 for r in d)
assert all(int(r['vertices_below_1mm'])==int(r['nonfinite_skin_vertices'])==
           int(r['mesh_zero_area_triangles'])==int(r['mesh_nonfinite_area_triangles'])==0 for r in s)
assert len(sections)==32
for r in sections:
    for name in ('skin','fat'):
        predicted=float(r[f'{name}_thickness_m'])/(float(r['projected_area_m2'])*float(r[f'{name}_modulus_pa']))
        assert math.isclose(predicted,float(r[f'{name}_compliance_m_per_n']),rel_tol=1e-6)
assert len(set(float(r['fat_thickness_m']) for r in sections)) > 1
samples={}
for row in impulses:
    i=int(row['contact_index'])
    step=int(row['accepted_step'])
    section=sections[i]
    assert row['source_geometry_index']==section['source_geometry_index']
    force=float(row['normal_impulse_ns'])/step_time
    assert force>=0
    samples.setdefault(step,{})[i]=force
for row in m:
    step=int(row['accepted_step'])
    v=sum(samples[step].values())
    assert abs(v-float(row['normal_force_last_physical_step_n']))<.001
cutoff=int(d[-1]['step'])-int(2/step_time)
loads=[samples[step] for step in samples if step>cutoff]
pressure={}
for i,section in enumerate(sections):
    late_mean_force=statistics.mean(x[i] for x in loads)
    stock=sum(float(section[f'{tissue}_thickness_m']) for tissue in ('skin','fat'))
    compliance=sum(float(section[f'{tissue}_compliance_m_per_n']) for tissue in ('skin','fat'))
    peak_force=max(t[i] for t in loads)
    pressure[i]={'source_vertex':int(section['source_geometry_index']),
                 'mean_force_n':late_mean_force,
                 'peak_late_force_n':peak_force,
                 'mean_effective_pressure_pa':late_mean_force/float(section['projected_area_m2']),
                 'peak_late_force_equivalent_strain':peak_force*compliance/stock}
weight=9.81*float(m[-1]['total_body_mass_kg'])
mean_force=sum(v['mean_force_n'] for v in pressure.values())
assert .98<mean_force/weight<1.02
end=m[-1]
com_speed=math.sqrt(sum(float(end[f'com_v{a}_m_s'])**2 for a in 'xyz'))
assert com_speed<.005
result.update({'late_force_n':mean_force,'weight_n':weight,'terminal_com_speed_m_s':com_speed,
    'late_loaded_sections':sum(v['mean_force_n']>0 for v in pressure.values()),
    'accepted_steps':int(d[-1]['step']),'simulated_seconds':float(d[-1]['time_s']),
    'checked_frames':len(s),'section_pressures':pressure,
    'maximum_late_force_equivalent_section_strain':max(v['peak_late_force_equivalent_strain'] for v in pressure.values()),
    'peak_contact_shortening_m':max(float(r['peak_penetration_m']) for r in d),
    'thickness_m':{t:[f(float(row[f'{t}_thickness_m']) for row in sections) for f in (min,max)] for t in ('skin','fat')},
})
(out/'load-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='section_pressures'},indent=2))
