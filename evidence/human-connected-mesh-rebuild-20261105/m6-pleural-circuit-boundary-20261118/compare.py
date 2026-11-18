import csv
import json
import re
from pathlib import Path
from statistics import mean

here = Path(__file__).resolve().parent
criteria = json.loads((here/'qualification.json').read_text())['criteria']

def load(arm):
    out = here/arm
    meta = json.loads((out/'execution.json').read_text())
    assert meta['exit_code']==0 and meta['stopped'] is None
    lines = (out/'stdout.log').read_text().splitlines()
    age = next(line for line in lines if line.startswith('resting_integrated_throughput '))
    terminal = json.loads(next(line.split('=',1)[1] for line in lines if line.startswith('stand_terminal_state=')))
    assert terminal['step_count']==criteria['end_step'] and not terminal['root_assistance']
    def rows(name):
        with (out/'native'/name).open(newline='') as f:
            return list(csv.DictReader(f))
    state = rows('resting-coupled.csv')
    surface = rows('resting-surface-audit.csv')
    support = rows('resting-com-momentum-diagnostic.csv')
    assert [int(r['step']) for r in state] == list(range(8,criteria['end_step']+1,8))
    assert all(int(r['vertices_below_1mm'])==int(r['nonfinite_skin_vertices'])==
               int(r['mesh_zero_area_triangles'])==int(r['mesh_nonfinite_area_triangles'])==0 for r in surface)
    assert all(float(r['root_assistance_n'])==float(r['root_assistance_nm'])==0 for r in state)
    circulation = {key:float(state[-1][key]) for key in ['pulmonary_ejected_ml','aortic_ejected_ml','blood_ml']}
    circulation['complete_cycles'] = int(state[-1]['complete_filling_ejection_cycles'])
    assert circulation['complete_cycles']>1 and circulation['pulmonary_ejected_ml']>0 and circulation['aortic_ejected_ml']>0
    flow = {
        **circulation,
        'viewer_rtf':float(re.search(r'\breal_time_factor=(\S+)',age)[1]),
        'maximum_blood_error_ml':max(abs(float(r['blood_error_ml'])) for r in state),
        'late_support_weight_ratio':mean(float(r['normal_force_last_physical_step_n']) for r in support if float(r['time_s'])>10)/(9.81*float(support[-1]['total_body_mass_kg'])),
    }
    stage = next((line for line in lines if line.startswith('resting_vascular_external_pressure=')),None)
    return terminal,state,flow,stage

base,csv_base,b,s1=load('pressure-off')
modified,csv_new,m,s2=load('pressure-on')
assert s1 is None and s2 and 'compartment_mask=' in s2
assert modified['initial_q']==base['initial_q'] and modified['initial_v']==base['initial_v']
differences = {k:max(abs(float(x[k])-float(y[k])) for x,y in zip(csv_base,csv_new))
               for k in ['pulmonary_ejected_ml','aortic_ejected_ml','lv_mmhg','rv_mmhg',
                         'aorta_mmhg','pulmonary_artery_mmhg','PaO2_mmhg','PaCO2_mmhg']}
bounds = criteria['late_support_weight_ratio']
gates = dict(throughput=all(x['viewer_rtf']>=criteria['viewer_accepted_time_rtf_min'] for x in (b,m)),
             mass=all(x['maximum_blood_error_ml']<=criteria['maximum_sampled_blood_volume_error_ml'] for x in (b,m)),
             support=all(bounds[0]<=x['late_support_weight_ratio']<=bounds[1] for x in (b,m)),
             pulmonary_ejection=differences['pulmonary_ejected_ml']>=criteria['minimum_nonzero_pulmonary_ejection_divergence_ml'],
             cardiac_pressure=max(differences['lv_mmhg'],differences['rv_mmhg'])>=criteria['minimum_nonzero_cardiac_pressure_divergence_mmhg'])
result={'reference':b,'pleural_pressure':m,'maximum_matched_differences':differences,'active_boundary':s2,'gates':gates}
(here/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(0 if all(gates.values()) else 1)
