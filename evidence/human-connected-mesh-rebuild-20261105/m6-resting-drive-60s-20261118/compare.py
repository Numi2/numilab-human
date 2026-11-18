import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

here=Path(__file__).resolve().parent
plan=json.loads((here/'qualification.json').read_text())
runtimes={}
def load(arm):
    path=here/arm
    info=json.loads((path/'execution.json').read_text())
    assert info['exit_code']==0 and info['stopped'] is None, info
    log=(path/'stdout.log').read_text()
    row=next(s for s in log.splitlines() if s.startswith('resting_integrated_throughput '))
    assert 'accepted_end_step=30000 ' in row
    rtf=float(re.search(r'\breal_time_factor=([0-9.]+)',row).group(1))
    terminal=json.loads(next(s.split('=',1)[1] for s in log.splitlines() if s.startswith('stand_terminal_state=')))
    assert not terminal['root_assistance'] and terminal['step_count']==30000
    def rows(name):
        with (path/'native'/name).open(newline='') as f:
            return list(csv.DictReader(f))
    state=rows('resting-coupled.csv')
    bed=rows('resting-com-momentum-diagnostic.csv')
    surface=rows('resting-surface-audit.csv')
    assert len(state)==len(bed)==3750
    assert [int(s['step']) for s in state]==list(range(8,30001,8))
    assert max(float(s['root_assistance_n']) for s in state)==0
    assert max(float(s['root_assistance_nm']) for s in state)==0
    assert all(int(s['vertices_below_1mm'])==int(s['mesh_zero_area_triangles'])==
               int(s['mesh_nonfinite_area_triangles'])==int(s['nonfinite_skin_vertices'])==0 for s in surface)
    assert int(surface[-1]['step']) >= 29999
    weight=9.81*float(bed[-1]['total_body_mass_kg'])
    support=mean(float(s['normal_force_last_physical_step_n']) for s in bed if float(s['time_s'])>58)
    runtimes[arm]={'viewer_rtf':rtf,'mean_bed_force_late_n':support,'body_weight_n':weight,
                   'breaths':int(state[-1]['breaths']),
                   'cardiac_cycles':int(state[-1]['complete_filling_ejection_cycles']),
                   'aortic_ejected_ml':float(state[-1]['aortic_ejected_ml']),
                   'max_blood_conservation_error_ml':max(abs(float(s['blood_error_ml'])) for s in state),
                   'checked_rendered_frames':len(surface),'process':info,
                   'physiology_trace_sha256':hashlib.sha256((path/'native/resting-coupled.csv').read_bytes()).hexdigest()}
    return state, terminal

def paused(records,start,end,key):
    window=[float(r[key]) for r in records if start <= float(r['time_s']) <= end+1e-5]
    assert window
    return mean(window)

def cumulative(records,start,end,key):
    def nearest(target):
        return float(min(records,key=lambda r:abs(float(r['time_s'])-target))[key])
    return nearest(end)-nearest(start)

control,ct=load('reference')
treatment,tt=load('drive-0p6')
assert ct['initial_q']==tt['initial_q'] and ct['initial_v']==tt['initial_v']
start,end=plan['intervention']['window_s']
prefix=[(a,b) for a,b in zip(control,treatment) if round(float(a['time_s']),5)<=start]
assert all(a==b for a,b in prefix)

def inspired_ratio(start,end):
    key='inspired_volume_accum_ml'
    return cumulative(treatment,start,end,key)/cumulative(control,start,end,key)

def gas_delta(start,end):
    return {g:paused(treatment,start,end,g)-paused(control,start,end,g)
            for g in ('PaCO2_mmhg','PaO2_mmhg')}

pre=plan['predictions']
ratio=inspired_ratio(start,end)
gases=gas_delta(*pre['lateInterventionGasWindow_s'])
recovery=gas_delta(*pre['recovery_window_s'])
recovery_ratio=inspired_ratio(*pre['recovery_window_s'])
mechanism_checks = {
    'ventilation_reduction': pre['inspiredVolume_8_20_ratio'][0]<=ratio<=pre['inspiredVolume_8_20_ratio'][1],
    'arterial_co2_increase': gases['PaCO2_mmhg']>=pre['arterialCO2DifferenceMin_mmhg'],
    'arterial_o2_decrease': gases['PaO2_mmhg']<=pre['arterialO2DifferenceMax_mmhg'],
    'gas_recovery': all(abs(recovery[g]) < abs(gases[g]) and abs(recovery[g])<=pre['gas_abs_max_mmhg'] for g in recovery),
    'ventilation_recovery': pre['late_inspired_volume_ratio'][0]<=recovery_ratio<=pre['late_inspired_volume_ratio'][1],
    'viewer_rate_both': all(x['viewer_rtf']>=plan['validity']['minimum_rendered_rtf'] for x in runtimes.values()),
    'bed_balance_both': all(.98<=x['mean_bed_force_late_n']/x['body_weight_n']<=1.02 for x in runtimes.values()),
}
report={
    'scope':plan['scope'], 'runs':runtimes, 'matched_prefix_samples':len(prefix),
    'inspired_volume_ratio_during_intervention':ratio,'gas_response_mean_difference_mmhg':gases,
    'inspired_volume_ratio_final_window':recovery_ratio,'final_gas_mean_difference_mmhg':recovery,
    'aortic_ejection_identical_all_samples':all(a['aortic_ejected_ml']==b['aortic_ejected_ml'] for a,b in zip(control,treatment)),
    'mechanism_checks':mechanism_checks}
(here/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
