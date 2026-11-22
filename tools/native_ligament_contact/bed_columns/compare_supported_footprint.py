"""Compare measured bed mechanics, physiology and physical anatomy at matched native steps."""
import csv
import json
import re
import statistics
import sys
from pathlib import Path


def load(case, name):
    return json.loads((case / name).read_text())


def table(case, name):
    with (case / name).open() as stream:
        return list(csv.DictReader(stream))


def bank_pairs(folder):
    result = {}
    counts = load(folder, 'physical-banks.json')
    for line in (folder/'target-witnesses.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['semantic'] == 51010 or row['ocular_source_id']:
            raise ValueError('This comparison requires independently classifying the ocular witnesses')
        key = row['semantic'], row['stable_id'], row['primitive']
        pairs = set(map(tuple, row['triangle_pairs']))
        if key in result or len(pairs) != row['count']:
            raise ValueError('Duplicate bank or witness')
        result[key] = pairs
    if sum(map(len, result.values())) != counts['physical_nonocular_pairs']:
        raise ValueError('Registration audit count mismatch')
    return result


def differences(first, last):
    shared = sum(len(pairs & last.get(bank, set())) for bank, pairs in first.items())
    return dict(persisting_pairs=shared, disappeared=sum(map(len,first.values()))-shared,
                appeared=sum(map(len,last.values()))-shared)


def measure(control, trial, early, observed):
    expected = load(trial,'terminal.receipt.json')
    start = load(early,'terminal.receipt.json')
    actual = load(observed,'terminal.receipt.json')
    for key in ('pack_file_sha256','accepted_body_state_sha256','accepted_respiration_state_sha256',
                'accepted_transaction_fingerprint','base_pack_file_sha256'):
        if start[key] != actual[key]:
            raise ValueError('Startup observation changed accepted state')
    if expected['base_pack_file_sha256'] != start['base_pack_file_sha256']:
        raise ValueError('Native runs use different initial geometry')
    runs = [control, trial]
    metrics = []
    loads = []
    areas = []
    for run in runs:
        launch = load(run,'launch.json')
        args = launch['argv']
        dt = float(args[args.index('--muscle-step-seconds')+1])
        if launch['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1' or dt != .002 or launch['headless']:
            raise ValueError('Different physical run mode')
        traces = table(run,'resting-coupled.csv')
        matched = [r for r in traces if int(r['step'])<=expected['accepted_step']]
        end = matched[-1]
        momentum = table(run,'resting-com-momentum-diagnostic.csv')
        window = {int(r['accepted_step']):r for r in momentum if 7000<int(r['accepted_step'])<=8000}
        if list(window)!=list(range(7008,8001,32)):
            raise ValueError('Missing matched late samples')
        force = {}
        for r in table(run,'resting-com-support-impulses.csv'):
            s = int(r['accepted_step'])
            if s not in window: continue
            if r['row_class'] != 'external_support_contact': raise ValueError('Unexpected support type')
            key = int(r['contact_index']), int(r['source_geometry_index'])
            force[key] = force.get(key,0.) + float(r['normal_impulse_ns'])/(dt*len(window))
        loads.append(force)
        a = {(int(r['contact_index']),int(r['source_geometry_index'])):float(r['projected_area_m2']) for r in table(run,'resting-local-bed-sections.csv')}
        if len(a)!=32 or a.keys()!=force.keys(): raise ValueError('Unsupported section identity')
        areas.append(a)
        mean = statistics.mean(float(r['normal_force_last_physical_step_n']) for r in window.values())
        if abs(mean-sum(force.values()))>.001: raise ValueError('Unclosed support reactions')
        metrics.append(dict(bed_area_m2=sum(a.values()),mean_sampled_bed_force_14_to_16_s_N=mean,
            sampled_bed_force_range_14_to_16_s_N=[f(float(r['normal_force_last_physical_step_n']) for r in window.values()) for f in (min,max)],
            endpoint={k:float(end[k]) for k in ('time_s','breaths','complete_filling_ejection_cycles','blood_ml','blood_error_ml','PaO2_mmhg',
                'PaCO2_mmhg','tidal_ml','last_lv_stroke_ml','root_assistance_n','root_assistance_nm')},
            sampled_pleural_pressure_Pa=[f(float(r['pleural_pa']) for r in matched) for f in (min,max)]))
    rows=[dict(row=k[0],geometry=k[1],original_area_m2=areas[0][k],corrected_area_m2=areas[1][k],
        original_mean_normal_N=loads[0][k],corrected_mean_normal_N=loads[1][k]) for k in sorted(areas[0])]
    initial = observed/'initial-envelope'
    controls = [initial,early/'envelope',trial/'envelope']
    witness_sets = list(map(bank_pairs,controls))
    log=(trial/'stdout.log').read_text()
    wall=float(re.search(r'^resting_integrated_body=completed simulated_s=\S+ wall_s=(\S+)',log,re.M)[1])
    frames=int(re.search(r'^resting_integrated_observer_profile accepted_callbacks=(\d+)',log,re.M)[1])
    return dict(control=metrics[0],corrected=metrics[1],regional=rows,
        envelopes=[dict(load(p,'physical-banks.json'),skin_self_intersections=load(p,'initial-envelope.json' if i==0 else 'accepted-envelope.json')['skin_self_intersections']) for i,p in enumerate(controls)],
        initial_to_64ms=differences(witness_sets[0],witness_sets[1]),
        initial_to_16s=differences(witness_sets[0],witness_sets[2]),early_to_16s=differences(*witness_sets[1:]),
        native_wall_s=wall,native_factor=expected['accepted_time_s']/wall,
        cold_wall_s=load(trial,'execution.json')['cold_wall_s'],cold_factor=expected['accepted_time_s']/load(trial,'execution.json')['cold_wall_s'],
        accepted_observer_callbacks=frames,mean_callbacks_per_native_wall_second=frames/wall,
        endpoint_sha256=expected['pack_file_sha256'])


if __name__ == '__main__':
    print(json.dumps(measure(*(Path(p).resolve() for p in sys.argv[1:])),indent=2,allow_nan=False))
