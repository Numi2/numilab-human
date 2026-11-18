import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

publish = Path(__file__).resolve().parent
work = publish.parent
load = work / 'resume-accepted-knee-load-001'
plan = json.loads((load/'accepted-load.plan.json').read_text())
control_ids = ('005', '009', '010', '011', '012', '013', '014', '015', '016')
report = {
    'device': 'Apple M6, SSH Numi',
    'system': 'macOS 27.0 (26A425), arm64',
    'reference_launch_sha256': hashlib.file_digest((load/'accepted-load.plan.json').open('rb'), 'sha256').hexdigest(),
    'native_binary': plan['native_binary'],
    'runtime_dependencies': plan['runtime_dependencies'],
    'current_source_overlays': plan['current_native_source_overlays'],
    'source_revision': plan['full_execution_source_pins']['head'],
    'source_dirty_patch': plan['full_execution_source_pins']['dirty_patch'],
    'dense_reference': {'run': 'resume-aortic-discrete-native-002', 'fem_nodes': 318226,
        'tetrahedra': 1475249, 'accepted_roots': 8, 'simulated_s': .000125,
        'wall_gpu_command_s': 197.528, 'real_time_factor_gpu_command_span': .000125/197.528},
    'controls': {},
    'admission_failures': {},
}
def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

for i in ('001', '002', '003', '006', '007', '008'):
    path = work/f'rescope-resting-native-{i}'
    log = (path/'stderr.log').read_text().splitlines()
    error = [line for line in log if 'failed error=' in line or line.startswith('tissue_mass_partition_failure')]
    report['admission_failures'][i] = error[-3:]

for i in control_ids:
    path = work/f'rescope-resting-native-{i}'
    if not (path/'execution.json').is_file():
        continue
    launch = json.loads((path/'launch.json').read_text())
    execution = json.loads((path/'execution.json').read_text())
    trace = path/'native/resting-coupled.csv'
    log = (path/'stdout.log').read_text().splitlines()
    body_lines = [l for l in log if l.startswith('resting_integrated_body=')]
    d = {'execution': execution,
        'live_presentation_requested': '--mechanics-only' not in launch['argv'],
        'launch_file': f'{i}-launch.json',
        'binary_sha256': digest(Path(launch['argv'][0])),
        'completed_record': body_lines[-1] if body_lines else None}
    d['direct_asset_sha256'] = {arg: digest(Path(arg))
                                for arg in launch['argv'][1:]
                                if arg.startswith('/') and Path(arg).is_file()
                                and not arg.endswith('.mov')}
    (publish/f'{i}-launch.json').write_text((path/'launch.json').read_text())
    report['controls'][i] = d
    throughput = [line for line in log if line.startswith('resting_integrated_throughput ')]
    if throughput:
        d['accepted_segment_throughput'] = throughput[-1]
    if execution['exit_code']:
        d['terminal_stderr'] = [line for line in (path/'stderr.log').read_text().splitlines()
                                if (line.startswith(('myosim_articulated_visual=failed',
                                                     'resting_transaction_encode_failure',
                                                     'human_joint_failure')))][-5:]
    if i in ('010', '011', '012', '013', '014', '015', '016'):
        stderr = (path/'stderr.log').read_text().splitlines()
        d['geometry_failure_witness'] = [
            line for line in stderr if line.startswith('resting_fem_failure_accepted_witness')]
        dispatches = [line for line in stderr if line.startswith(
            'matter_fgmres_gpu_indirect_dispatch_count=')]
        if dispatches:
            d['first_root_encoded_indirect_dispatches'] = dispatches[0]
        if (path/'common-failure.json').is_file():
            d['rejection_receipt_sha256'] = digest(path/'common-failure.json')
        d['regional_preparation'] = [line for line in log if line.startswith(('whole_body_tissue_mass_reference=', 'resting_fascia_reference_mode=', 'resting_fem_common=', 'resting_aortic_wall='))][-12:]
    progress = [l for l in log if l.startswith('human_standing_progress=accepted')]
    measures = [dict(re.findall(r'(\w+)=([^ \[]+)', line)) for line in progress]
    beginnings = [float(l.split('wall_elapsed_ms=', 1)[1].split()[0])
                  for l in log if l.startswith('human_execution_stage=authoritative_segment_begin ')]
    durations = [float(l.split('wall_elapsed_ms=', 1)[1].split()[0])
                 for l in log if l.startswith('human_execution_stage=authoritative_segment_end ')]
    if durations:
        d['first_segment_complete_at_execution_ms'] = durations[0]
    if measures:
        d['accepted_progress_records'] = len(measures)
        if i in ('012', '016') and len(durations)>=len(measures):
            d['consecutive_accepted_root_wall_ms'] = [
                end - start for start, end in zip(beginnings, durations[:len(measures)])]
        tail = [m for m in measures if float(m['simulated_seconds']) >= float(measures[-1]['simulated_seconds']) - 2]
        supports = [float(m['support_force_n']) for m in tail]
        d['support_last_two_seconds_N'] = {'mean': statistics.mean(supports), 'min': min(supports), 'max': max(supports)}
        d['accepted_sample_max_penetration_m'] = max(float(m['penetration_m']) for m in measures)
        d['terminal_accepted_progress'] = progress[-1]
    terminal = [json.loads(l.split('=',1)[1]) for l in log if l.startswith('stand_terminal_state=')]
    if terminal:
        t = terminal[-1]
        speeds = t['v']
        d['terminal_knee_rate_rad_s_at_indices_105_119'] = [speeds[105], speeds[119]]
        d['terminal_max_joint_speed_rad_s'] = max(abs(v) for v in speeds[6:])
        d['terminal_root_assistance'] = t['root_assistance']
    if not trace.is_file():
        continue
    with trace.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    d['trace_original_sha256'] = digest(trace)
    d['observed_simulated_seconds'] = float(rows[-1]['time_s'])
    d['throughput_including_process_wall'] = d['observed_simulated_seconds'] / execution['process_wall_seconds']
    d['trace_max_penetration_m'] = max(float(r['peak_penetration_m']) for r in rows)
    d['terminal_measurements'] = {k: float(rows[-1][k]) for k in (
        'breaths','tidal_ml','complete_filling_ejection_cycles','last_lv_stroke_ml','aortic_ejected_ml',
        'pulmonary_ejected_ml','PaO2_mmhg','PaCO2_mmhg','blood_error_ml')}
    d['observed_ranges'] = {k: [min(values), max(values)] for k in (
        'lung_volume_ml','airflow_ml_s','PaO2_mmhg','PaCO2_mmhg') if (values := [float(r[k]) for r in rows])}
    columns = ['time_s','step','lung_volume_ml','airflow_ml_s','alveolar_pa','pleural_pa','diaphragm_mm','rib_mm',
               'PaO2_mmhg','PaCO2_mmhg','aortic_ejected_ml','pulmonary_ejected_ml','blood_error_ml',
               'peak_penetration_m','root_assistance_n','root_assistance_nm']
    with (publish/f'{i}-trace.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, columns, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        w.writerows([*rows[:-1:16], rows[-1]])
    momentum = path/'native/resting-com-momentum-diagnostic.csv'
    if momentum.is_file():
        with momentum.open() as f:
            mrows = list(csv.DictReader(f))
        if mrows:
            d['terminal_com_speed_m_s'] = math.hypot(*(float(mrows[-1][f'com_v{axis}_m_s']) for axis in 'xyz'))
    if i == '009':
        render = [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in log
                  if line.startswith('resting_render_profile ')]
        if render:
            d['render_profile'] = {
                'submissions': len(render),
                'triangles_last_submission': int(render[-1]['triangles']),
                'median_gpu_ms': statistics.median(float(row['gpu_ms']) for row in render),
                'median_render_wall_ms': statistics.median(float(row['total_render_wall_ms']) for row in render),
            }
        surface = path/'native/resting-surface-audit.csv'
        if surface.is_file():
            with surface.open() as f:
                frames = list(csv.DictReader(f))
            d['surface_audit_record_count'] = len(frames)

(publish/'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
