import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

root = Path(sys.argv[1])
out = Path(sys.argv[2])
run_names = sys.argv[3:]
out.mkdir(exist_ok=True)

def read_rows(path, file_name):
    with (path / 'native' / file_name).open(newline='') as f:
        return list(csv.DictReader(f))

def tokens(line):
    return dict(pair.split('=', 1) for pair in line.split() if '=' in pair)

def close(actual, expected, tolerance):
    assert math.isfinite(actual) and abs(actual - expected) <= tolerance, (actual, expected, tolerance)

def summarize(run):
    path = root / run
    launch = json.loads((path / 'launch.json').read_text())
    process = json.loads((path / 'execution.json').read_text())
    step_dt = float(launch['argv'][launch['argv'].index('--muscle-step-seconds') + 1])
    steps = int(launch['argv'][launch['argv'].index('--muscle-step-count') + 1])
    control = launch['control']
    assert process['exit_code'] == 0 and process['stopped'] is None
    log = (path / 'stdout.log').read_text().splitlines()
    timing = tokens(next(line for line in log if line.startswith('resting_integrated_throughput ')))
    body = tokens(next(line for line in log if line.startswith('resting_integrated_body=completed ')))
    terminal = json.loads(next(line.split('=', 1)[1] for line in log if line.startswith('stand_terminal_state=')))
    assert terminal['step_count'] == steps and not terminal['root_assistance']
    assert body['physiology_body_clock'] == 'matched' and body['root_assistance'] == 'false'
    assert int(timing['accepted_end_step']) == steps
    assert not any(re.search(r'^human_standing_progress=(?!accepted\b)', line) for line in log)

    m = read_rows(path, 'resting-com-momentum-diagnostic.csv')
    physiology = read_rows(path, 'resting-coupled.csv')
    surface = read_rows(path, 'resting-surface-audit.csv')
    impulse_rows = read_rows(path, 'resting-com-support-impulses.csv')
    section_map = {}
    if not control:
        for row in read_rows(path, 'resting-local-bed-sections.csv'):
            index = int(row['contact_index'])
            assert index not in section_map
            section_map[index] = row
            for tissue in ('skin', 'fat'):
                expected = float(row[f'{tissue}_thickness_m']) / (float(row[f'{tissue}_modulus_pa']) * float(row['projected_area_m2']))
                close(float(row[f'{tissue}_compliance_m_per_n']), expected, expected * 1e-6)

    assert [int(row['accepted_step']) for row in m] == list(range(8, steps+1, 8))
    assert [int(row['step']) for row in physiology] == list(range(8, steps+1, 8))
    assert terminal['timestep_seconds'] == step_dt
    surfaces = [int(row['step']) for row in surface]
    assert surfaces[0] == 0 and 0 <= steps - surfaces[-1] <= 1
    assert max(b - a for a, b in zip(surfaces, surfaces[1:])) <= 32

    grouped = {}
    for row in impulse_rows:
        step = int(row['accepted_step'])
        slot = int(row['contact_index'])
        assert slot not in grouped.setdefault(step, {})
        grouped[step][slot] = row
    assert set(grouped) == {int(row['accepted_step']) for row in m}

    masses, speeds, load_closure_n, trace = [], [], [], []
    max_equivalent_mm = {'skin': 0.0, 'fat': 0.0}
    supported_geometry = set()
    for motion, gas in zip(m, physiology):
        step = int(motion['accepted_step'])
        time_s = float(motion['time_s'])
        close(float(gas['time_s']), time_s, 1e-8)
        mass = float(motion['total_body_mass_kg'])
        masses.append(mass)
        for field in ('root_assistance_n', 'root_assistance_nm', 'external_blood_ml', 'max_abs_body_mass_residual_kg'):
            close(float(gas[field]), 0, 1e-6)
        velocity = [float(motion[f'com_v{axis}_m_s']) for axis in 'xyz']
        speed = math.hypot(*velocity)
        speeds.append(speed)
        rows = grouped[step]
        assert set(rows) == set(range(32))
        impulses = []
        for index, row in rows.items():
            assert row['row_class'] == 'external_support_contact'
            if not control:
                assert row['source_geometry_index'] == section_map[index]['source_geometry_index']
            normal_impulse = float(row['normal_impulse_ns'])
            z_impulse = float(row['impulse_world_z_ns'])
            assert normal_impulse >= 0
            close(normal_impulse, z_impulse, 1e-7)
            impulses.append(z_impulse)
            if normal_impulse > 0:
                supported_geometry.add(row['source_geometry_index'])
            if not control and time_s >= steps * step_dt - 2:
                for tissue in max_equivalent_mm:
                    equivalent = normal_impulse / step_dt * float(section_map[index][f'{tissue}_compliance_m_per_n']) * 1000
                    max_equivalent_mm[tissue] = max(max_equivalent_mm[tissue], equivalent)
        support_force = float(motion['normal_force_last_physical_step_n'])
        closure = abs(math.fsum(impulses)/step_dt - support_force)
        load_closure_n.append(closure)
        assert closure <= 0.001
        trace.append({
            'step': step, 'time_s': time_s, 'bed_normal_n': support_force,
            'active_bed_rows': sum(impulse > 0 for impulse in impulses),
            'raw_gap_shortening_m': float(gas['peak_penetration_m']),
            'com_z_m': float(motion['com_z_m']), 'com_speed_m_s': speed,
            'lung_volume_ml': float(gas['lung_volume_ml']), 'airflow_ml_s': float(gas['airflow_ml_s']),
            'lv_ml': float(gas['lv_ml']), 'aortic_ejected_ml': float(gas['aortic_ejected_ml']),
            'PaO2_mmhg': float(gas['PaO2_mmhg']), 'PaCO2_mmhg': float(gas['PaCO2_mmhg']),
        })
    late = [row for row in trace if row['time_s'] >= steps * step_dt - 2]
    late_load = [row['bed_normal_n'] for row in late]
    expected_weight = masses[-1]*9.81
    peak = max(row['raw_gap_shortening_m'] for row in trace)
    late_peak = max(row['raw_gap_shortening_m'] for row in late)
    close(max(masses), 72, 1e-6)
    close(min(masses), 72, 1e-6)
    assert .98 <= statistics.mean(late_load)/expected_weight <= 1.02
    assert speeds[-1] < .005
    if not control:
        assert peak < .0125 and late_peak < .00625
        assert float(timing['real_time_factor']) >= .1
    for row in surface:
        for flag in ('vertices_below_1mm', 'nonfinite_skin_vertices', 'mesh_zero_area_triangles', 'mesh_nonfinite_area_triangles', 'functional_geometry_status'):
            assert int(row[flag]) == 0, (run, row['step'], flag)
        assert int(row['mesh_triangles_checked']) == 5238307

    frame_profiles = [tokens(line) for line in log if line.startswith('resting_render_profile ')]
    body_profiles = [tokens(line) for line in log if line.startswith('resting_native_profile ')]
    assert len(surface) == len(frame_profiles)
    frames_after_start = sum(int(frame['step']) >= int(timing['accepted_start_step']) for frame in frame_profiles)
    with (out / f'{run}-trace.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trace[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(trace[::16])
        if len(trace) % 16 != 1:
            writer.writerow(trace[-1])
    elapsed = float(body['simulated_s'])
    print(run, 'steady weight', statistics.mean(late_load), 'peak shortening', peak, 'late', late_peak)
    return {
        'control_hard_bed': control,
        'device': 'Apple M6',
        'physics_dt_s': step_dt,
        'accepted_steps': steps,
        'native_motion_audits': len(surface),
        'triangles_per_frame': 5238307,
        'lowest_displayed_skin_bed_gap_m': min(float(row['min_skin_bed_gap_m']) for row in surface),
        'root_assistance': False,
        'segment_physiology_body_clock': 'matched',
        'accepted_simulated_s': elapsed,
        'accepted_viewer_segment_rtf': float(timing['real_time_factor']),
        'accepted_viewer_segment_wall_s': float(timing['wall_s']),
        'process_seconds_with_startup': process['seconds_process'],
        'process_rtf_with_startup': elapsed/process['seconds_process'],
        'wall_clock_fps_accepted_segment': frames_after_start/float(timing['wall_s']),
        'median_render_gpu_ms': statistics.median(float(row['gpu_ms']) for row in frame_profiles),
        'median_accepted_eight_step_gpu_ms': statistics.median(float(row['gpu_ms']) for row in body_profiles),
        'bed_weight_n': expected_weight,
        'late_2s_mean_bed_reaction_n': statistics.mean(late_load),
        'late_bed_reaction_range_n': [min(late_load), max(late_load)],
        'late_active_support_row_range': [min(row['active_bed_rows'] for row in late), max(row['active_bed_rows'] for row in late)],
        'sampled_contact_source_count_with_nonzero_load': len(supported_geometry),
        'max_per_sample_bed_impulse_sum_disagreement_n': max(load_closure_n),
        'maximum_sampled_com_speed_m_s': max(speeds),
        'terminal_com_speed_m_s': speeds[-1],
        'maximum_com_vertical_drop_m': float(surface[0]['body_com_z_m']) - min(row['com_z_m'] for row in trace),
        'terminal_com_vertical_drop_m': float(surface[0]['body_com_z_m']) - trace[-1]['com_z_m'],
        'peak_raw_contact_gap_shortening_m': peak,
        'late_peak_raw_contact_gap_shortening_m': late_peak,
        'late_peak_force_equivalent_layer_shortening_mm': None if control else max_equivalent_mm,
        'last_lv_stroke_ml': float(physiology[-1]['last_lv_stroke_ml']),
        'final_aortic_ejection_ml': float(physiology[-1]['aortic_ejected_ml']),
        'final_pulmonary_ejection_ml': float(physiology[-1]['pulmonary_ejected_ml']),
        'final_blood_volume_error_ml': float(physiology[-1]['blood_error_ml']),
        'breaths': int(physiology[-1]['breaths']),
        'modeled_filling_ejection_cycles': int(physiology[-1]['complete_filling_ejection_cycles']),
        'end_PaO2_mmhg': float(physiology[-1]['PaO2_mmhg']),
        'end_PaCO2_mmhg': float(physiology[-1]['PaCO2_mmhg']),
        'last_tidal_volume_ml': float(physiology[-1]['tidal_ml']),
    }

results = {name: summarize(name) for name in run_names}
(out / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
