import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

work = Path.cwd()
current = work / 'local-bed-layers-002'
publication = Path(__file__).resolve().parent
runs = ('loaded-16s', 'rigid-8s', 'loaded-24s')
subprocess.run([sys.executable, str(publication/'check_motion.py'), str(current), str(publication), *runs],
                check=True)
results = json.loads((publication/'metrics.json').read_text())

def parsed(run, leaf):
    with (current/run/'native'/leaf).open(newline='') as f:
        return list(csv.DictReader(f))

def term(run):
    log = current/run/'stdout.log'
    with log.open() as f:
        return json.loads(next(line.split('=',1)[1] for line in f if line.startswith('stand_terminal_state=')))

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

matched = ('loaded-16s','rigid-8s')
t = [term(run) for run in runs]
for k in ('initial_q', 'initial_v'):
    assert all(e[k] == t[0][k] for e in t[1:])
controls = {}
for run in matched:
    motion = parsed(run, 'resting-com-momentum-diagnostic.csv')
    matched_motion = [r for r in motion if 6 <= float(r['time_s']) <= 8.000001]
    controls[run] = {k: statistics.mean(float(r[k]) for r in matched_motion)
                    for k in ('normal_force_last_physical_step_n', 'com_z_m')}
soft, rigid = (controls[key] for key in matched)
results['matched_initial_state'] = {
    'verified_matching_articulated_pose_and_velocity': True,
    'comparison_interval_s': [6,8],
    'mean_payloads': controls,
    'com_lower_in_compliant_run_mm': (rigid['com_z_m'] - soft['com_z_m']) * 1000,
}
setup = '380f2cd4864bfa2a3d445957d974fec32ae39bfaffca49beba01c5a3b25ad176'
for run in runs:
    run_path = current/run
    evidence = [
        'native/resting-coupled.csv',
        'native/resting-com-momentum-diagnostic.csv',
        'native/resting-com-support-impulses.csv',
        'native/resting-surface-audit.csv',
        'native/resting-human.mrvpack',
        'stdout.log',
        'stderr.log',
        'launch.json',
        'execution.json',
    ]
    if run != 'rigid-8s':
        evidence.append('native/resting-local-bed-sections.csv')
    results[run]['local_full_evidence'] = str(run_path)
    results[run]['raw_sha256'] = {name: digest(run_path/name) for name in evidence}
    assert results[run]['raw_sha256']['native/resting-human.mrvpack'] == setup
    shutil.copyfile(run_path/'launch.json', publication/f'{run}-launch.json')
    shutil.copyfile(run_path/'execution.json', publication/f'{run}-execution.json')

(publication/'metrics.json').write_text(json.dumps(results, indent=2)+'\n')
shutil.copyfile(current/'loaded-24s/native/resting-local-bed-sections.csv',
                publication/'resting-local-bed-sections.csv')
rows = parsed('loaded-24s', 'resting-com-support-impulses.csv')
rows = [row for row in rows if row['accepted_step'] == str(results['loaded-24s']['accepted_steps'])]
with (publication/'loaded-24s-terminal-support-impulses.csv').open('w',newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

late = results['loaded-24s']
print('Qualification:',json.dumps({key:value for key,value in late.items() if key != 'raw_sha256'},indent=2))
