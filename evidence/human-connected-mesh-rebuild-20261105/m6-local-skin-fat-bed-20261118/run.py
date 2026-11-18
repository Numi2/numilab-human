import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('--steps', type=int, required=True)
parser.add_argument('--rigid', action='store_true')
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()
assert 8 <= args.steps <= 12000
assert Path(args.name).name == args.name and args.name not in ('', '.', '..')
here = Path(__file__).resolve().parent
control = here.parent / 'rescope-resting-publication-20261118'
prior = json.loads((control/'metrics.json').read_text())
launch = json.loads((control/'009-launch.json').read_text())
build = json.loads((here/'build-verification.json').read_text())
argv = launch['argv']
environment = launch['environment']
out = here / args.name
argv[0] = build['binary']
argv[4] = str(out / 'native')
argv[argv.index('--muscle-step-count') + 1] = str(args.steps)
if args.headless:
    argv.append('--mechanics-only')
environment['NUMI_HUMAN_LOCAL_BED_LAYERS'] = '0' if args.rigid else '1'
environment['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT'] = str(out/'common-failure.json')
materials = {
    'section': 'two quasistatic normal layers in series on registered weighted skin; common retardation 0.012 s',
    'values': 'skin 2.5 mm and 200 kPa; fat 10 mm and 55 kPa; effective engineering reference',
    'planned_gates': {
        'no_rejected_physical_root': True,
        'no_root_assistance': True,
        'sole_bed_force': 'same accepted articulation impulses, no FEM or secondary ground row',
        'late_bed_force_fraction_of_weight': [0.98, 1.02],
        'peak_geometric_compression_lt_m': .0125,
        'late_geometric_compression_lt_m': .00625,
        'terminal_com_speed_lt_m_s': .005,
        'viewer_measured_segment_realtime_factor_gte': .1
    },
    'readout': 'The uncompressed witness gap measures model shortening. Impulse divided by dt is the support force; multiplying by compliance is the elastic-equivalent compression, which contains a viscous contribution during transients.'
}
def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
pins = prior['controls']['009']['direct_asset_sha256'] | {
    p: v['sha256'] for p, v in prior['runtime_dependencies'].items()}
pins[argv[0]] = build['sha256']
shader = here / 'metalrobo/MetalRobo.metallib'
pins[str(shader)] = digest(shader)
skin = shader.with_name('HumanRestingSkin.metallib')
pins[str(skin)] = digest(skin)
for path, sha in pins.items():
    assert digest(path) == sha, path
assert shutil.disk_usage(here).free > 10 * 1024**3 + 400 * 1024**2
out.mkdir()
(out/'launch.json').write_text(json.dumps({
    'argv': argv, 'environment': environment, 'cwd': launch['cwd'], 'pins': pins,
    'design': materials, 'control': args.rigid, 'headless': args.headless,
}, indent=2) + '\n')
baseline = {key: value for key, value in os.environ.items() if not key.startswith(('NUMI_', 'NM_'))}
with (out / 'stdout.log').open('wb') as so, (out / 'stderr.log').open('wb') as se:
    start = time.monotonic()
    with subprocess.Popen(argv, env=baseline | environment, cwd=launch['cwd'], stdout=so, stderr=se) as process:
        stop = None
        while process.poll() is None:
            if shutil.disk_usage(out).free < 10 * 1024**3:
                stop = 'disk floor'
            if time.monotonic() - start > 320:
                stop = 'wall budget'
            if stop:
                process.terminate()
                break
            time.sleep(0.5)
        code = process.wait(timeout=20)
    result = {'exit_code': code, 'seconds_process': time.monotonic()-start, 'stopped': stop}
    (out/'execution.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result, flush=True)
raise SystemExit(code)
