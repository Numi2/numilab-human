"""Reproduce a pinned short native resting control on SSH Numi."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('control', choices=('005', '009'))
parser.add_argument('output', type=Path, nargs='?',
                    help='New directory for the native run and logs')
parser.add_argument('--verify', action='store_true',
                    help='Check the archived input identities without running')
args = parser.parse_args()

here = Path(__file__).resolve().parent
manifest = json.loads((here / 'metrics.json').read_text())
run = manifest['controls'][args.control]
launch = json.loads((here / run['launch_file']).read_text())
argv = launch['argv']

pins = dict(run['direct_asset_sha256'])
pins.update((path, entry['sha256']) for path, entry in manifest['runtime_dependencies'].items())
pins[argv[0]] = run['binary_sha256']
for path, sha256 in pins.items():
    with Path(path).open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != sha256:
        parser.exit(1, f'Input hash mismatch for {path}\n')
if args.verify:
    print(f'Confirmed {len(pins)} pinned runtime and asset hashes for control {args.control}.')
    raise SystemExit(0)
if args.output is None:
    parser.error('provide a new output directory')
out = args.output.resolve()
if shutil.disk_usage(out.parent).free < 11 * 1024**3:
    parser.error('retain at least 11 GiB free before rerunning this short control')
out.mkdir()

argv[4] = str(out / 'native')
settings = launch['environment']
settings['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT'] = str(out / 'common-failure.json')
if '--resting-movie' in argv:
    argv[argv.index('--resting-movie') + 1] = str(out / 'native-viewer.mov')
env = {key: value for key, value in os.environ.items() if not key.startswith(('NUMI_', 'NM_'))}
env.update(settings)
with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
    run = subprocess.run(argv, cwd=launch['cwd'], env=env, stdout=stdout,
                         stderr=stderr, timeout=420)
raise SystemExit(run.returncode)
