"""Captured offline preparation; no physical stepping or source-array copies."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path('/Users/n/human-cardiac-material-frame-preparation-20260912')
SOURCE = ROOT / 'source'
ASSET = Path('/Users/n/human-cardiac-wall-anatomy-20260912/asset-final')
PYTHON = '/Users/n/human-cardiac-partition-20260912/venv/bin/python'
MANIFEST_SHA = 'e8cb0391623577efc4eac04e5710cf7c9a4757614e09d936f5af3889c37d56f1'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(1 << 20):
            h.update(data)
    return h.hexdigest()

def sources():
    return {str(p.relative_to(SOURCE)): digest(p) for p in sorted(SOURCE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}

def inputs():
    manifest = json.loads((ASSET / 'manifest.json').read_bytes())
    return {name: digest(ASSET / name) for name in ['manifest.json', *sorted(manifest['buffers'])]}

def save(value):
    (ROOT / 'execution.json').write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

command = [PYTHON, '-m', 'numilab_human.cardiac_material_frames', '--asset', str(ASSET),
           '--output', str(ROOT / 'frames'), '--asset-manifest-sha256', MANIFEST_SHA,
           '--conversion-policy', 'normalize-fibre-gram-schmidt-sheet-v1']
record = {'schema': 'HumanPack.cardiac-material-frame-preparation-execution.v1',
          'command': command, 'cwd': str(SOURCE), 'environment': {'PYTHONPATH': str(SOURCE / 'src')},
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'source_before_sha256': sources(), 'input_before_sha256': inputs(),
          'runner_sha256': digest(Path(__file__)), 'physical_steps': 0, 'status': 'running'}
save(record)
start = time.monotonic()
with (ROOT / 'conversion.log').open('wb') as log:
    result = subprocess.run(command, cwd=SOURCE, env={**os.environ, 'PYTHONPATH': str(SOURCE / 'src')},
                            stdout=log, stderr=subprocess.STDOUT)
record.update({'elapsed_seconds': time.monotonic()-start, 'returncode': result.returncode,
               'peak_child_rss_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
               'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'source_after_sha256': sources(), 'input_after_sha256': inputs(),
               'log_sha256': digest(ROOT / 'conversion.log')})
record['inputs_unchanged'] = record['input_before_sha256'] == record['input_after_sha256']
record['implementation_unchanged'] = record['source_before_sha256'] == record['source_after_sha256']
if result.returncode == 0:
    record['outputs_sha256'] = {p.name: digest(p) for p in sorted((ROOT / 'frames').iterdir())}
record['status'] = 'pass' if result.returncode == 0 and record['inputs_unchanged'] and record['implementation_unchanged'] else 'fail'
save(record)
print(json.dumps({key: record[key] for key in ('status','returncode','elapsed_seconds','peak_child_rss_bytes')}, sort_keys=True))
raise SystemExit(0 if record['status'] == 'pass' else 1)
