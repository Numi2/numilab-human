"""Replay the same settled pose and its accepted-state guards with full-contact residual descent."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path.cwd()
run = root / 'full-residual-armijo'
run.mkdir(exist_ok=False)
launch = json.loads((root / 'settled-ligament-candidate-launch.json').read_text())
for name, digest in launch['pins'].items():
    with open(name, 'rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
launch['argv'][4] = str(run / 'native')
launch['environment']['NUMI_HUMAN_COUPLED_TRACE_CSV'] = str(run / 'coupled.csv')
launch['environment']['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT'] = str(run / 'failure.json')
launch['environment']['NM_FEM_FRESH_DIRECTION_ALPHA_RESET'] = '1'
launch['environment']['NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT'] = 'all'
(run / 'launch.json').write_text(json.dumps(launch, indent=2)+'\n')
env = {k:v for k,v in os.environ.items() if not k.startswith(('NUMI_', 'NM_'))}
env.update(launch['environment'])
start = time.monotonic()
with (run/'stdout.log').open('wb') as out, (run/'stderr.log').open('wb') as err:
    try:
        code = subprocess.run(launch['argv'], cwd=launch['cwd'], env=env, stdout=out, stderr=err, timeout=235).returncode
    except subprocess.TimeoutExpired:
        code = None
result = {'exit':code, 'wall_s':time.monotonic()-start}
(run/'execution.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
