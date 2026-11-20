"""Test whether an extra Krylov restart reduces the final certified mechanical residual."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path.cwd()
run=root/'first-touch-krylov48-trial'
run.mkdir(exist_ok=False)
launch=json.loads((root/'registered-first-touch-trial/launch.json').read_text())
shader=root/'failure-owner-shader/NumiMatter.metallib'
with shader.open('rb') as stream:
    launch['pins'][str(shader)]=hashlib.file_digest(stream,'sha256').hexdigest()
for name, digest in launch['pins'].items():
    with open(name,'rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
cache=root/'registered-first-touch-trial/native/resting-human.mrvpack'
with cache.open('rb') as stream:
    cache_digest=hashlib.file_digest(stream,'sha256').hexdigest()
launch['pins'][str(cache)]=cache_digest
launch['environment'].update(
    NUMI_HUMAN_RESTING_FGMRES_BUDGET='48',
    NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE=str(cache),
    NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256=cache_digest,
)
launch['argv'][launch['argv'].index('--muscle-step-count')+1]='6'
launch['argv'][4]=str(run/'native')
launch['environment'].update(
    NUMI_MATTER_METALLIB_OVERRIDE=str(shader),
    NUMI_HUMAN_COUPLED_TRACE_CSV=str(run/'coupled.csv'),
    NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(run/'failure.json'),
)
(run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}
env.update(launch['environment'])
start=time.monotonic()
with (run/'stdout.log').open('wb') as out,(run/'stderr.log').open('wb') as err:
    try:
        code=subprocess.run(launch['argv'],cwd=launch['cwd'],env=env,stdout=out,stderr=err,timeout=320).returncode
    except subprocess.TimeoutExpired:
        code=None
result={'exit':code,'wall_s':time.monotonic()-start}
(run/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
