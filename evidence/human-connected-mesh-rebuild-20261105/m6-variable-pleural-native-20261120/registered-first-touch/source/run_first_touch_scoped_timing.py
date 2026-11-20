"""Record the exact rejecting GPU certificate on the weighted first-touch launch."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path.cwd()
run=root/'first-touch-scoped-timing-trial'
run.mkdir(exist_ok=False)
launch=json.loads((root/'registered-first-touch-trial/launch.json').read_text())
shader=root/'failure-owner-shader/NumiMatter.metallib'
with shader.open('rb') as stream:
    launch['pins'][str(shader)]=hashlib.file_digest(stream,'sha256').hexdigest()
for name, digest in launch['pins'].items():
    with open(name,'rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
library=root/'scoped-exact-stage-runtime/libmetalrobo.dylib'
with library.open('rb') as stream:
    launch['pins'][str(library)]=hashlib.file_digest(stream,'sha256').hexdigest()
launch['environment'].update(
    DYLD_LIBRARY_PATH=str(library.parent), NUMI_MATTER_GPU_TIMING='1',
    NUMI_MATTER_GPU_TIMING_STAGES='1', NUMI_MATTER_GPU_TIMING_ROOT='4',
    NUMI_MATTER_GPU_TIMING_MICROTICK='63',
)
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
