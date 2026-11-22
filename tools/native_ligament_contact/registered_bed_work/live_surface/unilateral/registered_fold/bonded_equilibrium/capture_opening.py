"""Keep the native opening mesh while the current release owns its temporary directory."""
import json, subprocess, sys, time
from pathlib import Path
from material_context import sha

case=Path(sys.argv[1]); run=case/'three'
job=json.loads((run/'launch.json').read_text());root=Path(job['argv'][4])
pack=root/'resting-human.mrvpack';captured=case/'opening.mrvpack'
while time.monotonic():
    if (run/'execution.json').exists():raise SystemExit('Native run ended before capturing opening')
    if pack.is_file() and 'human_execution_stage=native_horizon_begin' in (run/'stdout.log').read_text():break
    time.sleep(1)
subprocess.run(['cp','-c',str(pack),str(captured)],check=True)
h=sha(captured)
if h!=sha(pack):raise RuntimeError('Native opening changed during capture')
(case/'opening.json').write_text(json.dumps(dict(launch=sha(run/'launch.json'),pack=h,poses=sha(root.parent/'registered-initial-bodies.json')),indent=2)+'\n')
