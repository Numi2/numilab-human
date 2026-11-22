"""Keep the native opening mesh while the current release owns its temporary directory."""
import json, shutil, subprocess, sys, time
from pathlib import Path
from material_context import sha

case=Path(sys.argv[1]); run=case/'three'
job=json.loads((run/'launch.json').read_text());root=Path(job['argv'][4])
pack=root/'resting-human.mrvpack';captured=case/'opening.mrvpack'
while time.monotonic():
    if (run/'execution.json').exists():raise SystemExit('Native run ended before capturing opening')
    if pack.is_file() and 'human_execution_stage=native_horizon_begin' in (run/'stdout.log').read_text():break
    time.sleep(1)
h=sha(pack)
prior=case.parent/'free-interface'
if case!=prior and (prior/'opening-recovery.json').exists() and h==json.loads((prior/'opening.json').read_text())['pack']:
    receipt=json.loads((prior/'opening-recovery.json').read_text())
    z=prior/'opening.mrvpack.zst'
    if sha(z)!=receipt['verified_member_sha256'][z.name]:raise ValueError('Prior opening archive differs')
    subprocess.run(['cp','-c',str(z),str(case/z.name)],check=True)
    shutil.copyfile(prior/'opening-recovery.json',case/'opening-recovery.json')
else:
    subprocess.run(['cp','-c',str(pack),str(captured)],check=True)
    if h!=sha(captured):raise RuntimeError('Native opening changed during capture')
(case/'opening.json').write_text(json.dumps(dict(launch=sha(run/'launch.json'),pack=h,poses=sha(root.parent/'registered-initial-bodies.json')),indent=2)+'\n')
