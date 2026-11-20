"""Compile and replay the bounded settled-pose ligament candidate in isolation."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

here=Path.cwd()
base=here.parent/'resting-clearance-and-atrium-008/deep-pectoral-depth-build'
owner=Path.home()/'Work/skin-fat-native-integration-209'
build=here/'settled-ligament-preflight-002'
build.mkdir(exist_ok=False)
rows={}
for label in ('local','extended'):
    d=json.loads(Path(f'settled-ligament-{label}-candidates.json').read_text())
    rows.update(((v['name'],v['side']),v) for v in d['rest_pose_patch_candidates'] if not v.get('failed'))
assert len(rows)==8
catalog='NumiHumanRegisteredLigaments.hpp'
original=(owner/'apps'/catalog).read_text()
updated=original
for (name,side),row in rows.items():
    pattern=rf'^(  (?:else )?if\(n=="{name}"\) witness\()side==0\?(V3\{{[^}}]+\}}):(V3\{{[^}}]+\}}),side==0\?(V3\{{[^}}]+\}}):(V3\{{[^}}]+\}})\);$'
    found=re.search(pattern,updated,re.M)
    if not found:
        raise RuntimeError(name)
    lead,*parts=found.groups()
    for j,point in enumerate(row['seeds_m']):
        parts[2*j+side]='V3{'+','.join(format(x,'.17g') for x in point)+'}'
    updated=updated[:found.start()]+(f'{lead}side==0?{parts[0]}:{parts[1]},'
                                      f'side==0?{parts[2]}:{parts[3]});')+updated[found.end():]
(build/catalog).write_text(updated)
patch=''.join(difflib.unified_diff(original.splitlines(keepends=True),updated.splitlines(keepends=True),
                                   'a/apps/'+catalog,'b/apps/'+catalog))
Path('settled-ligament-candidate.patch').write_text(patch)
source='numilab_human_myosim_visual_probe.mm'
app='numi-human-native'
for path in (*base.glob('*.hpp'),base/'runtime.mm.o',base/source):
    os.link(path,build/path.name)
os.symlink(base/'metalrobo',build/'metalrobo',target_is_directory=True)
commands=json.loads((base/'build-commands.json').read_text())
paths={str(base/item):str(build/item) for item in (source,app,app+'.o','runtime.mm.o')}
commands=[[paths.get(a,a) for a in cmd] for cmd in commands]
(build/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
with open('settled-ligament-build.log','wb') as out:
    for cmd in commands:
        subprocess.run(cmd,cwd=owner/'build-native',stdout=out,stderr=subprocess.STDOUT,check=True)
executable=build/app
binary=(subprocess.check_output(['/usr/bin/otool','-L',str(executable)]).decode())
if '@rpath/libmetalrobo.dylib' not in binary:
    raise RuntimeError(binary)
setup=base.parent/'exact-trial-shader/NumiMatter.metallib'
assert setup.is_file()
d=json.loads(Path('prior-aligned-bounded-posture.json').read_text())
d['argv'][0]=str(executable)
d['argv'][4]=str(build/'native')
d['environment']['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(build/'failure.json')
d['pins'][str(executable)]=hashlib.sha256(executable.read_bytes()).hexdigest()
d['pins'][str(build/catalog)]=hashlib.sha256((build/catalog).read_bytes()).hexdigest()
for key in ('NUMI_MATTER_METALLIB','DYLD_LIBRARY_PATH'):
    if key in d['environment']: print(key,d['environment'][key],flush=True)
Path('settled-ligament-candidate-launch.json').write_text(json.dumps(d,indent=2)+'\n')
start=time.monotonic()
with open('settled-ligament-candidate.stdout','wb') as out,open('settled-ligament-candidate.stderr','wb') as err:
    try:
        code=subprocess.run(d['argv'],cwd=d['cwd'],env={**{k:v for k,v in os.environ.items()
            if not k.startswith(('NM_','NUMI_'))},**d['environment']},
            stdout=out,stderr=err,timeout=35).returncode
    except subprocess.TimeoutExpired:
        code=None
logs=Path('settled-ligament-candidate.stderr').read_text()
failures=[line for line in logs.splitlines() if line.startswith('registered ligament build failed ')]
r={'exit':code,'wall_s':time.monotonic()-start,'rejected_ligament_rows':failures}
Path('settled-ligament-candidate-execution.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
