"""Run the existing native admission with disk-backed temporary outputs, preserving every input."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from evaluate_registered_bed_transfer import digest
config, selection, binary, out = map(lambda x: Path(x).resolve(),sys.argv[1:])
out.mkdir(exist_ok=False)
original=json.loads(config.read_text());j=json.loads(config.read_text())
old=Path(j['argv'][4]).parent
assert old==Path('/Volumes/NumiVolumeAdmission') and not any(p.startswith(str(old)+'/') for p in j['pins'])
prepared=out/'source';prepared.mkdir()
j['argv'][4]=str(prepared/'native')
for key in ('NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP','NUMI_HUMAN_SKIN_REFERENCE_DUMP','NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT','NUMI_HUMAN_COUPLED_TRACE_CSV'):
    j['environment'][key]=str(prepared/Path(j['environment'][key]).relative_to(old))
assert not any('/Volumes/NumiVolumeAdmission' in v for v in j['environment'].values())
assert j['pins']==original['pins'] and j['environment']['DYLD_LIBRARY_PATH']==original['environment']['DYLD_LIBRARY_PATH']
source=prepared/'config.json';source.write_text(json.dumps(j,indent=2)+'\n')
with tempfile.TemporaryDirectory(prefix='native-volume-',dir=out) as temp:
    tmp=Path(temp)
    env=os.environ.copy();env.update(NUMI_SELECTED_TRIAL_MODE='motion',NUMI_SELECTED_RETAIN_TERMINAL_PACK='1',
        NUMI_HUMAN_REGISTERED_CELL_NODE_PREFIX=str(out/'physical'))
    cmd=[sys.executable,str(Path(__file__).with_name('admit_current_volume.py')),str(source),str(selection),str(binary),temp,str(out)]
    start=shutil.disk_usage(out).free
    proc=subprocess.run(cmd,env=env)
    files={}
    for path in ('coupled.csv','failure.json','registered-initial-bodies.json'):
        x=tmp/path
        if x.exists():
            shutil.copyfile(x,out/path);files[path]=digest(out/path)
    (out/'scratch.json').write_text(json.dumps({'disk_backed':True,'inputs_preserved':True,'original_config_sha256':digest(config),'free_at_start':start,
        'external_outputs':files,'launcher_status':proc.returncode},indent=2)+'\n')
sys.exit(proc.returncode)
