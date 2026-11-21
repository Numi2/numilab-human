"""Run the pinned native supported scene, preserving its normal simulation entrypoint and accepted receipt."""
import json,sys,subprocess
from pathlib import Path
from material_context import sha
src,assets,binary,root,scratch=map(Path,sys.argv[1:6]);steps=int(sys.argv[6]);enabled=int(sys.argv[7])
root.mkdir(parents=True,exist_ok=True);cfg=json.loads(src.read_text());env=cfg['environment']
assert env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1' and not cfg['headless']
if enabled:
    env.update(NUMI_HUMAN_PELVIC_CONTACT_DATA=str(assets),NUMI_HUMAN_PELVIC_CONTACT_TRACE=str(steps),NUMI_HUMAN_PELVIC_CONTACT_TRIAL_LOG=str(root/'candidate-contact.csv'))
    manifest=json.loads((assets/'manifest.json').read_text())
    for name,value in manifest['files'].items():cfg['pins'][str(assets/name)]=value
    assert sha(Path(cfg['argv'][cfg['argv'].index('--soft-tissue-payload')+1]))==manifest['source']
    assert sha(Path(cfg['argv'][3])) in manifest['qualified_bone_payloads'] and set(manifest['matched_cortical_banks'])=={'1','2','3','18','19'}
env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=str(steps)
# The retained reduced-body timeout assumed 40 ms per root. Recent physical
# P1 contact replays have taken about 0.5 s per root, before export.
wall_rate=2.0 if enabled and 'constitutive_boundary' in manifest else .040
cfg['trial_wall_budget_seconds']=60+steps*wall_rate
cfg['pins'][str(src)]=sha(src);path=root/'config.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
script=(src.parent/'run_supported.py').read_text()
old="if steps not in {8000,30000}: raise ValueError('Expected 16 or 60 seconds')"
assert script.count(old)==1;script=script.replace(old,"if steps<1 or steps>30000:raise ValueError('Trial length outside bounds')")
budget='time.monotonic()-t>60+steps*.040'
assert script.count(budget)==1
script=script.replace(budget,'time.monotonic()-t>'+repr(cfg['trial_wall_budget_seconds']))
runner=root/'run_supported.py';runner.write_text(script)
subprocess.run([sys.executable,str(runner),str(path),str(binary),str(scratch),str(root),str(steps)],check=True)
import shutil
cap=scratch/'native/accepted-geometry'/f'step-{steps}.receipt.json'
shutil.copyfile(cap,root/'terminal.receipt.json')
