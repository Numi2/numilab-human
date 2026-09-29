from pathlib import Path
import subprocess,json,hashlib,os,time
root=Path(__file__).resolve().parent;repo=root.parents[1];dest=root/'tests.final';dest.mkdir(exist_ok=True)
cmd=[str(repo/'.venv-mujoco312/bin/python'),'-m','pytest','-q','tests/test_organ_family_geometry.py','tests/test_lung_envelope.py','--basetemp',str(dest/'work')]
env={**os.environ,'PYTHONPATH':'src:tests:Sources/myosim/checkout','OPENBLAS_NUM_THREADS':'1','MTL_DEBUG_LAYER':'1',
 'NUMILAB_HUMAN_ORGAN_FAMILY_EVIDENCE':str(root),'NUMILAB_HUMAN_LUNG_ENVELOPE_EVIDENCE':str(root.parent/'lung-envelope-20260929')}
(dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
files=['src/numilab_human/organ_family_geometry.py','src/numilab_human/lung_envelope.py','src/numilab_human/torso_anatomy_audit.py','tests/test_organ_family_geometry.py','tests/test_lung_envelope.py','tests/test_torso_anatomy_source_audit.py','config/source-organ-family-composite.v1.json']
(dest/'executed-source.json').write_text(json.dumps({f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
start=time.monotonic();r=subprocess.run(cmd,cwd=repo,env=env,capture_output=True,text=True)
for file,data in [('stdout',r.stdout),('stderr',r.stderr),('exit.code',str(r.returncode)+'\n'),('wall.seconds',str(time.monotonic()-start)+'\n')]:
 (dest/file).write_text(data)
print(r.stdout,r.stderr);raise SystemExit(r.returncode)
