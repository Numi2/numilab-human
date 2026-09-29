from pathlib import Path
import json,hashlib,os,time,subprocess,gzip
root=Path(__file__).resolve().parent;repo=root.parents[1];out=root/'tests.final';out.mkdir(exist_ok=True)
cmd=[str(repo/'.venv-mujoco312/bin/python'),'-m','pytest','-q','tests/test_whole_visceral_source_geometry.py','--basetemp',str(out/'work')]
env={**os.environ,'PYTHONPATH':'src:tests:Sources/myosim/checkout','OPENBLAS_NUM_THREADS':'1','MTL_DEBUG_LAYER':'1','NUMILAB_HUMAN_WHOLE_VISCERAL_EVIDENCE':str(root)}
(out/'command.json').write_text(json.dumps({'cwd':str(repo),'argv':cmd,'selected_evidence':str(root)},indent=2)+'\n')
files=['src/numilab_human/organ_family_geometry.py','src/numilab_human/lung_envelope.py','src/numilab_human/torso_anatomy_audit.py','tests/test_whole_visceral_source_geometry.py','tests/test_torso_anatomy_source_audit.py','config/source-organ-family-composite.v1.json','config/source-organ-family-composite.v2.json']
(out/'executed-source.json').write_text(json.dumps({f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
for f in files:
 p=out/'executed-source'/f;p.parent.mkdir(parents=True,exist_ok=True);p.with_suffix(p.suffix+'.gz').write_bytes(gzip.compress((repo/f).read_bytes(),mtime=0))
t=time.monotonic();r=subprocess.run(cmd,cwd=repo,env=env,capture_output=True,text=True)
for f,v in [('stdout',r.stdout),('stderr',r.stderr),('exit.code',str(r.returncode)+'\n'),('wall.seconds',str(time.monotonic()-t)+'\n')]: (out/f).write_text(v)
print(r.stdout,r.stderr);raise SystemExit(r.returncode)
