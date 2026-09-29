from pathlib import Path
import hashlib,json,subprocess,os,time
root=Path(__file__).resolve().parent
repo=root.parents[1]
py=repo/'.venv-mujoco312/bin/python'
env={**os.environ,'PYTHONPATH':'src:tests:Sources/myosim/checkout','OPENBLAS_NUM_THREADS':'1','MTL_DEBUG_LAYER':'1'}
def run(cmd,dest):
 dest.mkdir(parents=True,exist_ok=True)
 (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 start=time.monotonic();r=subprocess.run(cmd,cwd=repo,env=env,capture_output=True,text=True,timeout=240)
 for file,data in [('stdout',r.stdout),('stderr',r.stderr),('exit.code',str(r.returncode)+'\n'),('wall.seconds',str(time.monotonic()-start)+'\n')]:
  (dest/file).write_text(data)
 print(dest.relative_to(root),r.returncode,flush=True)
 if r.returncode:print(r.stdout,r.stderr);raise SystemExit(r.returncode)
common=['--sources',str(repo/'Sources'),'--artifact',str(repo/'Build/myosim-fullbody'),'--registration',str(repo/'Build/knee-parity-registration-20260929/candidate.v6.registration.json'),'--base-payload',str(repo/'Build/lung-envelope-20260929/payload/thorax-lung-envelope.nhanatomy')]
run([str(py),'-m','numilab_human.organ_family_geometry','compose',*common,'--output',str(root/'payload.final')],root/'compose.final')
base=json.loads((root.parent/'lung-source-coverage-20260929/projected-neutral/command.json').read_text())
base[0]=str(root/'native/myosim-visual-probe');base[base.index('--torso-anatomy-payload')+1]=str(root/'payload.final/source-organ-family-anatomy.nhanatomy')
for name,pose in [('raw-source-rest',None),('projected-neutral',()),('torso-flexion',((7,-.4),(8,.1),(9,.2)))]:
 for mask in [63,256,512,1023]:
  dest=root/'final-native'/f'{name}-mask{mask}';cmd=list(base);cmd[4]=str(dest/'views')
  if pose is None:del cmd[cmd.index('--joint-equality-payload'):]
  for q,v in pose or ():cmd+=['--pose-q',str(q),str(v)]
  cmd+=['--torso-anatomy-layer-mask',str(mask)];run(cmd,dest)
