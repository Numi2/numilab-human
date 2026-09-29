from pathlib import Path
import json,shutil,subprocess,time,hashlib
root=Path(__file__).resolve().parent;repo=root.parents[1]
out=root/'native';out.mkdir(exist_ok=True)
source=Path('/Users/home/numi-human-standing-20260922/apps/numilab_human_myosim_visual_probe.mm')
shutil.copy2(source,out/'probe.executed.mm')
old=repo/'Build/organ-family-coverage-20260929/native'
a=json.loads((old/'command.json').read_text())
for k in ['compile','link']:a[k]=[v.replace(str(old),str(out)) for v in a[k]]
(out/'command.json').write_text(json.dumps(a,indent=2)+'\n')
for k in ['compile','link']:
 t=time.monotonic();r=subprocess.run(a[k],cwd=a['cwd'],capture_output=True,text=True,timeout=240)
 for f,data in [('stdout',r.stdout),('stderr',r.stderr),('exit.code',str(r.returncode)+'\n'),('wall.seconds',str(time.monotonic()-t)+'\n')]:
  (out/(k+'.'+f)).write_text(data)
 print(k,r.returncode,flush=True)
 if r.returncode:print(r.stdout,r.stderr);raise SystemExit(r.returncode)
(out/'identity.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'probe.executed.mm',out/'myosim-visual-probe',repo/'Build/tendon-surface-binding-20260929/runtime/lib/libmetalrobo.dylib']},indent=2)+'\n')
