from pathlib import Path
import json,subprocess,time,hashlib
root=Path('/Users/home/numilab-human')
old=json.load(open(root/'Build/source-topology-repair-20260929/captures.final/source-rest/command.json'))
out=root/'Build/right-choroid-laterality-repair-20260929/native/source-rest'
views=out/'views';views.mkdir(parents=True,exist_ok=True)
a=list(old['argv']);a[4]=str(views);a[a.index('--torso-anatomy-payload')+1]=str(root/'Build/right-choroid-laterality-repair-20260929/payload/right-choroid-laterality-candidate.nhanatomy');a.extend(['--hidden-anatomy-stable-id','587'])
(out/'command.json').write_text(json.dumps({'cwd':str(root),'argv':a},indent=2)+'\n')
t=time.monotonic()
p=subprocess.run(a,cwd=root,capture_output=True,text=True)
(out/'stdout.txt').write_text(p.stdout);(out/'stderr.txt').write_text(p.stderr);(out/'exit.code').write_text(str(p.returncode)+'\n');(out/'wall.seconds').write_text(str(time.monotonic()-t)+'\n')
print(json.dumps({'exit':p.returncode,'wall_seconds':time.monotonic()-t,'files':[x.name for x in views.iterdir()]}),flush=True)
raise SystemExit(p.returncode)
