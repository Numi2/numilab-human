from pathlib import Path
import json,subprocess,time
root=Path('/Users/home/numilab-human')
old=json.load(open(root/'Docs/media/right-choroid-laterality-20260929/native/command.json'))
out=root/'Build/right-eye-laterality-cleanup-20260929/native/source-rest'
views=out/'views';views.mkdir(parents=True,exist_ok=True)
a=list(old['argv']);a[4]=str(views);a[a.index('--torso-anatomy-payload')+1]=str(root/'Build/right-eye-laterality-cleanup-20260929/payload/right-eye-laterality-candidates.nhanatomy')
for sid in [413,426,427]:a.extend(['--hidden-anatomy-stable-id',str(sid)])
(out/'command.json').write_text(json.dumps({'cwd':str(root),'argv':a},indent=2)+'\n')
t=time.monotonic();p=subprocess.run(a,cwd=root,capture_output=True,text=True);elapsed=time.monotonic()-t
(out/'stdout.txt').write_text(p.stdout);(out/'stderr.txt').write_text(p.stderr);(out/'exit.code').write_text(str(p.returncode)+'\n');(out/'wall.seconds').write_text(str(elapsed)+'\n')
print(json.dumps({'exit':p.returncode,'wall_seconds':elapsed,'files':[x.name for x in views.iterdir()]}),flush=True)
raise SystemExit(p.returncode)
