"""Rebuild the full registered Human with the same source-qualified cardiac library."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
prior,heart,cwd,target=map(Path,sys.argv[1:]);dest=target/'build'
dest.mkdir(parents=True,exist_ok=True)
for name in ('NumiMatter.metallib','libmetalrobo.dylib'):
    (dest/name).hardlink_to(prior/'build'/name)
(other:=dest/'metalrobo').mkdir()
for name in ('HumanRestingSkin.metallib','MetalRobo.metallib'):
    (other/name).hardlink_to(heart/'metalrobo'/name)
for name in ('run_scene.py','strip.json'):shutil.copyfile(prior/name,target/name)
cmds=json.loads((prior/'commands.json').read_text())['human']
def change(arg):
    if arg.startswith('-DNUMI_HUMAN_RESPIRATION_METALLIB='):return '-DNUMI_HUMAN_RESPIRATION_METALLIB="'+str(other/'HumanRestingSkin.metallib')+'"'
    if arg in (str(prior/'build/numi-human-native.o'),str(prior/'build/numi-human-native')):return str(dest/Path(arg).name)
    return arg
updated=[[change(a) for a in cmd] for cmd in cmds]
(target/'build-commands.json').write_text(json.dumps(updated,indent=2)+'\n')
for cmd in updated:subprocess.run(cmd,cwd=cwd/'build-native',check=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
(target/'identities.json').write_text(json.dumps({str(p):sha(p) for p in [dest/'numi-human-native',dest/'libmetalrobo.dylib',dest/'NumiMatter.metallib',other/'HumanRestingSkin.metallib',other/'MetalRobo.metallib',prior/'build/numilab_human_myosim_visual_probe.mm']},indent=2)+'\n')
