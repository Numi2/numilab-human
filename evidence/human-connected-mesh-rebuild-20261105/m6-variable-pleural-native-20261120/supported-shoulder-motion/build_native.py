"""Relink the current guarded native scene using the bilateral containment check."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

p,d=map(Path,sys.argv[1:])
base=p.parent/'resting-clearance-and-atrium-015/staged-solver-accuracy-build'
scene=p/'floor-scene-build'; guarded=p/'guarded-runtime-admission-build'
c=d/'candidate'
build_dir=Path.home()/'Work/skin-fat-native-integration-209/build-native'
name='numilab_human_myosim_visual_probe.mm'
shutil.copy2(scene/name,c/name)
commands=json.loads((base/'build-commands.json').read_text())
compile_cmd=[str(c/Path(arg).name) if Path(arg) in {base/name,base/'numi-human-native.o'} else arg for arg in commands[0]]
compile_cmd[1:1]=['-iquote',str(scene),'-iquote',str(base)]
link=[str(guarded/'app-runtime.mm.o') if arg.endswith('/runtime.mm.o') else
      str(c/Path(arg).name) if Path(arg) in {base/'numi-human-native',base/'numi-human-native.o'} else
      str(scene/Path(arg).name) if Path(arg).parent==base else arg for arg in commands[-1]]
(c/'build-commands.json').write_text(json.dumps([compile_cmd,link],indent=2)+'\n')
for cmd in (compile_cmd,link): subprocess.run(cmd,check=True,cwd=build_dir)
