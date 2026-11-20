"""Tag competing Matter failure producers without relaxing any physical gate."""
import difflib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
base = root.parent/'resting-clearance-and-atrium-008/exact-trial-shader'
build = root/'failure-owner-shader'
build.mkdir(exist_ok=False)
for entry in (base/'src').rglob('*'):
    target = build/entry.relative_to(base)
    if entry.is_dir():
        target.mkdir(parents=True,exist_ok=True)
    elif entry.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(entry.resolve(),target)
changes = {
    'fgmres.metalinc': (
        'float4(alpha, objectAlpha, 0.0f, 0.0f)', 'float4(alpha, objectAlpha, -901.0f, 0.0f)'),
    'mixed_fem.metalinc': (
        'residualValid && correctionFinite && contactValid\n                    ? transport\n                    : float4(relativeResidual, relativeCorrection,\n                             volumeResidual, pressureResidual)',
        'residualValid && correctionFinite && contactValid\n                    ? transport\n                    : float4(relativeResidual, relativeCorrection,\n                             volumeResidual, -902.0f)')
}
patches = []
for name, (old, new) in changes.items():
    path = build/'src/metal'/name
    original = path.read_text()
    assert original.count(old)==1, (name, original.count(old))
    updated = original.replace(old,new)
    path.unlink()
    path.write_text(updated)
    patches.extend(difflib.unified_diff(original.splitlines(keepends=True),updated.splitlines(keepends=True),
                                        'a/matter/src/metal/'+name,'b/matter/src/metal/'+name))
(root/'failure-owner-shader.patch').write_text(''.join(patches))
commands = json.loads((base/'build-commands.json').read_text())
air = build/'failure-owner.air'
paths = {str(base/'src/matter_kernels.metal'):str(build/'src/matter_kernels.metal'),
         commands[0][-1]:str(air),str(base/'NumiMatter.metallib'):str(build/'NumiMatter.metallib')}
commands = [[paths.get(a,a) for a in cmd] for cmd in commands]
(build/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
with (root/'failure-owner-shader-build.log').open('wb') as log:
    for cmd in commands:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
air.unlink()
