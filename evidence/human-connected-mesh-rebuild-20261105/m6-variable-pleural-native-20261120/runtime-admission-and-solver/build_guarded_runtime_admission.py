"""Build the Human/Matter runtime with transactional initialization diagnostics."""
import difflib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
prior = root / 'pressure-with-abdominal-cavity-build'
app_baseline = root.parent / 'resting-clearance-and-atrium-015/staged-solver-accuracy-build'
original_scene = root / 'floor-scene-build'
out = root / 'guarded-runtime-admission-build'
out.mkdir(exist_ok=True)
directory = Path.home() / 'Work/skin-fat-native-integration-209/build-native'
early = '        diagnostics.encoded = true;\n        diagnostics.residentBytes = candidate->residentBytes;'
ready = '        state_ = std::move(candidate);\n        return diagnostics;'

for name in ('app-runtime.mm', 'runtime.mm'):
    text = (prior / name).read_text()
    assert text.count(early) == 1 and text.count(ready) == 1
    edited = text.replace(early, '        diagnostics.residentBytes = candidate->residentBytes;')
    edited = edited.replace(ready, '        state_ = std::move(candidate);\n'
                            '        diagnostics.encoded = true;\n        return diagnostics;')
    (out / name).write_text(edited)
    patch = difflib.unified_diff(text.splitlines(keepends=True), edited.splitlines(keepends=True),
                                fromfile='a/matter/src/runtime.mm', tofile='b/matter/src/runtime.mm', n=0)
    (root / f'guarded-admission-{name}.patch').write_text(''.join(patch))

shared = json.loads((prior / 'build-commands.json').read_text())
def relocated(command, names):
    return [str(out / Path(p).name) if Path(p).parent == prior and Path(p).name in names
            else p for p in command]
shared_compile = next(command for command in shared if str(prior / 'runtime.mm') in command)
shared_link = next(command for command in shared if str(prior / 'libmetalrobo.dylib') in command)
for command in (
    relocated(shared_compile, {'runtime.mm', 'runtime.mm.o'}),
    relocated(shared_link, {'runtime.mm.o', 'libmetalrobo.dylib'}),
):
    subprocess.run(command, cwd=directory, check=True)
older_static = root.parent / 'resting-clearance-and-atrium-007/shared-fascia-pressure-build'
compile_app = json.loads((older_static / 'build-commands.json').read_text())[0]
compile_app = [str(out / ('app-' + Path(p).name))
               if Path(p).parent == older_static and Path(p).name in {'runtime.mm', 'runtime.mm.o'}
               else p for p in compile_app]
subprocess.run(compile_app, cwd=directory, check=True)
link_app = json.loads((app_baseline / 'build-commands.json').read_text())[-1]
link_app = [str(out / 'app-runtime.mm.o') if p.endswith('/runtime.mm.o') else
            str(out / Path(p).name) if Path(p).parent == app_baseline and
            Path(p).name == 'numi-human-native' else
            str(original_scene / Path(p).name) if Path(p).parent == app_baseline else p
            for p in link_app]
subprocess.run(link_app, cwd=directory, check=True)
for shader in original_scene.glob('*.metallib'):
    target = out / shader.name
    if not target.exists():
        target.symlink_to(shader)
print('Patched static and dynamic owners linked')
