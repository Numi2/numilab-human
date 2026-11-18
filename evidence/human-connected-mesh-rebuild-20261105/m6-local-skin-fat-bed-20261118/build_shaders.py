import hashlib
import json
import shlex
import subprocess
from pathlib import Path

out = Path(__file__).resolve().parent
owner = Path('/Users/numi/Work/skin-fat-native-integration-209')
build = owner / 'build-native'
source = owner / 'src/metal'
commands = subprocess.check_output(
    ['/opt/homebrew/bin/ninja', '-t', 'commands', 'shaders/MetalRobo.metallib'],
    cwd=build, text=True).splitlines()[-1].split(' && ')
air = build / 'shaders/MetalRoboNumiHumanStand.air'
compile_argv = next(shlex.split(item) for item in commands
                    if 'metal ' in item and 'NumiHumanStand.metal' in item)
assert compile_argv[-4:] == ['-c', str(source/'NumiHumanStand.metal'), '-o', str(air)]
compile_argv[-4:] = ['-I', str(source), '-c', str(out/'NumiHumanStand.metal'),
                     '-o', str(out / air.name)]
link = next(shlex.split(item) for item in commands if ' metallib ' in item)
baseline = build / 'shaders/MetalRobo.metallib'
assert link[-2:] == ['-o', str(baseline)]
lib = out / 'metalrobo' / baseline.name
lib.parent.mkdir(exist_ok=True)
link[-1] = str(out / 'relinked-control.metallib')
subprocess.run(link, cwd=build, check=True)
def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
original = digest(baseline)
control = digest(out/'relinked-control.metallib')
print('original', original, 'relinked-control', control, flush=True)
assert control == original, 'relink would alter unrelated shader objects'
link[link.index(str(air))] = str(out / air.name)
link[-1] = str(lib)
(out/'shader-build-commands.json').write_text(json.dumps([compile_argv, link], indent=2)+'\n')
subprocess.run(compile_argv, cwd=build, check=True)
subprocess.run(link, cwd=build, check=True)
print('local-bed', digest(lib), flush=True)
