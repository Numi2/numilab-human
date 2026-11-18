from pathlib import Path
import json
import os
import shlex
import subprocess
import hashlib

out = Path(__file__).resolve().parent
owner = Path('/Users/numi/Work/skin-fat-native-integration-209')
build = owner / 'build-native'
source = out / 'numilab_human_myosim_visual_probe.mm'
obj = out / 'numi-human-native.o'
binary = out / 'numi-human-native'
shaders = build / 'shaders/MetalRobo.metallib'
assert hashlib.file_digest(shaders.open('rb'), 'sha256').hexdigest() == 'b52b5c1c5736da382682e4412aa9b458603066c0c5728a8869d02627aef97672'
cmds = json.loads((build / 'compile_commands.json').read_text())
cfg = next(row for row in cmds if row['file'].endswith(source.name)
           and 'CMakeFiles/numi-human-native.dir' in row['command'])
compile_argv = shlex.split(cfg['command'])
old_object = compile_argv[compile_argv.index('-o') + 1]
compile_argv[compile_argv.index('-o') + 1] = str(obj)
assert compile_argv[-2:] == ['-c', cfg['file']]
compile_argv[-1] = str(source)
compile_argv[1:1] = ['-DACCELERATE_NEW_LAPACK', '-iquote', str(owner/'apps')]
link_command = subprocess.check_output(
    ['/opt/homebrew/bin/ninja', '-t', 'commands', 'numi-human-native'],
    cwd=build, text=True).splitlines()[-1]
assert link_command.startswith(': && ') and link_command.endswith(' && :')
link_argv = shlex.split(link_command[5:-5])
link_argv += ['-framework', 'Accelerate']
link_argv[link_argv.index(old_object)] = str(obj)
link_argv[link_argv.index('-o') + 1] = str(binary)
(out / 'build-commands.json').write_text(json.dumps([compile_argv,link_argv],indent=2)+'\n')
for argv in (compile_argv,link_argv):
    subprocess.run(argv,cwd=build,check=True)
sha=hashlib.file_digest(binary.open('rb'),'sha256').hexdigest()
(out/'build-verification.json').write_text(json.dumps({'binary':str(binary),'sha256':sha,'bytes':binary.stat().st_size},indent=2)+'\n')
print(sha, flush=True)
