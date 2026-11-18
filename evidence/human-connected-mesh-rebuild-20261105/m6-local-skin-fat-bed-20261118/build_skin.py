from pathlib import Path
import json
import shlex
import subprocess

out = Path(__file__).resolve().parent
owner = Path('/Users/numi/Work/skin-fat-native-integration-209')
build = owner/'build-native'
target = 'matter/shaders/HumanRespiration.metallib'
row = subprocess.check_output(['/opt/homebrew/bin/ninja', '-t', 'commands', target],
                              cwd=build, text=True).splitlines()[-1].split(' && ')
compile_argv = shlex.split(row[1])
assert compile_argv[-4] == '-c'
compile_argv[-4:] = ['-I', str(owner/'matter/src'), '-c',
                    str(out/'human_respiration.metal'), '-o', str(out/'HumanRestingSkin.air')]
link_argv = shlex.split(row[2])
link_argv[-3:] = [str(out/'HumanRestingSkin.air'), '-o',
                  str(out/'metalrobo/HumanRestingSkin.metallib')]
(out/'skin-build-commands.json').write_text(json.dumps([compile_argv,link_argv],indent=2)+'\n')
for argv in (compile_argv,link_argv):
    subprocess.run(argv,cwd=build,check=True)
