import json
from pathlib import Path
import shlex
import subprocess

out=Path(__file__).resolve().parent
owner=out.parent.parent/'skin-fat-native-integration-209'
build=owner/'build-native'
cc=json.loads((build/'compile_commands.json').read_text())
donor=owner/'apps/NumiHumanRestingCoupling.hpp'
new=out/donor.name
source=donor.read_text()
assert source.count('#include "../matter/tools/human_resting_runtime.hpp"')==1
new.write_text(source.replace('#include "../matter/tools/human_resting_runtime.hpp"',
                              '#include "human_resting_runtime.hpp"'))

h=out/'human_resting_runtime.hpp'
text=h.read_text()
text=text.replace('[enc setBytes:&parameters.environment.x length:sizeof(float) atIndex:3];',
                   'const float pleuralReference=parameters.environment.x;\n'
                   '                [enc setBytes:&pleuralReference length:sizeof(pleuralReference) atIndex:3];')
h.write_text(text)
row=next(r for r in cc if r['file'].endswith('/matter/src/runtime.mm'))
arg=shlex.split(row['command'])
obj=out/'runtime.mm.o'
arg[arg.index('-o')+1]=str(obj)
assert arg[-1] == row['file']
arg[-1]=str(out/'runtime.mm')
arg[1:1]=['-I',str(out/'include'),'-iquote',str(owner/'matter/src')]
(out/'matter-build-command.json').write_text(json.dumps(arg,indent=2)+'\\n')
subprocess.run(arg,cwd=build,check=True)
