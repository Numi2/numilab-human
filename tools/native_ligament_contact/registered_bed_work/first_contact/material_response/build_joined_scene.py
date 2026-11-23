"""Build the existing native scene against the exact registered skin/fat pose generation."""
from pathlib import Path
import json,subprocess,sys
from compile_connected_joint import digest

def build(base,registered,out):
    receipt=json.loads((base/'registration.json').read_text())
    ids=json.loads((registered/'registration.json').read_text())
    header='NumiHumanLoadedSkinPoseData.inc'
    if digest(registered/header)!=ids['header'] or digest(base/'numi-human-native')!=receipt['executable']:
        raise ValueError('Native source generation mismatch')
    out.mkdir()
    for path in sorted(base.iterdir()):
        if path.suffix in ('.mm','.hpp','.inc'):
            (out/path.name).hardlink_to((registered if path.name==header else base)/path.name)
    commands=[[a.replace(str(base)+'/',str(out)+'/') for a in cmd] for cmd in receipt['commands']]
    with (out/'build.log').open('wb') as log:
        for cmd in commands:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(receipt,commands=commands,executable=digest(out/'numi-human-native'),parent_executable=receipt['executable'],skin_pose=digest(registered/'registration.json'))
    (out/'registration.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':build(*map(lambda v:Path(v).resolve(),sys.argv[1:]))
