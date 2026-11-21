"""Add an opt-in donor partition preflight to the exact full-tissue native executable."""
import json,shutil,subprocess,sys
from pathlib import Path
from material_context import sha
parent,out,app=map(Path,sys.argv[1:])
build=out/'build';build.mkdir(parents=True,exist_ok=True)
for file in ('libmetalrobo.dylib','NumiMatter.metallib','metalrobo/HumanRestingSkin.metallib','metalrobo/MetalRobo.metallib'):
    target=build/file;target.parent.mkdir(exist_ok=True)
    if not target.exists():target.hardlink_to(parent/'build'/file)
for name in ('run_scene.py','strip.json'):shutil.copyfile(parent/name,out/name)
cmd=json.loads((parent/'build-commands.json').read_text());src=Path(cmd[0][-1]);s=src.read_text()
def inject(needle,addition):
    global s
    if s.count(needle)!=1:raise ValueError(needle)
    s=s.replace(needle,addition)
inject('#include "NumiHumanTissueMassMoment.hpp"','#include "NumiHumanTissueMassMoment.hpp"\n#include "NumiHumanPelvicMassPreflight.hpp"')
inject('    const auto shift=[&](auto& p,std::uint32_t b) {','''    if(const char* path=std::getenv("NUMI_HUMAN_PELVIC_MASS_PREFLIGHT");path&&*path)
        checkPelvicMassLedger(path,rebased,offset);
    const auto shift=[&](auto& p,std::uint32_t b) {''')
target=build/src.name;target.write_text(s)
header=Path(__file__).parent/'NumiHumanPelvicMassPreflight.hpp';shutil.copyfile(header,build/header.name)
cmd=[[item.replace(str(parent/'build')+'/',str(build)+'/') for item in row] for row in cmd]
cmd[0][1:1]=['-iquote',str(src.parent)]
(out/'build-commands.json').write_text(json.dumps(cmd,indent=2)+'\n')
for row in cmd:subprocess.run(row,check=True,cwd=app/'build-native')
(out/'identities.json').write_text(json.dumps({str(p):sha(p) for p in (src,target,header,build/'numi-human-native',build/'NumiMatter.metallib',build/'libmetalrobo.dylib')},indent=2)+'\n')
