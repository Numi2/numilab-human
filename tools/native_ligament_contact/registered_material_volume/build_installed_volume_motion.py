"""Run the pose-verified selected material via the existing native simulation owner."""
import json,sys,subprocess
from pathlib import Path
from emit_skin_source_registration import sha
src,out,app=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file() and f.suffix in ('.hpp','.metalinc','.metal') and not (out/f.name).exists():(out/f.name).hardlink_to(f)
(out/'metalrobo').mkdir(exist_ok=True)
for f in (src/'metalrobo').glob('*.metallib'):
    if not (out/'metalrobo'/f.name).exists():(out/'metalrobo'/f.name).hardlink_to(f)
n='numilab_human_myosim_visual_probe.mm';s=(src/n).read_text()
original='''const char* full=std::getenv("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_SOURCE_ONLY");
        if(!full||std::strcmp(full,"1")!=0){error="selected physical skin requires its full source volume";return false;}'''
replacement='''const char* install=std::getenv("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_INSTALL");
        const char* preflight=std::getenv("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT");
        if(!install||std::strcmp(install,"1")||!preflight||std::strcmp(preflight,"1")){
            error="selected physical skin requires installation and native preflight";return false;
        }'''
if s.count(original)!=1:raise ValueError('Unrecognized physical support seed')
(out/n).write_text(s.replace(original,replacement))
cmd=[[a.replace(str(src)+'/',str(out)+'/') for a in row] for row in json.loads((src/'build-commands.json').read_text())]
(out/'build-commands.json').write_text(json.dumps(cmd,indent=2)+'\n')
for row in cmd:subprocess.run(row,cwd=app/'build-native',check=True)
(out/'identity.json').write_text(json.dumps({str(p):sha(p) for p in [src/n,out/n,out/'numi-human-native',*(out/'metalrobo').glob('*.metallib')]},indent=2)+'\n')
