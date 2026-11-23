"""Admit typed source-face embeddings through the existing I/D sidecar reader."""
import json,shutil,sys,subprocess
from pathlib import Path
from compile_connected_joint import digest
from prepare_live_cortex import exact

def build(original,out):
    out.mkdir();src=original/'app';dst=out/'app';dst.mkdir()
    for f in src.iterdir():
        if f.suffix in ('.hpp','.inc','.mm','.h'):shutil.copy2(f,dst/f.name)
    file=dst/'NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp'
    before='''   if(sv!=std::int64_t(n)||sf!=-1||!parseNumber(v[30],sv)||sv<0||
      std::uint64_t(sv)>=currentSource->vertexCount)
    return fail("I/D current source row is outside its pinned domain");'''
    after='''   if(sf<0){
    if(sv!=std::int64_t(n)||!parseNumber(v[30],sv)||sv<0||
       std::uint64_t(sv)>=currentSource->vertexCount)
     return fail("I/D current source vertex is outside its pinned domain");
   }else if(sv!=-1||v[30]!="-1"||std::uint64_t(sf)>=currentSource->faceCount){
    return fail("I/D embedded source face is outside its pinned domain");
   }'''
    file.write_text(exact(file.read_text(),before,after))
    build=json.loads((original/'result.json').read_bytes());commands=[[v.replace(str(src)+'/',str(dst)+'/') for v in cmd] for cmd in build['commands']]
    with (out/'build.log').open('wb') as f:
        for cmd in commands:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
    (out/'build.json').write_text(json.dumps(dict(commands=commands,source=digest(src/file.name),result=digest(file),executable=digest(dst/'numi-human-native')),indent=2)+'\n')
if __name__=='__main__':build(*(Path(s).resolve() for s in sys.argv[1:]))
