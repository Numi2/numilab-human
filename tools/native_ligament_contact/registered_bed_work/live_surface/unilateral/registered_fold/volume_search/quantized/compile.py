"""Compile the conservative Float32 publication of the volume-limited search step."""
from pathlib import Path
import json,subprocess,sys
from material_context import sha
def build(case):
    p=case/'quantized';out=p/'build';out.mkdir(exist_ok=True)
    src=case/'build';m=json.loads((src/'receipt.json').read_text())
    if sha(case/'direction.metalinc')!=m['source_sha256']:raise ValueError('Wrong parent shader')
    if not (out/'src').exists():subprocess.run(['cp','-cR',src/'src',out/'src'],check=True)
    f=out/'src/metal/fem.metalinc';s=(src/'src/metal/fem.metalinc').read_text()
    marker='            for(uint k=0;k<4;++k)direction[t.nodes[k]].xyz=next[k];'
    if s.count(marker)!=1:raise ValueError('Unexpected publication site')
    f.unlink();f.write_text(s.replace(marker,(p/'publish.inc').read_text()+'\n'+marker))
    commands=[[a.replace(str(src)+'/',str(out)+'/') for a in c] for c in m['commands'][:2]]
    for c in commands:subprocess.run(c,check=True)
    launch=json.loads((case/'parent/launch.json').read_text());env=launch['environment'];key='NUMI_MATTER_METALLIB_OVERRIDE';old=env[key]
    if launch['pins'].pop(old)!=m['metallib']:raise ValueError('Wrong parent Metal library')
    env[key]=str(out/'NumiMatter.metallib');launch['pins'][env[key]]=sha(out/'NumiMatter.metallib')
    (p/'parent').mkdir(exist_ok=True);(p/'parent/launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    (out/'receipt.json').write_text(json.dumps(dict(source=sha(f),parent=sha(src/'receipt.json'),commands=commands,metallib=sha(out/'NumiMatter.metallib')),indent=2)+'\n')
if __name__=='__main__':build(Path(sys.argv[1]).resolve())
