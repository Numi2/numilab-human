"""Build a recovery path for all-movable tetrahedra while retaining the original direction checks."""
import argparse,json,subprocess
from pathlib import Path
from compile_connected_joint import digest

def run(base,out):
    parent=json.loads((base/'build.json').read_bytes())
    src=base/'metal/src';dst=out/'metal/src';(dst/'metal').mkdir(exist_ok=True,parents=True)
    old=src/'metal/fem.metalinc'
    for f in [*src.iterdir(),*(src/'metal').iterdir()]:
        if not f.is_file() or f==old:continue
        link=dst/f.relative_to(src)
        if not link.exists():link.symlink_to(f.resolve())
    text=old.read_text()
    begin='''            // Adding a small correction to a large velocity'''
    stop='''            if(!(actual>lo && actual<hi))return 8u;'''
    # Restrict the change to the validated direction of the current linearization.
    s=text.index('inline uint registeredVolumeCellDirection')
    e=text.index('kernel void ',s);body=text[s:e]
    if any(body.count(k)!=1 for k in (begin,stop,'return 6u;')):raise ValueError('Unexpected registered direction owner')
    body=body.replace(begin,'            float3 beforeBalance[4];\n            for(uint k=0;k<4;++k)beforeBalance[k]=next[k];\n'+begin).replace(stop,(Path(__file__).parent/'represented_balance.inc').read_text().rstrip()).replace('return 6u;','return usedTranslationBalance?9u:6u;')
    shader=dst/'metal/fem.metalinc';shader.write_text(text[:s]+body+text[e:])
    cmds=[[a.replace(str(base/'metal')+'/',str(out/'metal')+'/') for a in cmd] for cmd in parent['commands']]
    with (out/'compile.log').open('w') as f:
        for cmd in cmds:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    new=out/'metal/NumiMatter.metallib'
    ident=dict(parent_shader=digest(base/'metal/NumiMatter.metallib'),original_source=digest(old),source=digest(shader),shader=digest(new),commands=cmds)
    if ident['parent_shader']!=parent['shader']:raise ValueError('Stale physical library')
    (out/'build.json').write_text(json.dumps(ident,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('base','out'):p.add_argument(n,type=lambda v:Path(v).resolve())
    run(**vars(p.parse_args()))
