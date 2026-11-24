"""Seed registered volume directions from the last failed trial in each tile."""
from pathlib import Path
import json,sys,subprocess,shutil
from compile_connected_joint import digest

def build(base,out):
    src=base/'metal/src';dst=out/'metal/src';(dst/'metal').mkdir(parents=True)
    for f in src.iterdir():
        if f.name=='metal':continue
        if f.name=='matter_kernels.metal':shutil.copy2(f,dst/f.name)
        else:(dst/f.name).symlink_to(f.resolve(),target_is_directory=f.is_dir())
    for f in (src/'metal').iterdir():
        if f.name!='fem.metalinc':(dst/'metal'/f.name).symlink_to(f.resolve(),target_is_directory=f.is_dir())
    p=src/'metal/fem.metalinc';s=p.read_text()
    begin=s.index('    const float4 r=partials[owner*NM_MIXED_LINE_SEARCH_MAX_STEPS*maxTiles+tile];')
    end=s.index('    const uint e=rows[tile];',begin)
    code='''    // All tiles of a healthy previous Newton iteration have a passing
    // record by its accepted trial. Stop at that tile's first passing record.
    // Later slots may contain scratch from an older iteration. The physical
    // certificate still runs on newly reconstructed trial coordinates.
    for(uint trial=0u;trial<NM_MIXED_LINE_SEARCH_MAX_STEPS;++trial){
        const float4 r=partials[(owner*NM_MIXED_LINE_SEARCH_MAX_STEPS+trial)*maxTiles+tile];
        if(r.w==0.f)break;
        if(r.w!=2.f && r.w!=4.f)continue;
        const uint witness=as_type<uint>(r.x);
        if(witness>=object.elementOffset && witness-object.elementOffset<object.elementCount &&
           (witness-object.elementOffset)/256u==tile)rows[tile]=witness;
    }
'''
    (dst/'metal/fem.metalinc').write_text(s[:begin]+code+s[end:])
    b=json.loads((base/'build.json').read_bytes())
    cmds=[[word.replace(str(base/'metal')+'/',str(out/'metal')+'/') for word in cmd] for cmd in b['commands'][:2]]
    for cmd in cmds:subprocess.run(cmd,check=True)
    (out/'build.json').write_text(json.dumps(dict(commands=cmds,metal=digest(out/'metal/NumiMatter.metallib'),source=digest(dst/'metal/fem.metalinc'),parent_source=digest(p),baseline_metal=b['baseline_metal']),indent=2)+'\n')
if __name__=='__main__':build(*(Path(s).resolve() for s in sys.argv[1:]))
