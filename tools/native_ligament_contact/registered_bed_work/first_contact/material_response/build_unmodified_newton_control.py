"""Keep both geometry certificates active and test the unmodified FGMRES correction."""
from pathlib import Path
import json,sys,shutil,subprocess
from compile_connected_joint import digest

def build(base,out):
    src=base/'metal/src';dst=out/'metal/src';(dst/'metal').mkdir(parents=True)
    for p in src.iterdir():
        if p.name=='metal':continue
        if p.name=='matter_kernels.metal':shutil.copy2(p,dst/p.name)
        else:(dst/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
    for p in (src/'metal').iterdir():
        if p.name!='fem.metalinc':(dst/'metal'/p.name).symlink_to(p.resolve())
    f=src/'metal/fem.metalinc';s=f.read_text()
    start=s.index(' {\n',s.index('kernel void nm_fem_registered_volume_active('))+3
    end=s.index('\n}\n',start)
    s=s[:start]+'    // Control experiment: the unmodified Newton proposal still passes all geometric and residual gates.\n    return;'+s[end:]
    (dst/'metal/fem.metalinc').write_text(s)
    cmds=[[w.replace(str(base/'metal')+'/',str(out/'metal')+'/')for w in c]for c in json.loads((base/'build.json').read_bytes())['commands'][:2]]
    for cmd in cmds:subprocess.run(cmd,check=True)
    (out/'build.json').write_text(json.dumps(dict(commands=cmds,baseline_metal=digest(base/'metal/NumiMatter.metallib'),metal=digest(out/'metal/NumiMatter.metallib'),parent_source=digest(f),source=digest(dst/'metal/fem.metalinc')),indent=2)+'\n')
if __name__=='__main__':build(*(Path(s).resolve() for s in sys.argv[1:]))
