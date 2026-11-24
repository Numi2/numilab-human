"""Keep the registered Newton direction representably inside the unchanged material domain."""
import json,sys,subprocess,shutil
from pathlib import Path
from compile_connected_joint import digest

def build(root,out):
    src=root/'metal/src';dst=out/'metal/src';(dst/'metal').mkdir(parents=True)
    target='fem.metalinc'
    for f in src.iterdir():
        if f.name=='metal':continue
        if f.name=='matter_kernels.metal':shutil.copy2(f,dst/f.name)
        else:(dst/f.name).symlink_to(f.resolve(),target_is_directory=f.is_dir())
    for f in (src/'metal').iterdir():
        if f.name!=target:(dst/'metal'/f.name).symlink_to(f.resolve(),target_is_directory=f.is_dir())
    s=(src/'metal'/target).read_text()
    before='''            // Leave ten percent of the present distance to the material limit.
            target=.1f*current+.9f*(J<=lo?lo:hi);'''
    after='''            // Only the search direction is changed. At an exactly attained limit
            // a fraction of current slack is zero, so the old target could round
            // straight back to the limit. Leave room for Float32 evaluation of
            // the three signed triple products, including their cancellation.
            const float3 u=abs(F[0]), v=abs(F[1]), wAbs=abs(F[2]);
            const float magnitude=dot(u,v.yzx*wAbs.zxy+v.zxy*wAbs.yzx);
            const float limit=J<=lo?lo:hi;
            const float roundoff=16.f*0x1p-23f*max(abs(limit),magnitude);
            const float slack=min(.25f*(hi-lo),max(.1f*abs(current-limit),roundoff));
            target=J<=lo?lo+slack:hi-slack;
            if(!(target>lo && target<hi))return;'''
    if s.count(before)!=2:raise ValueError('Unexpected direction owner')
    (dst/'metal'/target).write_text(s.replace(before,after))
    cmds=[[w.replace(str(root/'metal/')+'/',str(out/'metal')+'/') for w in cmd] for cmd in json.loads((root/'build.json').read_bytes())['commands'][:2]]
    for cmd in cmds:subprocess.run(cmd,check=True)
    lib=out/'metal/NumiMatter.metallib'
    if not lib.exists():raise ValueError('Output library missing')
    (out/'build.json').write_text(json.dumps(dict(commands=cmds,metal=digest(lib),source=digest(dst/'metal'/target),baseline_source=digest(src/'metal'/target),baseline_metal=digest(root/'metal/NumiMatter.metallib')),indent=2)+'\n')

if __name__=='__main__':build(*(Path(v).resolve() for v in sys.argv[1:]))
