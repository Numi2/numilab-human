"""Admit the observed 44 active tiles with a deterministic 64-color sparse solve."""
import json,os,shutil,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from build_coupled_volume_direction import replace_once
from compile_connected_joint import digest
def build(base,out):
    out.mkdir();rt=out/'runtime';ap=out/'app';rt.mkdir();ap.mkdir()
    src=base/'metal/src';dst=out/'metal/src';(dst/'metal').mkdir(parents=True)
    for f in src.iterdir():
        if f.name=='metal':continue
        if f.name=='matter_kernels.metal':shutil.copy2(f,dst/f.name)
        else:(dst/f.name).symlink_to(f.resolve(),target_is_directory=f.is_dir())
    for f in (src/'metal').iterdir():
        if f.name!='fem.metalinc':(dst/'metal'/f.name).symlink_to(f.resolve())
    markers=['hints','claim','active']
    def widen_names(s):
        for name in markers:s=s.replace('nm_fem_registered_volume_'+name+'_incident','nm_fem_registered_volume_'+name+'_incident64')
        return s
    for folder,name in [(rt,'runtime.mm'),(ap,'app-runtime.mm')]:
        s=(base/folder.name/name).read_text()
        for a,b in [('volumeTileCount+33u','volumeTileCount+65u'),('33u*sizeof(uint32_t)','65u*sizeof(uint32_t)'),('sparse[0]<=32u','sparse[0]<=64u'),('0x636f6c6f72737472ull','0x636f6c6f72363472ull')]:
            if a not in s:raise ValueError(a)
            s=s.replace(a,b)
        (folder/name).write_text(widen_names(s))
    os.link(base/'runtime/bed_response.inc',rt/'bed_response.inc')
    s=(src/'metal/fem.metalinc').read_text()
    begin=s.index('kernel void nm_fem_registered_volume_claim_incident');end=s.index('inline MRCompensatedScalar registeredNeg',begin)
    body=s[begin:end]
    for a,b in [('index<32u','index<64u'),('live>32u','live>64u'),('sorted[32]','sorted[64]'),('color[32]','color[64]'),('uint used=0;','uint used[2]={0u,0u};'),('used|=1u<<color[j];','used[color[j]>>5u]|=1u<<(color[j]&31u);'),('color[i]=ctz(~used);','color[i]=used[0]!=0xffffffffu?ctz(~used[0]):32u+ctz(~used[1]);')]:
        body=replace_once(body,a,b)
    (dst/'metal/fem.metalinc').write_text(widen_names(s[:begin]+body+s[end:]))
    parent=json.loads((base/'build.json').read_bytes());app=json.loads((base/'app/build.json').read_bytes())
    cpu=[[v.replace(str(base)+'/',str(out)+'/')for v in c]for c in parent['commands']]
    metalbase=base.parent/'incident-diagnostic';mc=json.loads((metalbase/'build.json').read_bytes())
    gpu=[[v.replace(str(metalbase/'metal')+'/',str(out/'metal')+'/')for v in c]for c in mc['commands'][-2:]]
    def stream(key,cmds):
        with (out/(key+'.log')).open('w')as f:
            for c in cmds:subprocess.run(c,check=True,stdout=f,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(3)as pool:
        jobs=[pool.submit(stream,k,c) for k,c in [('metal',gpu),('shared',[cpu[0]]),('static',[cpu[2]])]]
        for task in jobs:task.result()
    stream('link',[cpu[1],cpu[3]])
    parent.update(commands=cpu+gpu,additional_scratch_bytes=260,runtime=digest(rt/'libmetalrobo.dylib'),host_source=digest(rt/'runtime.mm'),
        shader=digest(out/'metal/NumiMatter.metallib'),shader_source=digest(dst/'metal/fem.metalinc'))
    app.update(commands=cpu[2:],runtime=parent['runtime'],source=digest(ap/'app-runtime.mm'),app=digest(ap/'numi-human-native'))
    (out/'build.json').write_text(json.dumps(parent,indent=2)+'\n');(ap/'build.json').write_text(json.dumps(app,indent=2)+'\n')
if __name__=='__main__':build(*(Path(v).resolve() for v in sys.argv[1:]))
