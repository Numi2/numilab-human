"""Use the current bed admissibility in both the preflight and actual Newton measurement."""
import json
from pathlib import Path
import subprocess
import sys
from material_context import sha

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def once(s,a,b):
    if s.count(a)!=1:raise ValueError(('Unexpected native measurement site',a[:65],s.count(a)))
    return s.replace(a,b)
def build(root):
    old=root/'bonded-equilibrium/bed-stop';src=old/'build';exp=old/'paired';out=exp/'build';out.mkdir(exist_ok=True)
    f=read(old/'three/failure.json')
    if f['matter_object_index']!=4294967295 or f['matter_diagnostics'][2:]!=[1,0]:raise ValueError('Different preflight/solve witness')
    def host(s):
        head=s.index('                if (state.releaseVolumeMode && restartCycle==0u) {\n                    dispatchThreads("nm_fem_registered_release_bed_begin_gate"')
        end=s.index('                dispatchGroups32("nm_fgmres_begin_distributed_finish",',head)
        s=s[:head]+'                if (restartCycle==0u) qualifyRegisteredBedBegin();\n'+s[end:]
        k='        const auto dispatchGlobalMeasurement = '
        s=once(s,k,'''        const auto qualifyRegisteredBedBegin = [&] {
            if (!state.releaseVolumeMode) return;
            dispatchThreads("nm_fem_registered_release_bed_begin_gate",1u,[&] {
                [encoder setBuffer:state.releaseFailingRow offset:0 atIndex:0];
                [encoder setBuffer:state.fgmresMeasurementPartials offset:0 atIndex:1];
            });
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        };
'''+k)
        k='                dispatchGroups32(\n                    "nm_fgmres_begin_distributed_finish",'
        s=once(s,k,'                qualifyRegisteredBedBegin();\n'+k)
        flag='ull^candidate->releaseVolumeMode);'
        return once(s,flag,'ull^0x4245445052464c54ull^candidate->releaseVolumeMode);')
    for n in ('runtime.mm','app-runtime.mm'):(out/n).write_text(host((src/n).read_text()))
    meta=read(src/'receipt.json');commands=[]
    for c in meta['commands'][2:5]:
        cmd=[s.replace(str(src)+'/',str(out)+'/') if not s.startswith(str(src/'src')) else s for s in c]
        commands.append(cmd);subprocess.run(cmd,check=True)
    orig=old/'owner';owner=exp/'owner'
    if not owner.exists():subprocess.run(['cp','-cR',str(orig),str(owner)],check=True)
    ident=read(orig/'identities.json');parent=read(old/'parent/launch.json')
    cmd=[v.replace(str(src)+'/',str(out)+'/').replace(str(orig)+'/',str(owner)+'/') for v in ident['commands'][1]]
    commands.append(cmd);subprocess.run(cmd,cwd=parent['cwd'],check=True)
    exe=owner/'numi-human-native';binary=sha(exe);ident.update(binary=binary,binary_sha256=binary,binary_path=str(exe));ident['commands'][1]=cmd;write(owner/'identities.json',ident)
    lib='libmetalrobo.dylib'
    if parent['pins'].pop(str(src/lib))!=sha(src/lib):raise ValueError('Stale solver library')
    parent['environment']['DYLD_LIBRARY_PATH']=str(out);parent['pins'][str(out/lib)]=sha(out/lib)
    dest=exp/'parent';dest.mkdir(exist_ok=True);write(dest/'launch.json',parent)
    write(out/'receipt.json',dict(parent=sha(src/'receipt.json'),shader_library=sha(src/'NumiMatter.metallib'),runtime=sha(out/'runtime.mm'),library=sha(out/lib),binary=binary,commands=commands))
if __name__=='__main__':build(Path(sys.argv[1]).resolve())
