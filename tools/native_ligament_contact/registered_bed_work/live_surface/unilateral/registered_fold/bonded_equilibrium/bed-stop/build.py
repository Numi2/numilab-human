"""Share the existing final Signorini/Coulomb condition with the distributed Newton latch."""
from pathlib import Path
import json, subprocess, sys
from material_context import sha

def replace(s,old,new):
    if s.count(old)!=1:raise ValueError(('unrecognized physical owner',old[:90],s.count(old)))
    return s.replace(old,new)
def read(path):return json.loads(path.read_text())
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def build(root):
    exp=root/'bonded-equilibrium/bed-stop';old=root/'material-direction/release';src=old/'build';out=exp/'build';out.mkdir(exist_ok=True)
    failure=read(root/'bonded-equilibrium/three/failure.json')
    if failure['matter_diagnostics'][3]!=-912 or failure['segment_start']!=0:raise ValueError('Different integrated failure')
    if not (out/'src').exists():subprocess.run(['cp','-cR',str(src/'src'),str(out/'src')],check=True)
    name='src/metal/contact.metalinc';shader=(src/name).read_text();tag='kernel void nm_fem_registered_bed_residual('
    a,b=shader.split(tag);b=replace(b,'    device float4* candidateHistories [[buffer(12)]],','''    device float4* candidateHistories [[buffer(12)]],
    device atomic_uint* firstBedViolation [[buffer(13)]],
    constant uint& qualifyNewton [[buffer(14)]],''')
    b=replace(b,'    if (!finalPass) return;\n    // An active dual','    // An active dual')
    site='    tangent[pairID] = float4(gap,margin,row.impulse.z,feasible ? 0.f : 1.f);'
    b=replace(b,site,'''    if (!finalPass) {
        if (qualifyNewton && !feasible)
            atomic_fetch_min_explicit(firstBedViolation,pairID,memory_order_relaxed);
        return;
    }
'''+site)
    shader=a+'''// One group's published support predicate is sufficient for this single-environment solver.
kernel void nm_fem_registered_release_bed_begin_gate(
    device const atomic_uint* firstBedViolation [[buffer(0)]],
    device float4* measurement [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i==0 && atomic_load_explicit(firstBedViolation,memory_order_relaxed)!=0xffffffffu)
        measurement[0].z=0.f;
}
'''+tag+b
    dst=out/name;dst.unlink(missing_ok=True);dst.write_text(shader)
    def host(code):
        k='            "nm_fem_registered_release_reset",'
        code=replace(code,k,k+'\n            "nm_fem_registered_release_bed_begin_gate",')
        k='            dispatchThreads("nm_fem_registered_bed_residual", pairTotal, [&] {'
        code=replace(code,k,'''            if (state.releaseVolumeMode && !finalPass) {
                // The loaded-volume certificate uses this word later and resets it itself.
                dispatchThreads("nm_fem_registered_release_reset",1u,[&] {
                    [encoder setBuffer:state.releaseFailingRow offset:0 atIndex:0];
                });
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
'''+k)
        k='                [encoder setBuffer:state.contactHistoriesCandidate offset:0u atIndex:12u];'
        code=replace(code,k,k+'''
                [encoder setBuffer:state.releaseVolumeMode?state.releaseFailingRow:state.dummy offset:0 atIndex:13];
                [encoder setBytes:&state.releaseVolumeMode length:sizeof(state.releaseVolumeMode) atIndex:14];''')
        k='                dispatchGroups32("nm_fgmres_begin_distributed_finish",'
        code=replace(code,k,'''                if (state.releaseVolumeMode && restartCycle==0u) {
                    dispatchThreads("nm_fem_registered_release_bed_begin_gate",1u,[&] {
                        [encoder setBuffer:state.releaseFailingRow offset:0 atIndex:0];
                        [encoder setBuffer:state.fgmresMeasurementPartials offset:0 atIndex:1];
                    });
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
'''+k)
        code=replace(code,'            if (begin)\n                if (state.releaseVolumeMode','            if (begin) {\n                if (state.releaseVolumeMode')
        code=replace(code,'environments, bindBeginFinish);\n            else','environments, bindBeginFinish);\n            } else')
        k='            candidate->releaseVolumeMode=unsigned(value[0]-\'0\');'
        code=replace(code,k,k+'''
            if (!candidate->femDistributedGlobalMeasurementsEnabled) {
                diagnostics.message="Registered coupled bed requires distributed Newton measurements";return diagnostics;
            }''')
        k='ull^candidate->releaseVolumeMode);'
        return replace(code,k,'ull^'+str(int(sha(dst)[:16],16))+'ull^candidate->releaseVolumeMode);')
    for n in ('runtime.mm','app-runtime.mm'):(out/n).write_text(host((src/n).read_text()))
    old_cmds=read(src/'receipt.json')['commands'];cmds=[[v.replace(str(src)+'/',str(out)+'/') for v in c] for c in old_cmds[:5]]
    for cmd in cmds:subprocess.run(cmd,check=True)
    original=root/'bonded-equilibrium/owner';dest=exp/'owner'
    if not dest.exists():subprocess.run(['cp','-cR',str(original),str(dest)],check=True)
    ident=read(original/'identities.json');parent=read(old/'parent-2/launch.json')
    cmd=[v.replace(str(original)+'/',str(dest)+'/').replace(str(src)+'/',str(out)+'/') for v in ident['commands'][1]]
    subprocess.run(cmd,cwd=parent['cwd'],check=True)
    exe=dest/'numi-human-native';h=sha(exe);ident.update(binary=h,binary_sha256=h,binary_path=str(exe));ident['commands'][1]=cmd;write(dest/'identities.json',ident)
    for key,n in [('NUMI_MATTER_METALLIB_OVERRIDE','NumiMatter.metallib'),('DYLD_LIBRARY_PATH','libmetalrobo.dylib')]:
        if parent['pins'].pop(str(src/n))!=sha(src/n):raise ValueError('Parent physics library differs')
        parent['environment'][key]=str(out if key=='DYLD_LIBRARY_PATH' else out/n);parent['pins'][str(out/n)]=sha(out/n)
    p=exp/'parent';p.mkdir(exist_ok=True);write(p/'launch.json',parent)
    write(out/'receipt.json',dict(parent=sha(src/'receipt.json'),shader=sha(dst),host=sha(out/'runtime.mm'),metallib=sha(out/'NumiMatter.metallib'),lib=sha(out/'libmetalrobo.dylib'),native=h,commands=cmds+[cmd]))
if __name__=='__main__':build(Path(sys.argv[1]).resolve())
