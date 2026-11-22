"""Add a GPU-owned active set sourced from the existing full-trial material validation."""
import hashlib,json,subprocess,sys
from pathlib import Path
from material_context import sha
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def replace(s,a,b):
    if s.count(a)!=1:raise ValueError(('unexpected source site',a[:80],s.count(a)))
    return s.replace(a,b)
def build(case):
    exp=case/'dynamic';out=exp/'build';out.mkdir(exist_ok=True)
    src=case/'build';parent=case/'quantized/build'
    if not (out/'src').exists():subprocess.run(['cp','-cR',parent/'src',out/'src'],check=True)
    f='src/metal/fem.metalinc';text=(parent/f).read_text()
    start=text.index('kernel void nm_fem_registered_volume_direction(')
    kernel=text[start:text.index('\n}',start)+2]
    dynamic=replace(kernel,'nm_fem_registered_volume_direction','nm_fem_registered_volume_active')
    dynamic=replace(dynamic,'constant uint* rows [[buffer(11)]]','device const uint* rows [[buffer(11)]]')
    dynamic=replace(dynamic,'    uint row [[thread_position_in_grid]]) {','    device atomic_uint* claims [[buffer(13)]],\n    uint row [[thread_position_in_grid]]) {')
    key='    const NMTetrahedronGPU t=environmentTetrahedron(d,tets,0u,rows[row]);'
    dynamic=replace(dynamic,key,'''    if(rows[row]==0xffffffffu)return;
'''+key+'''
    for(uint k=0;k<4;++k)
        if(atomic_load_explicit(&claims[t.nodes[k]],memory_order_relaxed)!=rows[row])return;''')
    dst=out/f;dst.unlink(missing_ok=True)
    dst.write_text(replace(text,kernel,kernel+'\n'+(exp/'hints.metalinc').read_text()+'\n'+dynamic))
    seed=read(src/'receipt.json');row=seed['rows'][0][0]
    # The declared opt-in Human parent already authenticates these native rows.
    def patch(code):
        member='    bool volumeDirection = false;'
        s=replace(code,member,member+'''
    id<MTLBuffer> volumeHints=nil, volumeClaims=nil;
    std::uint32_t volumeOwner=0u, volumeTileCount=0u;''')
        key='                candidate->volumeDirection=true;'
        s=replace(s,key,key+f'''
                candidate->volumeOwner=world.fem.tetrahedra[{row}u].identity.y;
                const auto& object=world.objects[candidate->volumeOwner];
                candidate->volumeTileCount=(object.elementCount+255u)/256u;
                if(!candidate->volumeTileCount || object.flags & NM_OBJECT_MIXED_FEM){{
                    diagnostics.message="registered material direction requires a nonmixed volume";return diagnostics;
                }}''')
        key='                    candidate->bedDualLinearization=privateScratch'
        s=replace(s,key,'''                    if(candidate->volumeDirection){
                        candidate->volumeHints=privateScratch<std::uint32_t>(candidate->device,candidate->volumeTileCount,valid,candidate->residentBytes);
                        candidate->volumeClaims=privateScratch<std::uint32_t>(candidate->device,world.dispatch.femNodeCount,valid,candidate->residentBytes);
                    }
'''+key)
        key='            "nm_fem_registered_volume_direction",'
        s=replace(s,key,'''            "nm_fem_registered_volume_hints",
            "nm_fem_registered_volume_claim",
            "nm_fem_registered_volume_active",
'''+key)
        key='if(!candidate->volumeDirection && !std::strcmp(name,"nm_fem_registered_volume_direction"))continue;'
        s=replace(s,key,'if(!candidate->volumeDirection && std::string_view(name).starts_with("nm_fem_registered_volume_"))continue;')
        static=s[s.index('                if(state.volumeDirection){'):s.index('                const std::uint32_t collapseCompletedBacktracking =')]
        bindings=static[static.index('                        setDispatch();'):static.index('                        [encoder setBytes:rows')]
        s=replace(s,static,'')
        key='                resetFreshDirectionCommonAlpha();\n                if (state.unilateralBed) {'
        s=replace(s,key,'''                resetFreshDirectionCommonAlpha();
                if(state.volumeDirection){
                    if(!state.femParallelBacktrackingEnabled || !state.femParallelBacktrackingPartials ||
                       (state.femBacktrackingConstitutiveValidityEnabled && !state.femParallelConstitutiveBacktrackingEnabled)){
                        diagnostics.message="registered volume hints require the active parallel geometry owner";return diagnostics;
                    }
                    const auto count=state.volumeTileCount;
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    dispatchThreads("nm_fem_registered_volume_hints",count,[&]{
                        setDispatch();
                        [encoder setBuffer:state.objects offset:0 atIndex:1];
                        [encoder setBuffer:state.femParallelBacktrackingPartials offset:0 atIndex:2];
                        [encoder setBuffer:state.volumeHints offset:0 atIndex:3];
                        [encoder setBuffer:state.volumeClaims offset:0 atIndex:4];
                        [encoder setBuffer:state.femTetrahedraCandidate offset:0 atIndex:5];
                        [encoder setBytes:&state.volumeOwner length:sizeof(state.volumeOwner) atIndex:6];
                        [encoder setBytes:&state.femParallelBacktrackingMaximumTiles length:sizeof(state.femParallelBacktrackingMaximumTiles) atIndex:7];
                        [encoder setBytes:&nonlinearIteration length:sizeof(nonlinearIteration) atIndex:8];
                    });
                    if(nonlinearIteration){
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        dispatchThreads("nm_fem_registered_volume_claim",count,[&]{
                            [encoder setBuffer:state.volumeHints offset:0 atIndex:0];
                            [encoder setBuffer:state.volumeClaims offset:0 atIndex:1];
                            [encoder setBuffer:state.femTetrahedraCandidate offset:0 atIndex:2];
                            [encoder setBytes:&count length:sizeof(count) atIndex:3];
                        });
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        dispatchThreads("nm_fem_registered_volume_active",count,[&]{
'''+bindings+'''                        [encoder setBuffer:state.volumeHints offset:0 atIndex:11];
                            [encoder setBytes:&count length:sizeof(count) atIndex:12];
                            [encoder setBuffer:state.volumeClaims offset:0 atIndex:13];
                        });
                    }
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                if (state.unilateralBed) {''')
        # Fingerprint both the active-set algorithm and the authenticated target.
        fingerprint=int(hashlib.sha256((sha(dst)+str(row)+str(seed['rows'])).encode()).hexdigest()[:16],16)
        mark='candidate->volumeDirection=true;'
        return replace(s,mark,mark+f'\n                candidate->executionFingerprint=mixFingerprint(candidate->executionFingerprint,{fingerprint}ull);')
    for name in ('runtime.mm','app-runtime.mm'):(out/name).write_text(patch((src/name).read_text()))
    cmds=[]
    def run(c,**opts):cmds.append(c);subprocess.run(c,check=True,**opts)
    for cmd in seed['commands'][:5]:
        run([v.replace(str(src)+'/',str(out)+'/') for v in cmd])
    baseline=case/'owner';owner=exp/'owner'
    if not owner.exists():subprocess.run(['cp','-cR',baseline,owner],check=True)
    launch=read(case/'parent/launch.json');env=launch['environment'];meta=read(baseline/'identities.json')
    cmd=[v.replace(str(baseline)+'/',str(owner)+'/').replace(str(src)+'/',str(out)+'/') for v in meta['commands'][1]]
    run(cmd,cwd=launch['cwd'])
    exe=owner/'numi-human-native';h=sha(exe)
    meta.update(binary=h,binary_sha256=h,binary_path=str(exe));meta['commands'][1]=cmd;save(owner/'identities.json',meta)
    for key,name in [('DYLD_LIBRARY_PATH','libmetalrobo.dylib'),('NUMI_MATTER_METALLIB_OVERRIDE','NumiMatter.metallib')]:
        if launch['pins'].pop(str(src/name))!=sha(src/name):raise ValueError('Native library mismatch')
        env[key]=str(out if key=='DYLD_LIBRARY_PATH' else out/name);launch['pins'][str(out/name)]=sha(out/name)
    (exp/'parent').mkdir(exist_ok=True);save(exp/'parent/launch.json',launch)
    save(out/'receipt.json',dict(parent=sha(src/'receipt.json'),source=sha(dst),active=sha(exp/'hints.metalinc'),metallib=sha(out/'NumiMatter.metallib'),library=sha(out/'libmetalrobo.dylib'),owner=h,commands=cmds))
if __name__=='__main__':build(Path(sys.argv[1]).resolve())
