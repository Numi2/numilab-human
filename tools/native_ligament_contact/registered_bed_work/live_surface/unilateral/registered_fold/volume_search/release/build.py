"""Place the native terminal loaded-volume check before the shared physical commit."""
from pathlib import Path
import json,subprocess,sys
from material_context import sha
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sub(s,a,b):
    if s.count(a)!=1:raise ValueError(('unknown site',a[:70],s.count(a)))
    return s.replace(a,b)
def build(case):
    exp=case/'release';out=exp/'build';out.mkdir(exist_ok=True)
    old=case/'dynamic';src=old/'build';shader=case/'coupled/build'
    if not (out/'src').exists():subprocess.run(['cp','-cR',shader/'src',out/'src'],check=True)
    name='src/metal/fem.metalinc';s=(shader/name).read_text()
    k='kernel void nm_fem_registered_volume_active(';before,after=s.split(k)
    site='    device atomic_uint* claims [[buffer(13)]],'
    after=sub(after,site,site+'\n    device const float4* releaseFloors [[buffer(14)]],\n    constant uint& releaseMode [[buffer(15)]],')
    after=sub(after,'max(d.numericalLimits.z,mat.validity.x)','max(max(d.numericalLimits.z,mat.validity.x),releaseMode==2u?releaseFloors[rows[row]].z:0.f)')
    s=before+k+after
    # Apply the additional bound at the existing current full-trial validator;
    # the final independent pass uses physical edge determinants, not inverse-rest arithmetic.
    tag='kernel void nm_fem_parallel_backtracking_tiles(';head,rest=s.split(tag);body,tail=rest.split('kernel void nm_fem_parallel_backtracking_finalize(')
    site='    constant uint& validateConstitutiveStress [[buffer(20)]],'
    body=sub(body,site,site+'\n    device const float4* releaseFloors [[buffer(21)]],\n    constant uint& useReleaseBounds [[buffer(22)]],')
    body=sub(body,'material.validity.x);','max(material.validity.x,useReleaseBounds?releaseFloors[object.elementOffset+localElement].z:0.f));')
    s=head+tag+body+'kernel void nm_fem_parallel_backtracking_finalize('+tail
    end='} // namespace numi_matter_metal';s=s[:s.rindex(end)]+(exp/'kernel.metalinc').read_text()+'\n'+end+'\n'
    dst=out/name;dst.unlink(missing_ok=True);dst.write_text(s)
    meta=read(src/'receipt.json');parent=read(case/'coupled/parent/launch.json')
    def patch(code):
        k='    bool volumeDirection = false;'
        s=sub(code,k,k+'\n    std::uint32_t releaseVolumeMode=0u;\n    id<MTLBuffer> releaseFloors=nil, releaseFailingRow=nil;')
        k='        candidate->acceptedStateProofProgramFingerprint ='
        s=sub(s,k,'''        if(const char* value=std::getenv("NM_FEM_REGISTERED_RELEASE_BOUNDS")){
            if((std::strcmp(value,"1")&&std::strcmp(value,"2")) || !candidate->volumeDirection){
                diagnostics.message="registered release bounds require the qualified volume and contact owner";return diagnostics;
            }
            candidate->releaseVolumeMode=unsigned(value[0]-'0');
            candidate->executionFingerprint=mixFingerprint(candidate->executionFingerprint,'''+str(int(sha(dst)[:16],16))+'''ull^candidate->releaseVolumeMode);
        }
'''+k)
        k='                    if(candidate->volumeDirection){'
        s=sub(s,k,(exp/'prepare.inc').read_text()+'\n'+k)
        k='            "nm_fem_registered_volume_active",'
        s=sub(s,k,k+'\n            "nm_fem_registered_release_reset",\n            "nm_fem_registered_release_check",\n            "nm_fem_registered_release_finish",')
        k='        for (const char* name : kernelNames) {'
        s=sub(s,k,k+'\n            if(!candidate->releaseVolumeMode && std::string_view(name).starts_with("nm_fem_registered_release_")) continue;')
        k='                            [encoder setBuffer:state.volumeClaims offset:0 atIndex:13];'
        s=sub(s,k,k+'\n                            [encoder setBuffer:state.releaseFloors ? state.releaseFloors : state.femSolution offset:0 atIndex:14];\n                            [encoder setBytes:&state.releaseVolumeMode length:sizeof(state.releaseVolumeMode) atIndex:15];')
        k='                        [encoder setBytes:&validateStress length:sizeof(validateStress)\n                                  atIndex:20u];'
        s=sub(s,k,k+'''
                        [encoder setBuffer:state.releaseFloors ? state.releaseFloors : state.femSolution offset:0 atIndex:21];
                        [encoder setBytes:&state.releaseVolumeMode length:sizeof(state.releaseVolumeMode) atIndex:22];''')
        k='            if (!encodeFEMNewtonCertificateTrace()) {'
        s=sub(s,k,'''            if(state.releaseVolumeMode){
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                dispatchThreads("nm_fem_registered_release_reset",1u,[&]{
                    [encoder setBuffer:state.releaseFailingRow offset:0 atIndex:0];
                });
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                dispatchThreads("nm_fem_registered_release_check",state.dispatch.tetrahedronCount,[&]{
                    setDispatch();
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0 atIndex:1];
                    [encoder setBuffer:state.femCandidate offset:0 atIndex:2];
                    [encoder setBuffer:state.releaseFloors offset:0 atIndex:3];
                    [encoder setBuffer:state.releaseFailingRow offset:0 atIndex:4];
                });
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                dispatchThreads("nm_fem_registered_release_finish",1u,[&]{
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0 atIndex:0];
                    [encoder setBuffer:state.femCandidate offset:0 atIndex:1];
                    [encoder setBuffer:state.releaseFloors offset:0 atIndex:2];
                    [encoder setBuffer:state.releaseFailingRow offset:0 atIndex:3];
                    [encoder setBuffer:state.statuses offset:0 atIndex:4];
                });
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
'''+k)
        return s
    for n in ('runtime.mm','app-runtime.mm'):(out/n).write_text(patch((src/n).read_text()))
    cmds=[]
    def run(c,**opts):cmds.append(c);subprocess.run(c,check=True,**opts)
    for c in meta['commands'][:5]:run([v.replace(str(src)+'/',str(out)+'/') for v in c])
    orig=old/'owner';owner=exp/'owner'
    if not owner.exists():subprocess.run(['cp','-cR',orig,owner],check=True)
    ident=read(orig/'identities.json');cmd=[v.replace(str(orig)+'/',str(owner)+'/').replace(str(src)+'/',str(out)+'/') for v in ident['commands'][1]]
    run(cmd,cwd=parent['cwd']);exe=owner/'numi-human-native';h=sha(exe)
    ident.update(binary=h,binary_sha256=h,binary_path=str(exe));ident['commands'][1]=cmd;write(owner/'identities.json',ident)
    env=parent['environment']
    for key,n,oldpath in [('NUMI_MATTER_METALLIB_OVERRIDE','NumiMatter.metallib',shader),('DYLD_LIBRARY_PATH','libmetalrobo.dylib',src)]:
        if parent['pins'].pop(str(oldpath/n))!=sha(oldpath/n):raise ValueError('Wrong physical owner')
        env[key]=str(out if key=='DYLD_LIBRARY_PATH' else out/n);parent['pins'][str(out/n)]=sha(out/n)
    for mode in (1,2):
        env['NM_FEM_REGISTERED_RELEASE_BOUNDS']=str(mode)
        p=exp/f'parent-{mode}';p.mkdir(exist_ok=True);write(p/'launch.json',parent)
    write(out/'receipt.json',dict(source=sha(dst),host=sha(out/'runtime.mm'),kernel=sha(exp/'kernel.metalinc'),prep=sha(exp/'prepare.inc'),metallib=sha(out/'NumiMatter.metallib'),lib=sha(out/'libmetalrobo.dylib'),binary=h,commands=cmds))
if __name__=='__main__':build(Path(sys.argv[1]).resolve())
