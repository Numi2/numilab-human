"""Keep loaded pelvic slots active without consuming discovery queue space a second time."""
import json
import subprocess
import sys
from pathlib import Path
from emit_skin_source_registration import sha


def replace(src, old, new, count=1):
    if src.count(old)!=count:
        raise ValueError('unexpected registered solver source: '+old)
    return src.replace(old,new)


def main(src, original, app, out):
    identity = json.loads((src/'identities.json').read_text())
    for file in (src/'numi-human-native',src/'metalrobo/HumanRestingSkin.metallib'):
        if identity[str(file)]!=sha(file):
            raise ValueError('registered parent mismatch')
    out.mkdir(exist_ok=False)
    for file in src.iterdir():
        if file.is_file() and file.suffix in ('.hpp','.metal','.metalinc','.mm','.inc'):
            (out/file.name).hardlink_to(file)
    libs=out/'metalrobo';libs.mkdir()
    (libs/'MetalRobo.metallib').hardlink_to(src/'metalrobo/MetalRobo.metallib')
    def edit(name, changes):
        file=out/name; text=file.read_text()
        for a,b,*count in changes: text=replace(text,a,b,*count)
        file.unlink();file.write_text(text)
    edit('pelvic_motion_registry.metalinc',[
        ('device atomic_uint* metrics,device uint2* queue){','device atomic_uint* metrics,device uint2* queue,device const uint2* mapped,device atomic_uint* live){'),
        ('    uint i=atomic_fetch_add_explicit(metrics+side', '''    // A bound dynamic facet needs activation, with its exact P1 identities intact.
    // Its slot remains stable for existing pressures and speculative rollback.
    for(uint j=0;j<256;++j){
        uint slot=side*256+j;
        if(all(mapped[slot]==key+1u)){
            atomic_store_explicit(live+slot,1u,memory_order_relaxed);
            return;
        }
    }
    uint i=atomic_fetch_add_explicit(metrics+side'''),
        ('else atomic_store_explicit(metrics+2,1u,memory_order_relaxed);','else atomic_fetch_or_explicit(metrics+2,1u,memory_order_relaxed);'),
        ('if(slot==256){atomic_store_explicit(metrics+2,1u,memory_order_relaxed);','if(slot==256){atomic_fetch_or_explicit(metrics+2,2u,memory_order_relaxed);'),
        ('if(i<3)atomic_store_explicit(metrics+i,0u','if(i<5)atomic_store_explicit(metrics+i,0u'),
        ('    uint count=min(256u,atomic_load_explicit(metrics+side,memory_order_relaxed));','''    uint demand=atomic_load_explicit(metrics+side,memory_order_relaxed);
    if(!lane)atomic_fetch_max_explicit(metrics+3+side,demand,memory_order_relaxed);
    uint count=min(256u,demand);''')])
    edit('pelvic_end_geometry.metalinc',[
        ('device uint2* queue [[buffer(13)]],uint i','device uint2* queue [[buffer(13)]],device const uint2* mapped [[buffer(14)]],uint i'),
        ('f.w,registered,registryMetrics,queue);','f.w,registered,registryMetrics,queue,mapped,live);'),
        ('registryMetrics [[buffer(6)]],uint lane','registryMetrics [[buffer(6)]],device uint4* demandTrace [[buffer(7)]],uint lane'),
        ('uint4 result=group[0];if(pelvicMotionRegistry && atomic_load_explicit(registryMetrics+2,memory_order_relaxed))result.w|=2u;if(d.x<d.z)trace[d.x]=result;','''uint4 result=group[0];
        if(pelvicMotionRegistry){
            uint flags=atomic_load_explicit(registryMetrics+2,memory_order_relaxed);
            result.w|=(flags<<1);
            if(d.x<d.z)demandTrace[d.x]=uint4(max(atomic_load_explicit(registryMetrics,memory_order_relaxed),atomic_load_explicit(registryMetrics+3,memory_order_relaxed)),
                max(atomic_load_explicit(registryMetrics+1,memory_order_relaxed),atomic_load_explicit(registryMetrics+4,memory_order_relaxed)),flags,result.x);
        }
        if(d.x<d.z)trace[d.x]=result;''')])
    edit('NumiHumanPelvicContact.hpp',[
        ('motionMetrics=nil,motionQueue=nil','motionMetrics=nil,demandTrace=nil,motionQueue=nil'),
        ('motionMetrics=alloc(12,true);','motionMetrics=alloc(20,true);demandTrace=alloc(std::max(1u,countTrace)*16,true);'),
        ('bind(13,motionQueue);','bind(13,motionQueue);bind(14,motionKeys);',2),
        ('if constexpr(pelvicMotionRegistry)bind(6,motionMetrics);run(1,128);','if constexpr(pelvicMotionRegistry){bind(6,motionMetrics);bind(7,demandTrace);}run(1,128);'),
        ('            auto keys=static_cast<const unsigned*>(savedKeys.contents);','''            auto demand=static_cast<const unsigned*>(demandTrace.contents);
            std::ofstream demands(std::filesystem::path(file).parent_path()/"registry-demand.csv");
            demands<<"candidate_step,right_new_pairs,left_new_pairs,overflow_flags,final_transverse_pairs\\n";
            unsigned end=std::min({encodedSteps,countTrace,1u+unsigned(static_cast<const float*>(accepted.contents)[0])});
            for(unsigned i=0;i<end;++i)demands<<i<<','<<demand[4*i]<<','<<demand[4*i+1]<<','<<demand[4*i+2]<<','<<demand[4*i+3]<<'\\n';
            auto keys=static_cast<const unsigned*>(savedKeys.contents);''')])
    shader=json.loads((original/'build-commands.json').read_text())[:2]
    host=json.loads((src/'build-commands.json').read_text())[-2:]
    cmds=[[arg.replace(str(original)+'/',str(out)+'/') for arg in cmd] for cmd in shader]
    cmds.extend([[arg.replace(str(src)+'/',str(out)+'/') for arg in cmd] for cmd in host])
    (out/'build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    for cmd in cmds:subprocess.run(cmd,cwd=app/'build-native',check=True)
    files=[out/name for name in ('pelvic_motion_registry.metalinc','pelvic_end_geometry.metalinc','NumiHumanPelvicContact.hpp','metalrobo/HumanRestingSkin.metallib','numi-human-native')]
    (out/'identities.json').write_text(json.dumps({str(f):sha(f) for f in files},indent=2)+'\n')


if __name__=='__main__':main(*(Path(a).resolve() for a in sys.argv[1:]))
