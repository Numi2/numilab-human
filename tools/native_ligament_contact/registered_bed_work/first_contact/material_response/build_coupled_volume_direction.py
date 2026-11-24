"""Compile a sparse, race-free volume search direction for adjacent active material rows."""
import json,os,re,shutil,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from compile_connected_joint import digest

def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unexpected source near '+old[:100])
    return text.replace(old,new)

def build(job,out):
    out.mkdir();parent=out.parent;runtime=parent/'binding/runtime';base=job/'persistent-volume-witness'
    source=base/'metal/src';dest=out/'metal/src';(dest/'metal').mkdir(parents=True)
    for f in source.iterdir():
        if f.name=='metal':continue
        if f.name=='matter_kernels.metal':shutil.copy2(f,dest/f.name)
        else:(dest/f.name).symlink_to(f.resolve(),target_is_directory=f.is_dir())
    for f in (source/'metal').iterdir():
        if f.name!='fem.metalinc':(dest/'metal'/f.name).symlink_to(f.resolve())
    rt=out/'runtime';rt.mkdir()
    for n in ('bed_response.inc','app-runtime.mm.o'):os.link(runtime/n,rt/n)
    host=(runtime/'runtime.mm').read_text();shader=(source/'metal/fem.metalinc').read_text()
    original=shader
    # Distinct entry points fail closed when a shader and host from different generations are paired.
    for k in ('hints','claim','active'):
        old='nm_fem_registered_volume_'+k;new=old+'_incident'
        if host.count('"'+old+'"')!=2:raise ValueError('Missing named runtime pipeline '+old)
        host=host.replace('"'+old+'"','"'+new+'"')
        shader=replace_once(shader,'kernel void '+old+'(','kernel void '+new+'(')
    host=replace_once(host,'candidate->device,candidate->volumeTileCount,valid,candidate->residentBytes);',
                      'candidate->device,candidate->volumeTileCount+33u,valid,candidate->residentBytes);')
    host=replace_once(host,'candidate->volumeDirection=true;', 'candidate->volumeDirection=true;\n                candidate->executionFingerprint=mixFingerprint(candidate->executionFingerprint,0x636f6c6f72737472ull);')
    shader=replace_once(shader,'const auto object=objects[owner];\n    if(tile>=','''const auto object=objects[owner];
    const uint tileCount=(object.elementCount+255u)/256u;
    if(tile==0u)atomic_store_explicit((device atomic_uint*)(rows+tileCount),0u,memory_order_relaxed);
    if(tile>=''')
    begin=shader.index('kernel void nm_fem_registered_volume_claim_incident')
    end=shader.index('kernel void nm_fem_registered_volume_active_incident',begin)
    claim=shader[begin:end].replace('device const uint* rows','device uint* rows')
    claim=replace_once(claim,'const uint4 id=tets[e].nodes;', '''const uint index=atomic_fetch_add_explicit((device atomic_uint*)(rows+count),1u,memory_order_relaxed);
    if(index<32u)rows[count+1u+index]=tile;
    const uint4 id=tets[e].nodes;''')
    shader=shader[:begin]+claim+shader[end:]
    begin=shader.index('kernel void nm_fem_registered_volume_active_incident')
    end=shader.index('inline MRCompensatedScalar registeredNeg',begin)
    kernel=shader[begin:end];header,body=kernel.split(' {\n',1)
    helper=re.sub(r' \[\[[^\]]+\]\]','',header).replace('kernel void nm_fem_registered_volume_active_incident','inline void registeredVolumeCellDirection')
    helper=replace_once(helper,'uint row)','uint row, bool allNodes)')
    body=replace_once(body,'owned[k]=atomic_load_explicit','owned[k]=allNodes || atomic_load_explicit')
    header=replace_once(header,'uint row [[thread_position_in_grid]])','''uint row [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]],
    uint width [[threads_per_threadgroup]])''')
    call='registeredVolumeCellDirection(d,micro,objects,materials,accepted,candidate,direction,tets,schedule,adaptive,states,rows,count,claims,releaseFloors,releaseMode,'
    sparse='''
    const uint live=rows[count];
    if(live>32u){
        CALLrow,false);
        return;
    }
    if(live==0u || group!=0u)return;
    threadgroup uint sorted[32];
    threadgroup uint color[32];
    threadgroup uint waves;
    if(lane==0u){
        waves=0;
        for(uint i=0;i<live;++i){
            const uint tile=rows[count+1u+i];
            uint at=i;
            while(at>0u && sorted[at-1u]>tile){sorted[at]=sorted[at-1u];--at;}
            sorted[at]=tile;
        }
        for(uint i=0;i<live;++i){
            const uint4 nodes=tets[rows[sorted[i]]].nodes;
            uint used=0;
            for(uint j=0;j<i;++j){
                const uint4 other=tets[rows[sorted[j]]].nodes;
                if(any(nodes==other.x)||any(nodes==other.y)||any(nodes==other.z)||any(nodes==other.w))
                    used|=1u<<color[j];
            }
            color[i]=ctz(~used);
            waves=max(waves,color[i]+1u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // No node is read concurrently with another row's write within a color.
    // Only solver directions change. Exact swept geometry and the freshly
    // assembled nonlinear residual still decide every physical update.
    for(uint wave=0;wave<waves;++wave){
        for(uint i=lane;i<live;i+=width){
            if(color[i]==wave){ CALLsorted[i],true); }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
'''.replace('CALL',call)
    shader=shader[:begin]+helper+' {\n'+body+header+' {\n'+sparse+'\n'+shader[end:]
    (dest/'metal/fem.metalinc').write_text(shader);(rt/'runtime.mm').write_text(host)
    hostcmds=json.loads((parent/'binding/build.json').read_bytes())['commands']
    metalcmds=json.loads((base/'build.json').read_bytes())['commands'][:2]
    def remap(cmd,source,destination):
        return [s.replace(str(source)+'/',str(destination)+'/') for s in cmd]
    cpu=[remap(hostcmds[i],runtime,rt)for i in (0,2)]
    gpu=[remap(c,base/'metal',out/'metal') for c in metalcmds]
    def compile_cmds(item):
        label,commands=item
        with (out/(label+'.log')).open('w') as log:
            for cmd in commands:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(compile_cmds,[('host',cpu),('metal',gpu)]))
    report=dict(parent_runtime=digest(runtime/'libmetalrobo.dylib'),runtime=digest(rt/'libmetalrobo.dylib'),
                parent_shader=digest(base/'metal/NumiMatter.metallib'),shader=digest(out/'metal/NumiMatter.metallib'),
                host_source=digest(rt/'runtime.mm'),shader_source=digest(dest/'metal/fem.metalinc'),
                additional_scratch_bytes=132,commands=cpu+gpu)
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':build(*(Path(s).resolve() for s in sys.argv[1:]))
